import numpy as np
import os
import argparse
import itertools
import pickle
import time
import warnings
from instances import gaussian_instance
from carving.carving_class import gaussian_carving

MACHINE_EPS = np.finfo(np.float64).eps

def run(seed, nsample, rho1, signal, equi, path, target_d=None, s=10, scale=False, sampling_only=False, deterministic=False):
    n = 300
    n1 = int(n * rho1)
    ntune = 100
    p = 300
    if equi:
        rho = .9
    else:
        rho = .9
    X, Y, beta = gaussian_instance(n=n+ntune, p=p, s=s, signal=np.sqrt(2 * np.log(p) * signal), rho=rho, sigma=1., random_signs=True, seed=seed, scale=scale, equicorrelated=equi)
    X_tune = X[n:]
    Y_tune = Y[n:]
    X = X[:n]
    Y = Y[:n]

    carving = gaussian_carving(X, Y, n1)
    lbd_list = np.sqrt(2*np.log(p)) / n * np.linspace(.1, 2, 50) 
    if target_d is not None:
        carving.fit(5, target_d=target_d)
    else:
        carving.fit(lbd_list, X_tune=X_tune, Y_tune=Y_tune, max_d=40)
    carving.prepare_inference()
    d = carving.d
    print("selected", d, 'variables')
    if d == 0:
        return
    beta_target = np.linalg.pinv(X[:, carving.E]).dot(X.dot(beta))  
    sig_level = 0.05

    # naive
    naive_pval, naive_ci = carving.naive_inference(sig_level)
    naive_covered = (beta_target <= naive_ci[:, 1]) * (beta_target >= naive_ci[:, 0])
    print('naive CI', naive_ci)
    # splitting
    splitting_pval, splitting_ci = carving.splitting_inference(sig_level)
    splitting_covered = (beta_target <= splitting_ci[:, 1]) * (beta_target >= splitting_ci[:, 0])
    print('splitting CI', splitting_ci)
    # selective MLE
    mle_approx_result = carving.approx_mle_inference('selected', sig_level)
    mle_approx_pval = np.array(mle_approx_result['pvalue'])
    mle_approx_ci = np.array(mle_approx_result[['lower_confidence', 'upper_confidence']])
    mle_approx_covered = (beta_target <= mle_approx_ci[:, 1]) * (beta_target >= mle_approx_ci[:, 0])
    mle_approx_time = mle_approx_result['time'].iloc[0]
    print("MLE CI", mle_approx_ci, mle_approx_pval)

    start = time.time()
    mle_sov_result = carving.mle_sov()
    mle_sov_time = time.time() - start
    mle_sov_ci = np.array(mle_sov_result[['lower confidence', 'upper confidence']])
    mle_sov_pval = np.array(mle_sov_result['pvalues'])
    mle_sov_covered = (beta_target <= mle_sov_ci[:, 1]) * (beta_target >= mle_sov_ci[:, 0])

    # exact bivariate pivot
    exact_pval = np.zeros(d)
    exact_ci = np.zeros((d, 2))
    for j in range(d):
        eta = np.eye(d)[j]
        params = carving.prepare_eta(eta)
        exact_pval[j] = carving.exact_bivar_pivot(j, params, 0., True)
        exact_ci[j] = carving.exact_bivar_interval(j, params, sig_level, True)
    exact_covered = (beta_target <= exact_ci[:, 1]) * (beta_target >= exact_ci[:, 0])
    print("exact CI", exact_ci, exact_pval)
    print("length", np.mean(exact_ci[:, 1] - exact_ci[:, 0]))

    # all IS
    n_sov = nsample
    start = time.time()
    allIS_result = carving.sampling_inference(n_sov=n_sov, seed=seed**2)
    allIS_time = time.time() - start
    allIS_ci = np.array(allIS_result[['lower confidence', 'upper confidence']])
    allIS_pval = np.array(allIS_result['pvalue'])
    allIS_covered = (beta_target <= allIS_ci[:, 1]) * (beta_target >= allIS_ci[:, 0])
    print('allIS CI', allIS_result)
    print("length", np.mean(allIS_ci[:, 1] - allIS_ci[:, 0]))

    # deterministic bound
    deter_ci = np.zeros((d, 2))
    deter_pval = np.zeros(d)
    start = time.time()
    if deterministic:
        for j in range(d):
            print(j)
            eta = np.eye(d)[j]
            params = carving.prepare_eta(eta)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                deter_pval[j] = carving.get_pvalue_upper(params, 0., two_sided=True)
                deter_ci[j] = carving.get_CI_deterministic(params, allIS_ci[j], sig_level=sig_level, two_sided=True)
    deter_covered = (beta_target <= deter_ci[:, 1]) & (beta_target >= deter_ci[:, 0])
    deter_time = time.time() - start
    print("deterministic CI", deter_ci)
    print("length", np.mean(deter_ci[:, 1] - deter_ci[:, 0]))

    # unbiased estimator
    E = carving.E
    beta_hat_2 = carving.beta_hat_2

    sov_ci = np.zeros((d, 2))
    sov_pval = np.zeros(d)
    start = time.time()
    for j in range(d):
        print('sov', j)
        eta = np.eye(d)[j]
        params = carving.prepare_eta(eta)
        sov_samples, sov_weights = carving.sample_auxillary(params, 0., 'sov', nsample=n_sov, seed=seed*1000+j+1)
        sov_pval[j] = carving.get_pvalue(params, 0., sov_samples, sov_weights)

        sov_samples, sov_weights = carving.sample_auxillary(params, beta_hat_2[j], 'sov', nsample=n_sov, seed=seed*1000+j+1)
        sov_ci[j] = carving.get_CI(params, sov_samples, beta_hat_2[j], sov_weights, sig_level=sig_level)
    sov_time = (time.time() - start)
    sov_covered = (beta_target <= sov_ci[:, 1]) * (beta_target >= sov_ci[:, 0])
    print('sov CI', sov_ci, sov_pval)

    # Gibbs
    n_gibbs = nsample
    burnin = 20
    gibbs_ci = np.zeros((d, 2))
    gibbs_pval = np.zeros(d)
    start = time.time()
    for j in range(d):
        print('gibbs', j)
        eta = np.eye(d)[j]
        params = carving.prepare_eta(eta)
        gibbs_samples = carving.sample_auxillary(params, 0., 'gibbs', nsample=n_gibbs, seed=j, burnin=burnin, from_mode=True)
        gibbs_pval[j] = carving.get_pvalue(params, 0., gibbs_samples, None)

        gibbs_samples = carving.sample_auxillary(params, beta_hat_2[j], 'gibbs', nsample=n_gibbs, seed=seed*1000+j+1, burnin=burnin, from_mode=True)
        if gibbs_samples is None:
            gibbs_ci[j] = np.nan
        else:
            gibbs_ci[j] = carving.get_CI(params, gibbs_samples, beta_hat_2[j], None, sig_level=sig_level)
    gibbs_time = (time.time() - start) 

    gibbs_covered = (beta_target <= gibbs_ci[:, 1]) * (beta_target >= gibbs_ci[:, 0])
    print('gibbs CI', gibbs_ci)

    pvalues = np.stack([naive_pval, splitting_pval, mle_approx_pval, mle_sov_pval, exact_pval, allIS_pval, gibbs_pval, sov_pval, deter_pval])
    ci = np.stack([naive_ci, splitting_ci, mle_approx_ci, mle_sov_ci, exact_ci, allIS_ci, gibbs_ci, sov_ci, deter_ci])
    covered = np.stack([naive_covered, splitting_covered, mle_approx_covered, mle_sov_covered, exact_covered, allIS_covered, gibbs_covered, sov_covered, deter_covered])
    print(np.mean(covered, 1))

    times = np.stack([mle_approx_time, mle_sov_time, allIS_time, gibbs_time, sov_time, deter_time])
    print('times', times)
    methods = ['naive', 'splitting', 'mle_approx', 'mle_sov', 'exact', 'allIS', 'gibbs', 'sov', 'deterministic']
    results = {'pvalue': pvalues, 'ci': ci, 'covered': covered, 'times': times, 'methods': methods}
    # print(results)
    
    if equi:
        xcov = 'equi'
    else:
        xcov = 'AR'
    if target_d is not None:
        filename = f'gaussian_mle_{n}_{n1}_{p}_s_{s}_targetd_{target_d}_{xcov}_rho_{rho}_signalfac_{signal}_nsample_{nsample}_{seed}'
    else:
        filename = f'gaussian_mle_{n}_{n1}_{p}_s_{s}_ntune_{ntune}_{xcov}_rho_{rho}_signalfac_{signal}_nsample_{nsample}_{seed}'
    if sampling_only:
        filename = filename + '_sampling.pkl'
    else:
        filename = filename + '.pkl'
    
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(results, f)
        
    print("saving results to", filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobid', type=int, default=0)
    parser.add_argument('--target_d', type=int, default=-1)
    parser.add_argument('--s', default=10, type=int)
    parser.add_argument('--rho1', default=.8, type=float)
    parser.add_argument('--equi', action='store_true', default=False)
    parser.add_argument('--sampling_only', action='store_true', default=False)
    parser.add_argument('--vary', default='signal', type=str)
    parser.add_argument('--rootdir', default='')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.rootdir, 'results/carving5'), exist_ok=True)
    l = 0
    m_list = np.arange(9, 10)
    nrep = 200
    target_d = args.target_d
    if target_d < 0:
        target_d = None
    
    if args.vary == 'signal':
        m = 9
        rho1 = 0.8
        s = args.s
        for signal, seed in itertools.product([.6, .9, 1.2], np.arange(nrep)):
            if l == args.jobid:
                nsample = 2**m
                print(signal, rho1, nsample, seed)
                break
            l = l + 1
    
    elif args.vary == 'nsample':
        signal = 0.6
        rho1 = 0.8
        s = args.s
        for m, seed in itertools.product(np.arange(8, 13), np.arange(nrep)):
            if l == args.jobid:
                nsample = 2**m
                print(signal, rho1, nsample, seed)
                break
            l = l + 1
    
    run(int(seed), nsample, rho1=rho1, signal=signal, equi=args.equi, path=os.path.join(args.rootdir, 'results/carving5'), target_d=target_d, s=s, scale=True, sampling_only=args.sampling_only)

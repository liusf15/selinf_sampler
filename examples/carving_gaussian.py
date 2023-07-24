import numpy as np
import os
import argparse
import itertools
import pickle
import yaml
import datetime
import time
import warnings
import git


from examples.instances import gaussian_instance
from src.carving_class import gaussian_carving

MACHINE_EPS = np.finfo(np.float64).eps

def run(config):
    seed = config['seed']
    signal = config['signal']
    n = config['n']
    n1 = int(config['n'] * config['rho1'])
    ntune = config['ntune']
    p = config['p']
    target_d = config['target_d']
    rho = config['rho']
    X, Y, beta = gaussian_instance(n=n+ntune, p=p, s=config['s'], signal=np.sqrt(2 * np.log(p) * signal), rho=rho, sigma=1., random_signs=True, seed=seed, scale=True, equicorrelated=config['cov_x']=='equi')
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
    n_sov = config['nsample']
    start = time.time()
    allIS_result = carving.sampling_inference(n_sov=n_sov, seed=seed**2)
    allIS_time = time.time() - start
    allIS_ci = np.array(allIS_result[['lower confidence', 'upper confidence']])
    allIS_pval = np.array(allIS_result['pvalue'])
    allIS_covered = (beta_target <= allIS_ci[:, 1]) * (beta_target >= allIS_ci[:, 0])
    print('allIS CI', allIS_result)
    print("length", np.mean(allIS_ci[:, 1] - allIS_ci[:, 0]))


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

    pvalues = np.stack([naive_pval, splitting_pval, mle_approx_pval, mle_sov_pval, exact_pval, allIS_pval, gibbs_pval, sov_pval])
    ci = np.stack([naive_ci, splitting_ci, mle_approx_ci, mle_sov_ci, exact_ci, allIS_ci, gibbs_ci, sov_ci])
    covered = np.stack([naive_covered, splitting_covered, mle_approx_covered, mle_sov_covered, exact_covered, allIS_covered, gibbs_covered, sov_covered])
    print(np.mean(covered, 1))

    times = np.stack([mle_approx_time, mle_sov_time, allIS_time, gibbs_time, sov_time])
    print('times', times)
    methods = ['naive', 'splitting', 'mle_approx', 'mle_sov', 'exact', 'allIS', 'gibbs', 'sov']
    results = {'pvalue': pvalues, 'ci': ci, 'covered': covered, 'times': times, 'methods': methods}
    print(results)

    
    with open(config['savename'], 'wb') as f:
        pickle.dump(results, f)

    print("saving results to", config['savename'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--jobid', type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f.read())

    l = 0
    nrep = 200
    m = 9
    nsample = 2**m
    for signal, seed in itertools.product([.6, .9, 1.2], np.arange(nrep)):
        if l == args.jobid:
            print(signal, seed)
            config['seed'] = seed
            config['signal'] = signal
            break
        l = l + 1
    
    path = os.path.join(config['rootdir'], datetime.datetime.now().strftime("%Y_%m_%d"))
    filename = f'carving_gaussian_{config["n"]}_{config["rho1"]}_{config["p"]}_s_{config["s"]}_targetd_{config["target_d"]}_{config["cov_x"]}_rho_{config["rho"]}_nsample_{nsample}_signalfac_{signal}'
    path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)

    repo = git.Repo(search_parent_directories=True)
    config['git_sha'] = repo.head.object.hexsha
    with open(os.path.join(path, 'config.yaml'), 'w', encoding='utf8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    config['savename'] = os.path.join(path, f'seed_{seed}.pkl')
    run(config)

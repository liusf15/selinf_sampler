import numpy as np
import cvxpy as cvx
from scipy.special import ndtr
import pickle
import argparse
import itertools
import os
import pandas as pd
import time
from src.core import st_cdf, sample_sov, Gibbs, gibson_ordering
from examples.instances import HIV_NRTI
from src.carving_class import gaussian_carving

MACHINE_EPS = np.finfo(float).eps

def prepare_parameters():
    drug_class = 'NRTI' # NRTI p=91, NNRTI p=103, PI
    datafile = "https://hivdb.stanford.edu/_wrapper/pages/published_analysis/genophenoPNAS2006/DATA/{}_DATA.txt".format(drug_class)
    X, Y, muts = HIV_NRTI(drug='3TC', datafile=datafile, min_occurrences=11, standardize=True)
    n, p = X.shape
    n1 = int(n * 0.8)
    carving = gaussian_carving(X, Y, n1)
    carving.fit(tune_lambda='fixed_d', target_d=17)
    carving.prepare_inference()
    d = carving.d
    print("selected", d, 'variables')
    print("lambda", carving.lbd)

    b_means = np.zeros((d, d))
    b_covs = np.zeros((d, d, d))
    c1s = np.zeros((d, d))
    c2s = np.zeros(d)
    orderings = np.zeros((d, d), dtype=int)
    for j in range(d):
        eta = np.eye(d)[j]
        params = carving.prepare_eta(eta)
        b_means[j] = params.mu_b_added
        b_covs[j] = params.cov_b
        c1s[j][j] = -params.mu_theta_multi_b[j] / np.sqrt(params.var_theta)
        c2s[j] = (params.theta_hat - params.mu_theta_added) / np.sqrt(params.var_theta)
        ordering, _, a_ord, b_ord, L_ord = gibson_ordering(np.copy(b_covs[j]), np.copy(-b_means[j]), np.ones(d)*np.Inf)  
        orderings[j] = ordering

    matrices = {'b_means': b_means, 'b_covs': b_covs, 'c1s': c1s, 'c2s': c2s, 'ordering': orderings, 'muts': muts[carving.selected]}
    with open(f'examples/hiv_matrices_n_{n}_n1_{n1}_p_{p}_d_{d}.pkl', 'wb') as f:
        pickle.dump(matrices, f)
    print("saved matrices to", f'examples/hiv_matrices_n_{n}_n1_{n1}_p_{p}_d_{d}.pkl')



def run(var_id, Sigma, mu, ordering, c1, c2, seedid, n_sov=2048, rootdir=''):
    seed = seed_seq[seedid]
    d = Sigma.shape[0]

    pvalues = {}
    times = {}

    # Gibbs
    L = np.linalg.cholesky(Sigma)
    eigval, eigvec = np.linalg.eigh(Sigma)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]
    nPC = np.where(np.cumsum(eigval) >= 0.5 * np.sum(eigval))[0][0]
    tmp = np.where(eigval > eigval.mean())[0]
    if len(tmp) > 0:
        nPC = max(nPC, tmp[-1])
    PCs = eigvec[:, :nPC].T
    x = cvx.Variable(d)
    cvxprob = cvx.Problem(cvx.Minimize((1/2) * cvx.quad_form(x, np.eye(d))), [-L @ x <= mu])
    cvxprob.solve()
    initial = x.value
    eta = np.eye(d)[var_id]

    burnin = 20
    n_gibbs = n_sov * 5
    start = time.time()
    samples = Gibbs(mu, Sigma, initial=L @ initial + mu, maxit=n_gibbs+burnin, seed=seed, PC_every=20, eigval=eigval[:nPC], PCs=PCs, eta=None, eta_every=-1)[burnin:]
    times['gibbs'] = time.time() - start
    cdf_ = np.mean(ndtr(samples @ c1 + c2))
    pvalues['gibbs'] = 2 * min(cdf_, 1 - cdf_)

    # one sample for all
    if var_id == 0:
        drug_class = 'NRTI' # NRTI p=91, NNRTI p=103, PI
        datafile = "https://hivdb.stanford.edu/_wrapper/pages/published_analysis/genophenoPNAS2006/DATA/{}_DATA.txt".format(drug_class)
        X, Y, muts = HIV_NRTI(drug='3TC', datafile=datafile, min_occurrences=11, standardize=True)
        n, p = X.shape
        n1 = int(n * 0.8)
        carving = gaussian_carving(X, Y, n1)
        carving.fit(tune_lambda='fixed_d', target_d=17)
        carving.prepare_inference()
        d = carving.d
        params0 = carving.prepare_eta(np.zeros(d))
        start = time.time()
        eta0_samples, eta0_weights = carving.sample_auxillary(params0, 0., 'sov', nsample=n_sov, seed=seed)
        sov_IS_pval = np.zeros(d)
        for j in range(d):
            eta = np.eye(d)[j]
            params = carving.prepare_eta(eta)
            sov_IS_pval[j] = carving.get_pvalue_eta0(params, 0., eta0_samples, eta0_weights, two_sided=True)

    # gibson ordering
    mu_ord = mu[ordering]
    L_ord = np.linalg.cholesky(Sigma[ordering][:, ordering])
    c1_ord = c1[ordering]

    # SOV
    start = time.time()
    samples, weights = sample_sov(-mu_ord, np.ones(d)*np.Inf, L_ord, n_sov, seed, spherical=1)
    times['sov'] = time.time() - start
    samples = samples @ L_ord.T + mu_ord
    cdf_ = np.mean(ndtr(samples @ c1_ord + c2) * weights) / np.mean(weights)
    pvalues['sov'] = 2 * min(cdf_, 1 - cdf_)

    # SOV MC
    start = time.time()
    samples, weights = sample_sov(-mu_ord, np.ones(d)*np.Inf, L_ord, n_sov, seed, spherical=1, rqmc=0)
    times['sov_mc'] = time.time() - start
    samples = samples @ L_ord.T + mu_ord
    cdf_ = np.mean(ndtr(samples @ c1_ord + c2) * weights) / np.mean(weights)
    pvalues['sov_mc'] = 2 * min(cdf_, 1 - cdf_)

    # sov + preint
    start = time.time()
    num, den = st_cdf(mu_ord, L_ord, c1_ord, c2, n_sov, seed, rqmc=True)
    times['sov_preint'] = time.time() - start
    cdf_ = num / den
    if np.isnan(cdf_):
        raise ValueError('cdf is nan')
    pvalues['sov_preint'] = 2 * min(cdf_, 1 - cdf_)

    # preint with reorder
    new_ordering = np.zeros(d, dtype=int)
    msk = np.ones(d, dtype=bool)
    msk[var_id] = False
    ordering_ = gibson_ordering(np.copy(Sigma[msk][:, msk]), np.copy(-mu[msk]), np.ones(d-1)*np.Inf)[0]
    ordering_[ordering_ >= var_id] += 1
    new_ordering[:-1] = ordering_
    new_ordering[-1] = var_id

    mu_ord = mu[new_ordering]
    L_ord = np.linalg.cholesky(Sigma[new_ordering][:, new_ordering])
    c1_ord = c1[new_ordering]

    start = time.time()
    num, den = st_cdf(mu_ord, L_ord, c1_ord, c2, n, seed, rqmc=True)
    times['sov_preint_reorder'] = time.time() - start
    cdf_ = num / den
    if np.isnan(cdf_):
        raise ValueError('cdf is nan')
    pvalues['sov_preint_reorder'] = 2 * min(cdf_, 1 - cdf_)

    results = pd.DataFrame({'pvalues': pvalues, 'times': times})
    savepath = os.path.join(rootdir, f'n_633_n1_{n1}_p_{p}_d_{d}_nsov_{n_sov}_ngibbs_{n_gibbs}_burnin_{burnin}')
    os.makedirs(savepath, exist_ok=True)
    filename = os.path.join(savepath, f'var_{var_id}_seed_{seedid}.csv')
    results.to_csv(filename)
    print('saved to', filename)

    if var_id == 0:
        pd.DataFrame(sov_IS_pval).to_csv(os.path.join(savepath, f'SOV_IS_seed_{seedid}.csv'))

if __name__ == "__main__":
    p = 91
    d = 17
    n = 633
    n1 = int(n * .8)
    if not os.path.exists(f'examples/hiv_matrices_n_{n}_n1_{n1}_p_{p}_d_{d}.pkl'):
        prepare_parameters()
    with open(f'examples/hiv_matrices_n_{n}_n1_{n1}_p_{p}_d_{d}.pkl', 'rb') as f:
        matrices = pickle.load(f)

    print(matrices['muts'].shape)
    parser = argparse.ArgumentParser()
    parser.add_argument('date', type=str)
    parser.add_argument('--jobid', type=int, default=0)
    parser.add_argument('--rootdir', type=str, default='')
    args = parser.parse_args()

    nrep = 50
    with open('../entropy.txt', 'r') as f:
        entropy = int(f.read())
    seed_seq = np.random.SeedSequence(entropy).generate_state(nrep, dtype=np.uint32)

    l = 0
    for var_id, seedid in itertools.product(range(d), range(nrep)):
        if l == args.jobid:
            print('variable', var_id, 'seed', seedid)
            break
        l += 1
    Sigma = matrices['b_covs'][var_id]
    mu = matrices['b_means'][var_id]
    c1 = matrices['c1s'][var_id]
    c2 = matrices['c2s'][var_id]
    a = np.copy(-mu)
    c1_ = c1[var_id]
    ordering = matrices['ordering'][var_id]

    path = os.path.join(args.rootdir, args.date)
    
    run(var_id, Sigma, mu, ordering, c1, c2, seedid, n_sov=2048, rootdir=path)


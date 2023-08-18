import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal 
from numpy.linalg import inv
from tqdm import trange
from scipy.special import owens_t, ndtr, ndtri
from scipy.optimize import minimize
from scipy.optimize import root
from autograd import grad, jacobian
import autograd.numpy as agnp
from autograd.scipy.stats import norm as agnorm
import math
import warnings

import pyximport
pyximport.install()
from src.cython_core import sample_sov, Gibbs, joint_cdf_bivnormal, st_cdf


MACHINE_EPS = np.finfo(np.float64).eps

def gibson_ordering(Sigma, a, b):
    d = len(a)
    orders = []
    num_ = 0
    den_ = 0
    exp_y = np.zeros(d)
    L = np.zeros((d, d))
    orders = np.arange(d)
    for j in range(d):
        num_ = L[:, :j] @ exp_y[:j]
        den_ = np.diag(Sigma) - np.sum(L[:, :j]**2, 1)
        den_[den_ < 0] = 0
        lowers = (a - num_) / np.sqrt(den_)
        uppers = (b - num_) / np.sqrt(den_)
        exp_len = ndtr(uppers) - ndtr(lowers)
        new_j = np.argmin(exp_len[j:]) + j
        # swap
        Sigma[:, [j, new_j]] = Sigma[:, [new_j, j]]
        Sigma[[j, new_j], :] = Sigma[[new_j, j], :]
        L[:, [j, new_j]] = L[:, [new_j, j]]
        L[[j, new_j], :] = L[[new_j, j], :]
        a[[j, new_j] ] = a[[new_j, j]]
        b[[j, new_j] ] = b[[new_j, j]]
        orders[[j, new_j]] = orders[[new_j, j]]
        # compute j-th column of L
        L[j, j] = np.sqrt(Sigma[j, j] - np.sum(L[j, :j]**2))
        L[j+1:, j] = (Sigma[j+1:, j] - L[j+1:, :j] @ L[j, :j]) / L[j, j]
        if np.any(np.isnan(L)):
            assert False, 'L is nan'
        ta_j = lowers[j]
        tb_j = uppers[j]
        exp_y[j] = (np.exp(-ta_j**2/2) - np.exp(-tb_j**2/2)) / (ndtr(tb_j) - ndtr(ta_j)) / np.sqrt(2*np.pi)
    return orders, Sigma, a, b, L

def sample_sov_reorder(mean, L, nsample, seed):
    """
    sample x ~ N(mean, L @ L') | x_j > 0 \forall 1\leq j\leq d
    U: (n, d) or (n, d-1)
    """
    # reorder variables
    d = len(mean)
    b = np.ones(d) * np.Inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ordering, _, a_ord, b_ord, L_ord = gibson_ordering(L @ L.T, np.copy(-mean), np.copy(b))  
    shift = np.zeros(d)

    samples_ord, weights = sample_sov(a_ord, b_ord, L_ord, nsample, seed, shift=shift)
    samples_ord = samples_ord @ L_ord.T - a_ord
    samples = np.zeros_like(samples_ord)
    samples[:, ordering] = samples_ord
    if not np.all(np.min(samples, 0) >= - 1e-7):
        print('negative samples')
    return samples, weights

def upper_bound(l, u, L):
    d_ = len(l)
    def psi(xmu, l, u, L):
        x = xmu[:d_]
        mu = xmu[d_:]
        e = -agnp.dot(x, mu) + agnp.dot(mu, mu) / 2 #+ np.log(norm.cdf(np.dot(a, x) + b))
        for i in range(len(mu)):
            if i == 0:
                tmp = 0
            else:
                tmp = L[i, :i] @ x[:i]
            ui_ = (u[i] - tmp) / L[i, i]
            li_ = (l[i] - tmp) / L[i, i]
            e += agnp.log(agnorm.cdf(ui_ - mu[i]) - agnorm.cdf(li_ - mu[i]))
        return e
    res = root(grad(psi), np.zeros(2*d_), args=(l, u, L), jac=jacobian(grad(psi)), method='hybr')
    assert res.success, "optimization failed"
    assert np.all(L @ res.x[:d_] - l >= 0), "not feasible"
    upper_bound = np.exp(psi(res.x, l, u, L))
    return upper_bound

def upper_bound_numerator(mean, L, c1, c2, initial=None, verbose=False):
    c1_t = L.T @ c1
    c2_t = c2 + agnp.dot(c1, mean)
    l = -mean
    d_ = len(l)
    def psi(xmu, l, L):
        x = xmu[:d_]
        mu = xmu[d_:]
        e = -agnp.dot(x, mu) + agnp.dot(mu, mu) / 2 
        c3 = agnp.dot(c1_t, x) + c2_t
        if c3 < 0:
            e += agnp.log(agnorm.cdf(c3))
        else:
            e += agnp.log(1. - agnorm.cdf(-c3))
        for i in range(len(mu)):
            if i == 0:
                tmp = 0
            else:
                tmp = L[i, :i] @ x[:i]
            li_ = (l[i] - tmp) / L[i, i] - mu[i]
            if li_ > 0:
                e += agnp.log(agnorm.cdf(-li_))
            else:
                e += agnp.log(1. - agnorm.cdf(li_))
        return e
    if initial is None:
        initial = np.zeros(2*d_)

    res = root(grad(psi), initial, args=(l, L), jac=jacobian(grad(psi)), method='lm')
    if not res.success:
        if verbose:
            print("failed finding upper bound")
        return 0.
    # assert np.all(L @ res.x[:d_] - l >= 0), "not feasible"
    upper_bound = np.exp(psi(res.x, l, L))
    return upper_bound

def lower_bound(l, u, L, verbose=False):
    d_ = len(l)
    Sigma = L @ L.T
    Sigma_inv = agnp.linalg.inv(Sigma)
    def obj(nusigma, l, u, Sigma_inv):
        d_ = len(l)
        nu = nusigma[:d_]
        sigma = nusigma[d_:]
        alpha = (l - nu) / sigma
        beta = (u - nu) / sigma
        beta_cdf = agnorm.cdf(beta)
        alpha_cdf = 1 - agnorm.cdf(-alpha)
        # alpha_cdf[alpha > 3] = 1 - agnorm.cdf(-alpha[alpha > 3])
        p = beta_cdf - alpha_cdf
        f1 = (alpha * agnorm.pdf(alpha) - beta * agnorm.pdf(beta)) / p
        f2 = (agnorm.pdf(alpha) - agnorm.pdf(beta)) / p
        mean_X = nu + sigma * f2
        res = .5 * agnp.sum(agnp.diag(Sigma_inv) * sigma**2 * (1 + f1 - f2**2) )
        res = res + .5 * agnp.dot(mean_X, Sigma_inv @ mean_X) 
        res = res - agnp.sum(f1 / 2 + agnp.log(agnp.sqrt(2 * math.pi * math.exp(1)) * sigma * p))
        return res
    x0 = np.zeros(2*d_)
    x0[:d_] = 0.
    x0[d_:] = np.sqrt(np.diag(Sigma))
    bounds = [(-np.Inf, np.Inf)]*d_ + [(0, np.Inf)]*d_

    res = minimize(obj, x0, args=(l, u, Sigma_inv), jac=grad(obj), method='Nelder-Mead', options={'maxiter': 2000}) #, method='BFGS')
    # res = minimize(obj, x0, args=(l, u, Sigma_inv), jac=grad(obj), method='BFGS', options={'maxiter': 100000}) #, method='BFGS')
    # if not res.success:
    #     res = minimize(obj, res.x, args=(l, u, Sigma_inv), jac=grad(obj), method='BFGS', options={'maxiter': 100000})
    #     if not res.success:
    #         res = minimize(obj, res.x, args=(l, u, Sigma_inv), jac=grad(obj), method='Nelder-Mead', options={'maxiter': 20000}) #, method='BFGS')
    #         if not res.success:
    #             res = minimize(obj, x0, args=(l, u, Sigma_inv), jac=grad(obj), method='Nelder-Mead', options={'maxiter': 20000})
    #             if not res.success:
    #                 res = minimize(obj, res.x, args=(l, u, Sigma_inv), jac=grad(obj), method='BFGS', options={'maxiter': 100000}) 
    #                 if not res.success:
    #                     if verbose:
    #                         print("failed finding lower bound")
    lower_bound = np.exp(-res.fun) / np.sqrt(2 * math.pi)**d_ / np.sqrt(np.linalg.det(Sigma))
    return lower_bound

def trunc_norm_1d(u, mu, sigma, a, b):
    z = norm.ppf(norm.cdf((a - mu) / sigma) + (norm.cdf((b - mu) / sigma) - norm.cdf((a - mu) / sigma)) * u)
    return mu + sigma * z

def ci_bisection(get_pvalue, sd, right, left, sig_level=0.05, tol=1e-6):
    incre = sd / 5
    ## right end
    pval_t = get_pvalue(right)
    if pval_t < sig_level:
        up = right
        t = right
        for i in range(100):
            t = t - incre
            pval_t = get_pvalue(t)
            if pval_t >= sig_level - tol:
                lo = t
                p_lo = pval_t
                up = t + incre
                break
        if i == 99:
            lo = right
            up = right + 10 * sd
    else:
        lo = right
        t = right
        for i in range(100):
            t = t + incre
            pval_t = get_pvalue(t)
            if pval_t <= sig_level + tol:
                up = t
                p_up = pval_t
                lo = t - incre
                break
    # bisection
    for i in range(100):
        if up - lo <= tol:
            break
        mid = (lo + up) / 2
        p_mid = get_pvalue(mid)
        if p_mid < sig_level:
            up = mid
        if p_mid >= sig_level:
            lo = mid
        if abs(p_mid - sig_level) <= tol:
            break
    if i == 99:
        raise AssertionError("did not converge")
    rightend = up
    
    ## left end
    pval_t = get_pvalue(left)
    if pval_t < sig_level:
        lo = left
        t = left
        for i in range(100):
            t = t + incre
            pval_t = get_pvalue(t)
            if pval_t >= sig_level - tol:
                up = t
                p_up = pval_t
                lo = t - incre
                break
        if i == 99:
            up = left
            lo = left - 10 * sd
    else:
        up = left
        t = left
        for i in range(100):
            t = t - incre
            pval_t = get_pvalue(t)
            if pval_t <= sig_level + tol:
                lo = t
                p_lo = pval_t
                up = t + incre
                break
    # bisection
    for i in range(100):
        if up - lo <= tol:
            break
        mid = (lo + up) / 2
        p_mid = get_pvalue(mid)
        if p_mid < sig_level:
            lo = mid
        if p_mid >= sig_level:
            up = mid
        if abs(p_mid - sig_level) <= tol:
            break
    if i == 99:
        raise AssertionError("did not converge")
    leftend = lo
    return (leftend, rightend)

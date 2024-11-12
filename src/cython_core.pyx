import numpy as np, cython
cimport numpy as np
from scipy.special import ndtr, ndtri, owens_t
from scipy.linalg import inv
from scipy.stats import qmc
from libc.math cimport exp, sqrt

DTYPE_float = float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = int
ctypedef np.int_t DTYPE_int_t
ctypedef np.intp_t DTYPE_intp_t

cdef double MACHINE_EPS = np.finfo(float).eps
cdef double PI = np.pi

@cython.boundscheck(False)
@cython.cdivision(False)
def sample_sov(np.ndarray[DTYPE_float_t, ndim=1] a, 
               np.ndarray[DTYPE_float_t, ndim=1] b, 
               np.ndarray[DTYPE_float_t, ndim=2] L, 
               int n,
               unsigned int seed=1,
               int spherical=1,
               int rqmc=1,
               np.ndarray[DTYPE_float_t, ndim=1] shift=None):
    """
    Generate n weighted samples from

                        N(0, I) \Indc {Lx \in [a, b]}

    by the separation-of-variable method

    L has to be lower triangular and has positive diagonals
    
    SOV method:
    Importance sampling proposal: g(x_1) g(x_2|x_1) ... g(x_d|x_{1:d-1})
    where g(x_i) = \varphi(x_i)\Indc{ta_i \leq x_i \leq tb_i } / (\Phi(tb_i) - \Phi(ta_i) )
    where ta_i, and tb_i depend on x_{1:i-1}:
    ta_i = (a_i - \sum_{j=1}^{i-1}L_{ij}x_j ) / L_{ii}
    tb_i = (b_i - \sum_{j=1}^{i-1}L_{ij}x_j ) / L_{ii}

    Parameters
    -----------

    a: (d, ) lower bound of Lx

    b: (d, ) upper bound of Lx

    L: (d, d) 

    n: number of samples to generate

    seed: random seed

    spherical: boolean, return x if true; else return Lx

    rqmc: boolean, use RQMC or not

    shift: default is None (i.e. no shift); if not none, g(x_i) is replaced by g(x_i - shift[i])

    Return
    ---------

    samples: (n, d)

    weights: (n, ) importance weights for each sample

    """

    cdef int d = len(a)

    # generate uniform samples by scrambled Sobol sequence
    cdef np.ndarray[DTYPE_float_t, ndim=2] U = np.empty((n, d), float)
    if rqmc == 1:
        soboleng = qmc.Sobol(d, scramble=True, seed=seed)
        U = soboleng.random(n) * (1 - MACHINE_EPS) + .5 * MACHINE_EPS
    else:
        rng = np.random.default_rng(seed)
        U = rng.random((n, d))

    if shift is None:
        shift = np.zeros(d, float)
    
    cdef double[:, :] samples = np.empty((n, d), float)
    cdef double[:] weights = np.empty(n, float)

    cdef double[:] x = np.empty(d, dtype=float)
    cdef double[:] u = np.empty(d, dtype=float)
    cdef Py_ssize_t k, j, i
    cdef double aj_, bj_, tmp, w, lo, hi, factor

    for k in range(n):
        u = U[k]
        w = 1.
        for j in range(d):
            tmp = 0.
            if j > 0:
                for i in range(j):
                    tmp = tmp + L[j, i] * x[i]
            aj_ = (a[j] - tmp) / L[j, j] - shift[j]
            bj_ = (b[j] - tmp) / L[j, j] - shift[j]
            if aj_ > 0:
                lo = 1 - ndtr(-aj_)
            else:
                lo = ndtr(aj_)
            if bj_ > 0:
                hi = 1 - ndtr(-bj_)
            else:
                hi = ndtr(bj_)
            factor = hi - lo
            uj_ = lo + factor * u[j]
            uj_ = uj_ * (1 - MACHINE_EPS) + .5 * MACHINE_EPS
            if uj_ > 0.5:
                x[j] = -ndtri(1 - uj_) + shift[j]
            else:
                x[j] = ndtri(uj_) + shift[j]
            w = w * exp(.5 * shift[j]**2 - x[j] * shift[j]) * factor
        if spherical:
            samples[k] = x
        else:
            for ii in range(d):
                tmp = 0
                for jj in range(d):
                    tmp = tmp + L[ii, jj] * x[jj]
                samples[k, ii] = tmp
        weights[k] = w
    return np.array(samples), np.array(weights)



@cython.boundscheck(False)
@cython.cdivision(False)
def joint_cdf_bivnormal(double h, 
                        double k, 
                        double rho): 
    """
    Compute the probability of P(x < h, y < k) where x, y are N(0, 1) with correlation rho
    """
    cdef double ph, pk, phk
    if h < 0:
        ph = ndtr(h)
    else:
        ph = 1 - ndtr(-h)
    if k < 0:
        pk = ndtr(k)
    else:
        pk = 1 - ndtr(-k)
    if min(h, k) < 0:
        phk = ndtr(min(h, k))
    else:
        phk = 1 - ndtr(-min(h, k))
    if rho == 0:
        return ph * pk
    if rho == 1:
        return phk
    if rho == -1:
        return (ph + pk - 1) * (h + k >= 0)
    if h == k == 0:
        return .25 + np.arcsin(rho / (2 * PI))

    cdef double rho2 = sqrt(1 - rho**2)
    if h * k > 0:
        J = 0
    elif h * k == 0 and h + k >= 0:
        J = 0
    else:
        J = 1    
    return .5 * ph + .5 * pk - owens_t(h, (k - rho * h) / (h * rho2)) - owens_t(k, (h - rho * k) / (k * rho2)) - .5 * J
    

@cython.boundscheck(False)
@cython.cdivision(False)
def st_cdf(np.ndarray[DTYPE_float_t, ndim=1] mu, 
           np.ndarray[DTYPE_float_t, ndim=2] L, 
           np.ndarray[DTYPE_float_t, ndim=1] c1,
           double c2,
           int n,
           unsigned int seed,
           np.ndarray[DTYPE_float_t, ndim=1] shift=None,
           int rqmc=1,
           int debug=1):
    """
    Soft-truncated normal CDF

    Compute the expectation 

                EE {Phi(c_1' x + c_2) }, where x ~ N(mu, Sigma=L@L.T) \indc{x >= 0}

    Parameters:
    ----------

    mu: (d, ) mean of Gaussian

    L: (d, d) should be lower triangular and have positive diagonals (i.d. Cholesky decomposition of the Gaussian covariance)

    c1: (d, ) as above

    c2: as above

    n: number of RQMC samples to use

    seed: random seed

    shift: similar as in sample_sov

    Returns:
    ---------

    estimator of numerator

    estimator of denominator

    (so the estimator of the probability is the ratio)
    """

    cdef int d = len(mu)
    # generate uniform samples by scrambled Sobol sequence
    cdef np.ndarray[DTYPE_float_t, ndim=2] U = np.empty((n, d-1), float)
    if rqmc == 0:
        rng = np.random.default_rng(seed)
        U = rng.random((n, d-1))
    else:
        if debug == 1:
            soboleng = qmc.Sobol(d, scramble=True, seed=seed)
            U = soboleng.random(n)[:, :d-1] * (1 - MACHINE_EPS) + .5 * MACHINE_EPS
        else:
            soboleng = qmc.Sobol(d-1, scramble=True, seed=seed)
            U = soboleng.random(n) * (1 - MACHINE_EPS) + .5 * MACHINE_EPS

    if shift is None:
        shift = np.zeros(d, float)

    cdef double[:] weights = np.empty(n, float)
    cdef double[:] numerators = np.empty(n, float)

    cdef double[:] x = np.empty(d, dtype=float)
    cdef double[:] u = np.empty(d, dtype=float)
    cdef Py_ssize_t k, j
    cdef double aj_, tmp, w, lo, factor, tmp2
    cdef np.ndarray[DTYPE_float_t, ndim=1] c1_t = L.T @ c1

    cdef double c2_t = c2
    for j in range(d):
        c2_t = c2_t + c1[j] * mu[j]
    cdef double c2_
    cdef double c1_ = c1_t[-1]

    cdef np.ndarray[DTYPE_float_t, ndim=1] a = -mu

    for k in range(n):
        u = U[k]
        w = 1.
        for j in range(d):
            tmp = 0.
            if j > 0:
                for i in range(j):
                    tmp = tmp + L[j, i] * x[i]
            aj_ = (a[j] - tmp) / L[j, j] - shift[j]
            if aj_ > 0:
                lo = (1 - ndtr(-aj_))
            else:
                lo = ndtr(aj_)
            factor = 1 - lo
            if j < d-1:  
                uj_ = lo + factor * u[j]
                uj_ = uj_ * (1 - MACHINE_EPS) + .5 * MACHINE_EPS
                if uj_ > 0.5:
                    x[j] = -ndtri(1 - uj_) + shift[j]
                else:
                    x[j] = ndtri(uj_) + shift[j]
                w = w * exp(.5 * shift[j]**2 - x[j] * shift[j]) * factor
            else:
                c2_ = c2_t
                for i in range(d-1):
                    c2_ = c2_ + c1_t[i] * x[i]
                num = w * joint_cdf_bivnormal(-aj_, c2_ / sqrt(1 + c1_*c1_), c1_ / sqrt(1 + c1_*c1_))
                w = w * factor
        weights[k] = w
        numerators[k] = num
    return np.mean(numerators), np.mean(weights)

def Gibbs(np.ndarray[DTYPE_float_t, ndim=1] mu, 
          np.ndarray[DTYPE_float_t, ndim=2] Sigma, 
          np.ndarray[DTYPE_float_t, ndim=1] initial, 
          int maxit, 
          unsigned int seed, 
          int PC_every=-1, 
          np.ndarray[DTYPE_float_t, ndim=1] eigval=None,  # descending order
          np.ndarray[DTYPE_float_t, ndim=2] PCs=None,  # (nPC, d), eigvec.T
          np.ndarray[DTYPE_float_t, ndim=1] eta=None, 
          int eta_every=-1):
    """
    Gibbs sampler for sampling Gaussian vectors truncated to the positive orthant
                    
                            x ~ N(mu, Sigma) \indc{x >= 0} 
    
    Each step moves either in a coordinate direction or one of the principal component (PC) of Sigma or the eta direction. If a step moves in the PC directions, each PC direction is chosen with probability proportional to eigval.

    Parameters
    ----------

    mu: (d, ) mean of Gaussian

    Sigma: (d, d) covariance of Gaussian
    
    initial: (d, ) start point

    maxit: number of samples to draw

    seed: random seed for reproducibility

    PC_every: how often to move in the PC directions

    eigval: top eigenvalues of Sigma in descending order 

    PCs: eigenvectors of Sigma corresponding to eigval

    eta_every: how often to move in the eta direction

    eta: (d, ) bias direction
    
    Returns
    ----------

    samples: (n, d)

    """
    cdef int d = len(mu)
    rng = np.random.default_rng(seed) 
    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = rng.random(maxit)
    cdef np.ndarray[DTYPE_intp_t, ndim=1] coord_idx = rng.integers(0, d, size=maxit)

    cdef int nPC
    cdef np.ndarray[DTYPE_int_t, ndim=1] PC_idx
    if PC_every > 0:
        nPC = PCs.shape[0]
        PC_idx = rng.choice(nPC, size=maxit, replace=True, p=eigval / np.sum(eigval))

    # pre-compute the conditional mean multipliers and conditional covariances
    # for coordinate directions
    cdef np.ndarray[DTYPE_float_t, ndim=1] x_t = initial.copy()
    cdef np.ndarray[DTYPE_float_t, ndim=2] cond_mean_factors = np.empty((d, d-1), float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] cond_sigmas = np.empty(d, float)
    cdef int i, j, t
    for i in range(d):
        i_ = np.concatenate([np.arange(i), np.arange(i+1, d)])
        cond_mean_factors[i] = Sigma[i, i_] @ inv(Sigma[i_][:, i_])
        cond_sigmas[i] = np.sqrt(Sigma[i, i] - Sigma[i, i_] @ inv(Sigma[i_][:, i_]) @ Sigma[i_, i])

    # for the eta direction
    cdef DTYPE_float_t cond_sigma_eta
    cdef np.ndarray[DTYPE_float_t, ndim=1] cond_mean_factor_eta
    if eta is not None:
        cond_sigma_eta = np.sqrt(np.dot(eta, Sigma @ eta))
        cond_mean_factor_eta = -inv(Sigma) @ eta * cond_sigma_eta**2
    cdef np.ndarray[DTYPE_float_t, ndim=2] samples = np.empty((maxit, d), float)
    cdef int flag_eta = 0
    cdef int flag_PC = 0
    cdef int docoord = 1
    cdef int doeta = 1
    cdef double cond_mean, cond_sigma, lower, upper, lower_u, upper_u, u_, zz
    cdef np.ndarray[DTYPE_float_t, ndim=1] theta
    for t in range(maxit):
        docoord = 1

        flag_eta = flag_eta + 1
        flag_PC = flag_PC + 1

        if flag_eta == eta_every:
            doeta = 1
            docoord = 0
            theta = eta.copy()
            flag_eta = 0
        elif flag_PC == PC_every:
            doeta = 0
            docoord = 0
            theta = PCs[PC_idx[t]].copy()
            flag_PC = 0
        
        lower = -1e12
        upper = 1e12
        u = usample[t]
        if docoord:  # move in a coordinate: idx
            idx = coord_idx[t]
            cond_mu = mu[idx]
            for j in range(d):
                if j < idx:
                    cond_mu += cond_mean_factors[idx, j] * (x_t[j] - mu[j])
                elif j > idx:
                    cond_mu += cond_mean_factors[idx, j-1] * (x_t[j] - mu[j])

            # x_t[idx] = trunc_norm_1d(u, cond_mu, cond_sigmas[idx], 0, np.Inf)
            lower_u = -cond_mu / cond_sigmas[idx]
            lower_u = ndtr(lower_u) if lower_u < 0 else 1 - ndtr(-lower_u)
            upper_u = 1
            u_ = lower_u + (upper_u - lower_u) * u
            u_ = u_ * (1 - MACHINE_EPS) + .5 * MACHINE_EPS
            zz = ndtri(u_) if u_ < 0.5 else -ndtri(1 - u_)
            x_t[idx] = cond_mu + cond_sigmas[idx] * zz
        else: 
            if doeta:  # move in the eta direction
                cond_mu = 0
                for j in range(d):
                    cond_mu += cond_mean_factor_eta[j] * (x_t[j] - mu[j])
                cond_sigma = cond_sigma_eta
            else:  # move in a PC direction
                cond_mu = 0
                for j in range(d):
                    cond_mu += -PCs[PC_idx[t], j] * (x_t[j] - mu[j])
                cond_sigma = sqrt(eigval[PC_idx[t]])
            for j in range(d):  # compute lower/upper bound for the 1-dim Gaussian
                if theta[j] != 0:
                    val = -x_t[j] / theta[j]
                if theta[j] > 0 and (lower < val):
                    lower = val
                elif theta[j] < 0 and (upper > val):
                    upper = val
            
            if lower < upper:
                lower_u = (lower - cond_mu) / cond_sigma
                lower_u = ndtr(lower_u) if lower_u < 0 else 1 - ndtr(-lower_u)
                upper_u = (upper - cond_mu) / cond_sigma
                upper_u = ndtr(upper_u) if upper_u < 0 else 1 - ndtr(-upper_u)
                u_ = lower_u + (upper_u - lower_u) * u
                u_ = u_ * (1 - MACHINE_EPS) + .5 * MACHINE_EPS
                zz = ndtri(u_) if u_ < 0.5 else -ndtri(1 - u_)
                x_t = x_t + theta * (cond_mu + cond_sigma * zz)
            else:
                pass  # do not move
                # print("lower > upper")

        samples[t] = np.copy(x_t)
    return np.array(samples)


import numpy as np
import pandas as pd

def _design(p, rho, equicorrelated):
    """
    Create an equicorrelated or AR(1) design.
    """
    if equicorrelated:
        Sigma = rho * np.ones([p, p]) + (1 - rho) * np.eye(p)
    else:
        Sigma = rho ** abs(np.arange(p).reshape(1, -1) - np.arange(p).reshape(-1, 1))
    return np.linalg.cholesky(Sigma)

def gaussian_instance(n=100,
                      p=200,
                      s=7,
                      sigma=1.,
                      rho=0.,
                      signal=7,
                      random_signs=False,
                      df=np.inf,
                      scale=True,
                      center=True,
                      equicorrelated=True,
                      seed=1):


    """
    A testing instance for the LASSO.
    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    sigma : float
        Noise level

    rho : float 
        Correlation parameter. Must be in interval [0,1] for
        equicorrelated, [-1,1] for AR(1).

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    scale : bool
        Scale columns of design matrix?

    center : bool
        Center columns of design matrix?

    equicorrelated : bool
        Should columns of design be equi-correlated
        or AR(1)?

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.

    Notes
    -----
        
    The size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
    If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    """

    chol = _design(p, rho, equicorrelated)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).dot(chol.T)

    if center:
        X -= X.mean(0, keepdims=True)

    beta = np.zeros(p) 
    signal = np.atleast_1d(signal)
    if signal.shape == (1,):
        beta[:s] = signal[0] 
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * rng.binomial(1, 0.5, size=(s,)) - 1.)
    rng.shuffle(beta)
    beta /= np.sqrt(n)

    if scale:
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)

    noise = rng.standard_normal(n)
    
    Y = (X.dot(beta) + noise) * sigma
    return X, Y, beta * sigma, chol, scaling


def logistic_instance(n=100,
                      p=200,
                      s=7,
                      rho=0.3,
                      signal=14,
                      random_signs=False, 
                      scale=True, 
                      center=True, 
                      equicorrelated=True,
                      seed=1):
    """
    A testing instance for the LASSO.
    Design is equi-correlated in the population,
    normalized to have columns of norm 1.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    rho : float 
        Correlation parameter. Must be in interval [0,1] for
        equicorrelated, [-1,1] for AR(1).

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    scale : bool
        Scale columns of design matrix?

    center : bool
        Center columns of design matrix?

    equicorrelated : bool
        Should columns of design be equi-correlated
        or AR(1)?

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigmaX : np.ndarray((p,p))
        Row covariance.

    Notes
    -----
        
    The size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
    If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.
    """

    chol = _design(p, rho, equicorrelated)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).dot(chol.T)
    if center:
        X -= X.mean(0)[None,:]

    beta = np.zeros(p) 
    signal = np.atleast_1d(signal)
    if signal.shape == (1,):
        beta[:s] = signal[0] 
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * rng.binomial(1, 0.5, size=(s,)) - 1.)
    rng.shuffle(beta)
    beta /= np.sqrt(n)

    if scale:
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)

    eta = np.dot(X, beta)
    pi = np.exp(eta) / (1 + np.exp(eta))

    Y = rng.binomial(1, pi)
    return X, Y, beta


def HIV_NRTI(drug='3TC', 
             standardize=True, 
             datafile=None,
             min_occurrences=11):
    """
    Download 
        http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt
    and return the data set for a given NRTI drug.
    The response is an in vitro measurement of log-fold change 
    for a given virus to that specific drug.
    Parameters
    ----------
    drug : str (optional)
        One of ['3TC', 'ABC', 'AZT', 'D4T', 'DDI', 'TDF']
    standardize : bool (optional)
        If True, center and scale design X and center response Y.
    datafile : str (optional)
        A copy of NRTI_DATA above.
    min_occurrences : int (optional)
        Only keep positions that appear
        at least a minimum number of times.
        
    """

    if datafile is None:
        datafile = "https://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt"
    NRTI = pd.read_table(datafile, na_values="NA")

    NRTI_specific = []
    NRTI_muts = []
    mixtures = np.zeros(NRTI.shape[0])
    for i in range(1,241):
        d = NRTI['P%d' % i]
        for mut in np.unique(d):
            if mut not in ['-','.'] and len(mut) == 1:
                test = np.equal(d, mut)
                if test.sum() >= min_occurrences:
                    NRTI_specific.append(np.array(np.equal(d, mut))) 
                    NRTI_muts.append("P%d%s" % (i,mut))

    NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)

    X_NRTI = np.array(NRTI_specific, float)
    Y = np.asarray(NRTI[drug]) # shorthand
    keep = ~np.isnan(Y).astype(bool)
    X_NRTI = X_NRTI[np.nonzero(keep)]; Y=Y[keep]
    Y = np.array(np.log(Y), float); 

    if standardize:
        Y -= Y.mean()
        X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
    return X_NRTI, Y, np.array(NRTI_muts)

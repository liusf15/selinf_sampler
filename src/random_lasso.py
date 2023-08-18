import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
from scipy.special import ndtr, ndtri
import cvxpy
import pandas as pd
import time
from collections import namedtuple

from regreg.smooth.glm import glm
from selectinf.algorithms import lasso
from selectinf.randomized.lasso import split_lasso 
from selectinf.base import selected_targets, full_targets

from src.core import sample_sov_reorder, Gibbs, joint_cdf_bivnormal, ci_bisection, st_cdf, sample_sov

MACHINE_EPS = np.finfo(np.float64).eps

Params = namedtuple('Params', ['eta', 'theta_hat', 'cov_b', 'mu_b_added', 'mu_b_multi', 'var_theta', 'mu_theta_added', 'mu_theta_multi_b', 'mu_theta_multi_theta'])

class random_lasso():
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.p = X.shape[1]
        
    def prepare_eta(self, eta):
        nu = np.dot(eta, self.Lambda @ eta) 
        if nu > 0:
            c = self.Lambda @ eta / nu
        else:
            c = np.zeros(self.d)
        Dc = self.D @ c
        theta_hat = np.dot(eta, self.beta_hat)
        beta_perp = self.beta_hat - c * theta_hat
        HDc = self.H @ Dc
        if nu > 0:
            var_theta = 1 / (1 / nu + np.dot(Dc, HDc))
        else:
            var_theta = 0
        cov_b_inv = self.H - var_theta * np.outer(HDc, HDc)
        cov_b = np.linalg.inv(cov_b_inv)
        r = self.r_ + self.Q_1_ @ beta_perp
        Kr = self.K @ r
        mu_b_added = cov_b @ (-Kr + var_theta * np.outer(HDc, Dc) @ Kr)
        if nu > 0:
            mu_b_multi = var_theta / nu * cov_b @ HDc
            mu_theta_multi_theta = var_theta / nu
        else:
            mu_b_multi = np.zeros(self.d)
            mu_theta_multi_theta = 0
        mu_theta_multi_b = var_theta * HDc
        mu_theta_added = var_theta * np.dot(Dc, Kr)
        return Params(eta, theta_hat, cov_b, mu_b_added, mu_b_multi, var_theta, mu_theta_added, mu_theta_multi_b, mu_theta_multi_theta)

    def naive_inference(self, sig_level=0.05, beta=None):
        """
        Inference without correction
        """
        sd = np.sqrt(np.diag(self.Lambda))
        q = ndtri(sig_level / 2)
        lower = self.beta_hat + q * sd
        upper = self.beta_hat - q * sd
        if beta is None:
            pvals = 2 * ndtr(-abs(self.beta_hat / sd))
        else:
            pvals = 2 * ndtr(-abs((self.beta_hat - beta) / sd))
        return pvals, np.stack([lower, upper]).T
    
    def splitting_inference(self, sig_level=0.05, beta=None):
        """
        Inference using the remaining data only
        """
        sd = np.sqrt(np.diag(self.Sigma_2))
        q = ndtri(sig_level / 2)
        lower = self.beta_hat_2 + q * sd
        upper = self.beta_hat_2 - q * sd
        if beta is None:
            pvals = 2 * ndtr(-abs(self.beta_hat_2 / sd))
        else:
            pvals = 2 * ndtr(-abs((self.beta_hat_2 - beta) / sd))
        return pvals, np.stack([lower, upper]).T

    def exact_bivar_pivot(self, j, params, theta, two_sided=True):
        """
        Method proposed by Panigrahi et al. 2022
        Exact Selective Inference with Randomization
        http://arxiv.org/abs/2212.12940
        """
        d = self.d
        xi_obs = self.D @ self.beta_lasso
        theta_hat = params.theta_hat
        theta_cond_var = params.var_theta
        theta_mean_multi_theta = params.mu_theta_multi_theta
        theta_mean_added = params.mu_theta_added
        theta_mean_multi_b = params.mu_theta_multi_b
        c1 = theta_mean_multi_b[j]

        msk_j = np.concatenate([np.arange(j), np.arange(j+1, d)])
        xi_cov = params.cov_b
        xi_mean_added = params.mu_b_added
        xi_mean_multi = params.mu_b_multi
        xi_cov_inv = np.linalg.inv(xi_cov[msk_j][:, msk_j])
        xi_j_cond_sigma = np.sqrt(xi_cov[j, j] - np.dot(xi_cov[j, msk_j], xi_cov_inv @ xi_cov[msk_j, j] ) )

        xi_mean = xi_mean_added + xi_mean_multi * theta
        xi_j_cond_mean = xi_mean[j] + xi_cov[j, msk_j] @ xi_cov_inv @ (xi_obs[msk_j] - xi_mean[msk_j] )
        den = ndtr(xi_j_cond_mean / xi_j_cond_sigma)

        f = np.sqrt(c1**2 * xi_j_cond_sigma**2 + theta_cond_var)
        rho_ = -c1 * xi_j_cond_sigma / f
        c0 = theta_mean_added + theta_mean_multi_theta * theta + np.dot(theta_mean_multi_b[msk_j], xi_obs[msk_j])
        num = joint_cdf_bivnormal((theta_hat - c1 * xi_j_cond_mean - c0) / f, xi_j_cond_mean / xi_j_cond_sigma, rho_)
        cdf_ = num / den
        if two_sided:
            return 2 * min(cdf_, 1 - cdf_)
        return cdf_

    def exact_bivar_interval(self, j, params, sig_level=0.05, two_sided=True):
        """
        Method proposed by Panigrahi et al. 2022
        Exact Selective Inference with Randomization
        http://arxiv.org/abs/2212.12940
        """
        d = self.d
        xi_obs = self.D @ self.beta_lasso
        theta_hat = params.theta_hat
        theta_cond_var = params.var_theta
        theta_mean_multi_theta = params.mu_theta_multi_theta

        theta_mean_added = params.mu_theta_added
        theta_mean_multi_b = params.mu_theta_multi_b
        c1 = theta_mean_multi_b[j]

        msk_j = np.concatenate([np.arange(j), np.arange(j+1, d)])
        
        xi_cov = params.cov_b
        xi_mean_added = params.mu_b_added
        xi_mean_multi = params.mu_b_multi
        xi_cov_inv = np.linalg.inv(xi_cov[msk_j][:, msk_j])
        xi_j_cond_sigma = np.sqrt(xi_cov[j, j] - np.dot(xi_cov[j, msk_j], xi_cov_inv @ xi_cov[msk_j, j] ) )
        
        f = np.sqrt(c1**2 * xi_j_cond_sigma**2 + theta_cond_var)
        rho_ = -c1 * xi_j_cond_sigma / f

        def _pvalue_(theta):
            xi_mean = xi_mean_added + xi_mean_multi * theta
            xi_j_cond_mean = xi_mean[j] + xi_cov[j, msk_j] @ xi_cov_inv @ (xi_obs[msk_j] - xi_mean[msk_j] )
            den = ndtr(xi_j_cond_mean / xi_j_cond_sigma)

            c0 = theta_mean_added + theta_mean_multi_theta * theta + np.dot(theta_mean_multi_b[msk_j], xi_obs[msk_j])
            num = joint_cdf_bivnormal((theta_hat - c1 * xi_j_cond_mean - c0) / f, xi_j_cond_mean / xi_j_cond_sigma, rho_)
                
            cdf_ = num / den
            if np.isnan(cdf_):
                raise ValueError("nan")
            if cdf_ < 0:
                print("cdf < 0")        
            if cdf_ > 1:
                print("cdf > 1")
            cdf_ = min(max(cdf_, 0.), 1.)
            if two_sided:
                return 2 * min(cdf_, 1 - cdf_)
            return cdf_

        nu = params.var_theta / params.mu_theta_multi_theta
        sd = np.sqrt(nu * (1 + self.kappa))
        splitting_right = theta_hat - 2 * sd * ndtri(sig_level / 2) 
        splitting_left = theta_hat + 2 * sd * ndtri(sig_level / 2)
        
        return ci_bisection(_pvalue_, sd, splitting_right, splitting_left, sig_level=sig_level, tol=1e-6)

    def sample_auxillary(self, params, theta, method, seed, nsample, **gibbs_params):
        """
        helper function for sampling
        """
        mean_b = params.mu_b_added + params.mu_b_multi * theta
        cov_b = params.cov_b
        L = np.linalg.cholesky(cov_b)
        if method == 'sov':
            samples, weights = sample_sov_reorder(mean_b, L, nsample, seed)
            return samples, weights
        if method == 'gibbs':
            initial = np.zeros(self.d)
            if gibbs_params['from_mode']:
                try:
                    x = cvxpy.Variable(self.d)
                    cvxprob = cvxpy.Problem(cvxpy.Minimize((1/2) * cvxpy.quad_form(x, np.eye(self.d))),
                                    [-L @ x <= mean_b])
                    cvxprob.solve()
                    initial = x.value
                    initial = L @ initial + mean_b
                except:
                    initial = np.zeros(self.d)
            eigval, eigvec = np.linalg.eigh(cov_b)
            eigval = eigval[::-1]
            eigvec = eigvec[:, ::-1]
            nPC = np.where(np.cumsum(eigval) >= 0.5 * np.sum(eigval))[0][0]
            tmp = np.where(eigval > eigval.mean())[0]
            if len(tmp) > 0:
                nPC = max(max(nPC, tmp[-1]), 1)
            PCs = eigvec[:, :nPC].T
            
            if self.d == 1:
                mean_ = mean_b[0]
                sigma_ = np.sqrt(cov_b[0, 0])
                lo = ndtr(-mean_ / sigma_) if mean_b >= 0 else 1 - ndtr(mean_ / sigma_)
                rng = np.random.default_rng(seed)
                usamples = rng.random(nsample)
                samples = ndtri(usamples * (1 - lo) + lo) * sigma_ + mean_
                samples = samples.reshape(nsample, 1)
            else:
                samples = Gibbs(mean_b, cov_b, initial=initial, maxit=nsample+gibbs_params['burnin'], seed=seed, PC_every=20, eigval=eigval[:nPC], PCs=PCs, eta=None, eta_every=-1)[-nsample:]
            if samples is None:
                print("Gibbs sampling failed")
                return None
            if np.min(samples) <= -1e-7:
                print('invalid Gibbs samples')
            return samples
    
    def get_pvalue_preint(self, params, theta, nsample=512, seed=1, two_sided=True, return_num_den=False):
        """
        compute p-values, pre-integrating the last coordinate
        """
        c1 = -params.mu_theta_multi_b / np.sqrt(params.var_theta)
        c2 = (params.theta_hat - params.mu_theta_added - params.mu_theta_multi_theta * theta) / np.sqrt(params.var_theta)
        b_mean = params.mu_b_added + params.mu_b_multi * theta
        L = np.linalg.cholesky(params.cov_b)
        num, den = st_cdf(b_mean, L, c1, c2, nsample, seed)
        if return_num_den:
            return num, den
        cdf_ = num / den
        if two_sided:
            return 2 * min(cdf_, 1 - cdf_)
        return cdf_

    def get_pvalue(self, params, theta, samples, weights=None, two_sided=True):
        """
        compute p-values given samples of b
        """
        theta_cond_mean = params.mu_theta_added + params.mu_theta_multi_theta * theta + samples @ params.mu_theta_multi_b
        theta_cond_sd = np.sqrt(params.var_theta)
        if weights is None:
            cdf_ = np.mean(norm.cdf((params.theta_hat - theta_cond_mean) / theta_cond_sd))
        else:
            cdf_ = np.mean(norm.cdf((params.theta_hat - theta_cond_mean) / theta_cond_sd) * weights) / np.mean(weights)
        if two_sided:
            return 2 * min(cdf_, 1 - cdf_)
        else:
            return cdf_

    def get_CI(self, params, samples, theta_sample, weights=None, sig_level=0.05, two_sided=True):
        """
        construct intervals given samples of b
        """
        theta_cond_mean_ = params.mu_theta_added + samples @ params.mu_theta_multi_b
        var_theta = params.var_theta
        HDc = params.mu_theta_multi_b / var_theta
        theta_cond_sd = np.sqrt(params.var_theta)
        theta_hat = params.theta_hat

        if weights is None:
            weights = np.ones(len(samples))
        xi_HDc = samples @ HDc
        nu = var_theta / params.mu_theta_multi_theta
        sd = np.sqrt(nu * (1 + self.kappa))
        splitting_right = theta_hat - sd * ndtri(sig_level / 2)
        splitting_left = theta_hat + sd * ndtri(sig_level / 2)

        def _pvalue_(theta):
            weights_2 = weights * np.exp(var_theta / nu * xi_HDc * (theta - theta_sample))
            theta_cond_mean = theta_cond_mean_ + params.mu_theta_multi_theta * theta
            cdf_ = np.mean(norm.cdf((params.theta_hat - theta_cond_mean) / theta_cond_sd) * weights_2) / np.mean(weights_2)
            if not two_sided:
                return cdf_
            return 2 * min(cdf_, 1 - cdf_)        
        return ci_bisection(_pvalue_, sd, splitting_left, splitting_right, sig_level=sig_level, tol=1e-6)

    def get_pvalue_eta0(self, params, theta, samples, weights, two_sided=True):
        """
        get p-values based on samples of b, which are sampled from N(\bar\mu, \bar\Sigma)|_O, need additional importance weighting
        """
        nu = params.var_theta / params.mu_theta_multi_theta
        theta_hat = params.theta_hat
        var_theta = params.var_theta
        HDc = params.mu_theta_multi_b / var_theta
        xi_HDc = samples @ HDc
        weights_2 = np.exp(var_theta * xi_HDc ** 2 / 2)
        weights_2 = weights_2 * np.exp(theta * var_theta / nu * xi_HDc)
        tau = params.mu_theta_added - theta_hat
        weights_2 = weights_2 * np.exp(tau * xi_HDc)
        weights_2 = weights_2 * weights
        theta_cond_sd = np.sqrt(var_theta)
        den = np.mean(weights_2)
        theta_cond_mean = params.mu_theta_added + samples @ params.mu_theta_multi_b + params.mu_theta_multi_theta * theta
        num = np.mean(norm.cdf((theta_hat - theta_cond_mean) / theta_cond_sd) * weights_2)
        cdf_ = num / den
        if not two_sided:
            return cdf_
        return 2 * min(cdf_, 1 - cdf_)

    def get_CI_eta0(self, params, samples, weights, sig_level=0.05, two_sided=True):
        """
        construct intervals based on samples of b, which are sampled from N(\bar\mu, \bar\Sigma)|_O, need additional importance weighting
        """
        theta_hat = params.theta_hat
        var_theta = params.var_theta
        nu = params.var_theta / params.mu_theta_multi_theta
        HDc = params.mu_theta_multi_b / var_theta
        xi_HDc = samples @ HDc
        weights_ = weights * np.exp(var_theta * xi_HDc ** 2 / 2)
        tau = params.mu_theta_added - theta_hat
        weights_ = weights_ * np.exp(tau * xi_HDc)
        theta_cond_sd = np.sqrt(var_theta)

        sd = np.sqrt(nu * (1 + self.kappa))
        splitting_right = theta_hat - sd * ndtri(sig_level / 2)  
        splitting_left = theta_hat + sd * ndtri(sig_level / 2)

        theta_cond_mean_ = params.mu_theta_added + samples @ params.mu_theta_multi_b
        def _pvalue_(theta):
            weights_2 = weights_ * np.exp(theta * var_theta / nu * xi_HDc)
            den = np.mean(weights_2)
            theta_cond_mean = theta_cond_mean_ + params.mu_theta_multi_theta * theta
            num = np.mean(norm.cdf((theta_hat - theta_cond_mean) / theta_cond_sd) * weights_2)
            cdf_ = num / den
            if not two_sided:
                return cdf_
            return 2 * min(cdf_, 1 - cdf_)
            
        return ci_bisection(_pvalue_, sd, splitting_left, splitting_right, sig_level=sig_level, tol=1e-6)
        
    def sampling_inference(self, sig_level=0.05, two_sided=True, n_sov=512, seed=1, return_time=False):
        """
        compute p-values and confidence intervals for all the selected variables 
        """
        start = time.time()
        params0 = self.prepare_eta(np.zeros(self.d))
        eta0_samples, eta0_weights = self.sample_auxillary(params0, 0., 'sov', nsample=n_sov, seed=seed)
        ci = np.zeros((self.d, 2))
        pval = np.zeros(self.d)
        for j in range(self.d):
            eta = np.eye(self.d)[j]
            params = self.prepare_eta(eta)
            pval[j] = self.get_pvalue_eta0(params, 0., eta0_samples, eta0_weights, two_sided=two_sided)
            ci[j] = self.get_CI_eta0(params, eta0_samples, eta0_weights, sig_level=sig_level, two_sided=two_sided)
        end = time.time()
        res = pd.DataFrame(ci, columns=['lower_confidence', 'upper_confidence'])
        res['pvalue'] = pval
        if return_time:
            res['time'] = end - start
        return res

    def sel_loglik(self, beta_E, return_grad=False, return_hess=False, nsov=512, seed=1):
        """
        log-likelihood of the selective likelihood
        """
        beta_hat_prec = np.linalg.inv(self.Lambda)
        b_marginal_prec = (1 - self.rho1) * self.H
        b_marginal_cov = self.H_inv / (1 - self.rho1)
        b_marginal_mean = self.D @ beta_E - self.H_inv @ self.D @ self.r_[self.E] * (self.kappa / self.sigma**2)

        L = np.linalg.cholesky(b_marginal_cov) 
        samples, weights = sample_sov(-b_marginal_mean, np.ones(self.d) * np.Inf, L, nsov, seed=seed)
        Z = samples @ L.T + b_marginal_mean

        ll = -0.5 * (self.beta_hat - beta_E).T @ beta_hat_prec @ (self.beta_hat - beta_E) - np.log(np.mean(weights))
        res = {'loglik': ll}

        if return_grad:
            # gradient of the quadratic term
            grad_0 = beta_hat_prec @ (beta_E - self.beta_hat)

            # gradient of log-denominator
            grad_1 = self.D @ b_marginal_prec @ np.mean((Z - b_marginal_mean) * weights[:, None], 0) / np.mean(weights)
            res['grad'] = grad_0 + grad_1
        if return_hess:
            H_0 = -beta_hat_prec
            H_1 = (Z - b_marginal_mean).T @ np.diag(weights) @ (Z - b_marginal_mean) - b_marginal_cov * np.sum(weights)
            H_1 = self.D @ b_marginal_prec @ H_1 @ b_marginal_prec @ self.D / np.sum(weights) - np.outer(grad_1, grad_1)
            res['hess'] = H_0 - H_1
        return res
        
    def mle_sov(self, nsov=512, seed=1, sig_level=0.05, verbose=False, return_time=False):
        """
        selective MLE-based inference
        """
        beta_E = np.copy(self.beta_hat)
        lr = 0.01
        maxit = 10000
        lls = []
        start = time.time()
        for i in range(maxit):
            tmp = self.sel_loglik(beta_E, return_grad=True, nsov=nsov, seed=seed)
            if np.isnan(tmp['loglik']):
                raise ValueError("loglik is nan")
            grad = tmp['grad']
            lls.append(tmp['loglik'])
            beta_E = beta_E - lr * grad
            if np.linalg.norm(lr * grad) <= 1e-4:
                if verbose:
                    print("converged")
                break
            if (i+1) % 10 == 0:
                if verbose:
                    print(i+1, lls[-1])
                if (lls[-1] - np.mean(lls[-10:-1])) / np.mean(lls[-10:-1]) < 0.01:
                    if verbose:
                        print("converged")
                    break
        if i == maxit - 1:
            print("warning: did not converge")
        beta_mle = beta_E
        mle_cov = inv(-self.sel_loglik(beta_mle, return_grad=True, return_hess=True, nsov=nsov, seed=seed)['hess'])
        sds = np.sqrt(np.diag(mle_cov))
        lower = beta_mle + ndtri(sig_level / 2) * sds
        upper = beta_mle - ndtri(sig_level / 2) * sds
        end = time.time()
        res = pd.DataFrame(np.vstack([beta_mle, lower, upper]).T, columns=['MLE', 'lower_confidence', 'upper_confidence'])
        res['pvalues'] = 2 * norm.cdf(-np.abs(beta_mle) / sds)
        if return_time:
            res['time'] = end - start
        return res

class linear_carving(random_lasso):
    def __init__(self, X, Y, n1) -> None:
        super().__init__(X, Y)
        self.X_1 = X[:n1]
        self.Y_1 = Y[:n1]
        self.n1 = n1
        self.n = X.shape[0]
        self.rho1 = self.n1 / self.n
        Hat = X @ np.linalg.inv(X.T @ X) @ X.T
        self.sigma = np.linalg.norm(Y - Hat @ Y) / np.sqrt(self.n - self.p)
    
    def fit(self, tune_lambda='cv_min', lbd=None, X_tune=None, Y_tune=None, target_d=None, max_d=None):
        if tune_lambda in ['cv_min', 'cv_1se']: # 5-fold cross validation
            rng = np.random.default_rng(1)
            idx = rng.permutation(self.n1)
            n_fold = 5
            idx_folds = np.array_split(idx, n_fold)
            if lbd is None or not hasattr(lbd, '__len__'):
                lbd = np.logspace(np.log10(np.sqrt(np.log(self.p) / self.n1) ) - 1, np.log10(np.sqrt(np.log(self.p) / self.n1)) + 1, 20) / np.sqrt(self.n)
            cv_err = np.zeros((n_fold, len(lbd)))
            n1_ = self.n1 / n_fold * (n_fold - 1)
            for k in range(n_fold):
                X_holdout = self.X_1[idx_folds[k]]
                Y_holdout = self.Y_1[idx_folds[k]]
                X_k = np.delete(self.X_1, idx_folds[k], 0)
                Y_k = np.delete(self.Y_1, idx_folds[k], 0)
                g = glm.gaussian(X_k, Y_k)
                for i in range(len(lbd)):
                    model_ = lasso.lasso(g, lbd[i] * n1_)
                    beta_ = model_.fit()
                    cv_err[k, i] = np.mean((Y_holdout - X_holdout @ beta_)**2)
                    if len(model_.active) == 0:
                        cv_err[k, i:] = cv_err[k, i]
                        break
            cv_err = np.mean(cv_err, 0)
            cv_std = np.std(cv_err, 0)
            if tune_lambda == 'cv_min':
                self.lbd = lbd[np.argmin(cv_err)]
            elif tune_lambda == 'cv_1se': # 1 standard error rule
                self.lbd = lbd[np.where(cv_err <= cv_err.min() + cv_std)[0][0]]
            else:
                raise ValueError("tune_lambda must be 'cv_min' or 'cv_1se'")
            g = glm.gaussian(self.X_1, self.Y_1)
            self.lbd = self.lbd * self.n1
            self.model = lasso.lasso(g, self.lbd)
            self.beta_lasso = self.model.fit()
            self.selected = np.zeros(self.p, np.bool_)
            self.selected[self.model.active] = True
        
        elif tune_lambda == 'extra_data':
            g = glm.gaussian(self.X_1, self.Y_1)
            lbd = np.logspace(np.log10(np.sqrt(np.log(self.p) / self.n1) ) - 1, np.log10(np.sqrt(np.log(self.p) / self.n1)) + 1, 20) / np.sqrt(self.n)
            min_err = np.Inf
            for lbd_ in lbd:
                model_ = lasso.lasso(g, lbd_ * self.n1)
                beta_lasso_ = model_.fit()
                selected_ = np.zeros(self.p, np.bool_)
                selected_[model_.active] = True
                if max_d is not None and sum(selected_) > max_d:
                    print(sum(selected_), "greater than max_d")
                    continue
                X_E_ = self.X[:, selected_]
                beta_hat_ = inv(X_E_.T @ X_E_) @ X_E_.T @ self.Y
                pred_err = np.mean((Y_tune - X_tune[:, selected_] @ beta_hat_)**2)
                print(lbd_, sum(selected_), pred_err)
                if pred_err < min_err:
                    min_err = pred_err
                    beta_lasso = beta_lasso_
                    selected = selected_
                    model = model_
                    lbd_best = lbd_
                if sum(selected_) == 0:
                    break
            self.lbd = lbd_best * self.n1
            self.beta_lasso = beta_lasso
            self.selected = selected
            self.model = model

        elif tune_lambda == 'fixed_d':
            g = glm.gaussian(self.X_1, self.Y_1)
            lo = 0
            hi = 10 * np.sqrt(np.log(self.p) / self.n1) / np.sqrt(self.n)
            model_ = lasso.lasso(g, hi * self.n1)
            beta_lasso_ = model_.fit()
            if len(model_.active) < target_d:
                for i in range(100):
                    mid = (lo + hi) / 2
                    model_ = lasso.lasso(g, mid * self.n1)
                    beta_lasso_ = model_.fit()
                    if len(model_.active) < target_d:
                        hi = mid
                    elif len(model_.active) > target_d:
                        lo = mid
                    else:
                        lbd = mid
                        break
            else:
                lbd = hi
            self.lbd = lbd * self.n1
            self.model = lasso.lasso(g, self.lbd)
            self.beta_lasso = self.model.fit()
            self.selected = np.zeros(self.p, np.bool_)
            self.selected[self.model.active] = True
        elif tune_lambda == 'theory':
            g = glm.gaussian(self.X_1, self.Y_1)
            self.lbd = self.sigma * np.sqrt(np.log(self.p) / self.n1) / np.sqrt(self.n) 
            self.lbd = self.lbd * self.n1 
            self.model = lasso.lasso(g, self.lbd) 
            self.beta_lasso = self.model.fit()
            self.selected = np.zeros(self.p, np.bool_)
            self.selected[self.model.active] = True
        else:
            raise NotImplementedError("tune_lambda must be 'cv_min', 'cv_1se', 'extra_data', 'fixed_d', or 'theory'")
        self.d = int(sum(self.selected))
        self.E = self.selected
        self.beta_lasso = self.beta_lasso[self.selected]
        self.X_E = self.X[:, self.selected]

    def prepare_inference(self, target='selected'):
        self.beta_hat = inv(self.X_E.T @ self.X_E) @ self.X_E.T @ self.Y
        self.subgrad = self.X_1.T @ (self.Y_1 - self.X_1[:, self.selected] @ self.beta_lasso) 
        if not (np.alltrue(abs(self.subgrad[~self.selected]) < self.lbd) and np.allclose(self.subgrad[self.selected], self.lbd * np.sign(self.beta_lasso), atol=1e-4)): 
            raise AssertionError("incorrect subgradient")
        self.Sigma_X = self.X.T @ self.X 
        if target == 'selected':
            self.Lambda = self.sigma**2 * np.linalg.inv(self.Sigma_X[self.selected][:, self.selected])
        elif target == 'full':
            self.Lambda = self.sigma**2 * np.linalg.inv(self.Sigma_X)
            self.Lambda = self.Lambda[self.selected][:, self.selected]
        N = -self.X.T @ (self.Y - self.X_E @ self.beta_hat) * self.rho1
        self.r_ = self.subgrad + N
        self.Q_1_ = -self.X.T @ self.X_E * self.rho1  

        self.signs = np.sign(self.beta_lasso)
        self.D = np.diag(self.signs)
        signed_XE = self.X[:, self.E] @ self.D
        self.H = self.rho1 / (1-self.rho1) * signed_XE.T @ signed_XE / self.sigma**2
        self.H_inv = np.linalg.inv(self.H)
        self.kappa = self.rho1 / (1 - self.rho1) 
        self.beta_hat_2 = inv(self.X[self.n1:, self.E].T @ self.X[self.n1:, self.E]) @ self.X[self.n1:, self.E].T @ self.Y[self.n1:]
        if target == 'selected':
            self.Sigma_2 = self.sigma**2 * inv(self.X[self.n1:, self.E].T @ self.X[self.n1:, self.E])
        elif target == 'full':
            self.Sigma_2 = self.sigma**2 * inv(self.X[self.n1:].T @ self.X[self.n1:])
            self.Sigma_2 = self.Sigma_2[self.selected][:, self.selected]
        J_E = np.eye(self.p)[self.E]
        self.K = 1 / (self.sigma**2 * (1 - self.rho1)) * self.D @ J_E

    def approx_mle_inference(self, target='selected', sig_level=0.05):
        feature_weights_ = np.ones(self.p) * self.lbd / self.rho1 #/ self.n
        perturb = np.zeros(self.n, dtype=bool)
        perturb[:self.n1] = True
        selector = split_lasso.gaussian(self.X, self.Y, feature_weights_, proportion=self.rho1, estimate_dispersion=True)
        signs = selector.fit(perturb=perturb)
        nonzero = signs != 0
        if sum(nonzero) != self.d:
            raise AssertionError("different selection")
        selector.setup_inference(dispersion=self.sigma**2)
        if target == 'selected':
            target_spec = selected_targets(selector.loglike, selector.observed_soln, dispersion=self.sigma**2)
        elif target == 'full':
            target_spec = full_targets(selector.loglike, selector.observed_soln, nonzero, dispersion=self.sigma**2)
        else:
            raise NotImplementedError
        result = selector.inference(target_spec, 'selective_MLE', level=1-sig_level)
        return result

import torch
import torch.distributions as ds


class RegressionModel:
    def __init__(self, d, phi, mu, cov, output_var):
        one = torch.ones(d)
        self.phi = phi
        self.mu = mu * one
        self.cov = cov * one.diag()
        self.output_var = output_var

    def sample_w(self, n):
        mu, cov = self.mu, self.cov
        return ds.MultivariateNormal(mu, cov).sample((n,))
    
    def posterior_w_params(self, x, y):
        mu, prec = self.mu, self.cov.inverse()
        x = self.phi(x)

        cov = (2 * x.T @ x + prec).inverse()
        mu = 2 * cov @ x.T @ y

        return mu, cov
    
    def _pred_params(self, x, mu, cov):
        sig = self.output_var
        x = self.phi(x)

        var = sig + (x @ cov @ x.T).diag()
        mu = mu @ x.T

        return mu, var.diag()

    def prior_pred_params(self, x):
        mu, cov = self.mu, self.cov
        return self._pred_params(x, mu, cov)

    def posterior_pred_params(self, x, y):
        mu, cov = self.posterior_w_params(x, y)
        return self._pred_params(x, mu, cov)

    def log_prior_pred(self, x, y):
        mu, cov = self.prior_pred_params(x)
        return ds.MultivariateNormal(mu, cov).log_prob(y)

    def log_posterior_pred(self, x, y):
        mu, cov = self.posterior_pred_params(x, y)
        return ds.MultivariateNormal(mu, cov).log_prob(y)

import torch
import torch.distributions as ds
from util import polyval, OUTPUT_VAR


class PolyGaussianRegressionModel():
    def __init__(self, weights_mean, weights_var, terms, D=None, output_var=OUTPUT_VAR):
        assert terms in ('even', 'odd', 'all')
        assert type(weights_var) in (float, int)
        if torch.is_tensor(weights_mean):
            assert D == None or weights_mean.shape == (D+1,)
        else:
            assert D >= 0
            weights_mean = weights_mean * torch.ones(D+1)

        self.weights_mean = weights_mean
        self.weights_var = weights_var
        self.terms = terms
        self.polyval = lambda x, w: polyval(x, w, terms=terms)
        self.output_var = output_var

    def predict(self, x, w):
        assert w.shape == self.weights_mean.shape
        return self.polyval(x, w)      

    def sample_model_weights(self, n_samples=1):
        mean = self.weights_mean
        cov = self.weights_var * torch.eye(len(mean))
        weights_prior = ds.MultivariateNormal(self.weights_mean, covariance_matrix=cov)
        samples = weights_prior.sample((n_samples,))
        if n_samples == 1:
            return samples.flatten()
        return samples

    def prior_log_likelihood(self, x, y):
        mean = self.polyval(x, self.weights_mean)
        vars = self.output_var + self.weights_var*self.polyval(x**2, torch.ones_like(self.weights_mean))
        return ds.MultivariateNormal(loc=mean, covariance_matrix=vars.diag()).log_prob(y)


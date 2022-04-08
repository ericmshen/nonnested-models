import torch
import torch.distributions as ds
from scipy.interpolate import approximate_taylor_polynomial
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_data(n):
    x = ds.Uniform(-1, 1).sample((n,1))
    return x


def add_noise(x, output_var):
    noise = torch.randn_like(x) * (output_var ** 0.5)
    return x + noise


def prior_pred_cross_entropy(m1, m2):
    x = torch.linspace(-1, 1, 100)
    mu1, cov1 = m1.prior_pred_params(x)
    mu2, cov2 = m2.prior_pred_params(x)
    return ds.MultivariateNormal(mu1, cov1 + cov2).log_prob(mu2)


def pred_cross_entropy(m1, m2, samples, n):
    w_samples = m1.sample_w(samples)
    entropy = 0
    for w in w_samples:
        x = generate_data(n)
        y = add_noise(m1.phi(x) @ w, m1.output_var)
        entropy += m2.log_posterior_pred(x, y)
    return entropy / samples / n


def log_bf(m1, m2, x, y):
    ll1 = m1.log_prior_pred(x, y)
    ll2 = m2.log_prior_pred(x, y)
    log_bf = ll1 - ll2
    return log_bf


def bf_success_rate(m1, m2, samples, n):
    if m1 is m2:
        return 0.5
    w_samples = m1.sample_w(samples)
    success_rate = 0
    for w in w_samples:
        x = generate_data(n)
        y = add_noise(m1.phi(x) @ w, m1.output_var)
        if log_bf(m1, m2, x, y) >= 0:
            success_rate += 1 / samples
    return success_rate


def plot_diversity(m, ax, samples=5000):
    x = torch.linspace(-1, 1, 200)
    phi = m.phi(x)
    w = m.sample_w(samples)
    y = torch.einsum('jd,id->ij', phi, w)

    q = torch.linspace(0.05, 0.95, 100)
    percentiles = torch.quantile(y, q, dim=0)

    for i in range(len(q)//2):
        low, high = percentiles[i], percentiles[len(q) - i - 1]
        low, high = low.flatten(), high.flatten()
        ax.fill_between(x, low, high, color='#FF1300', alpha=2/len(q), edgecolor='None')

    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(x.min(), x.max())


def taylor_coeff_var(m, deg, samples=5000):
    w = m.sample_w(samples)
    coeffs = torch.zeros(samples, deg+1)
    for i, w in enumerate(w):
        def f(x):
            x = torch.as_tensor(x, dtype=torch.float)
            return torch.as_tensor(m.phi(x).double() @ w.double(), dtype=torch.double)
        c = approximate_taylor_polynomial(f, 0, deg, 1).coeffs
        coeffs[i] = torch.as_tensor(c.data).flip(0)
    return coeffs.var(0, False)


def calculate_heatmap(models, f):
    heatmap = torch.zeros(len(models), len(models))
    with tqdm(total=heatmap.numel()) as pbar:
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                pbar.set_description(f'Heatmap (cell {i:>2},{j:>2})')
                heatmap[i,j] = f(m1, m2)
                pbar.update(1)
        pbar.set_description('Heatmap done!')
    return heatmap

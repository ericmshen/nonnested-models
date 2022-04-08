import torch
import torch.distributions as ds

OUTPUT_VAR = 0.01


def add_output_noise(x, output_var=OUTPUT_VAR):
    return x + ds.Normal(0, output_var**0.5).sample(x.shape)


def bayes_factor(m1, m2, x, y):
    log_l1 = m1.prior_log_likelihood(x, y)
    log_l2 = m2.prior_log_likelihood(x, y)
    log_bf = log_l1 - log_l2
    return log_bf, log_l1, log_l2


def same_poly_model_family(m1, m2):
    return (
      torch.equal(m1.weights_mean, m2.weights_mean)
      and m1.weights_var == m2.weights_var
      and m1.terms == m2.terms
    )


def polyval(x, coeffs, terms='all'):
  '''
  Numerically stable method of calculating:
    'odd'  ->  w_0 + w_1*x^1 + w_2*x^2 ...
    'even' ->  w_0 + w_1*x^2 + w_2*x^4 ...
    'odd'  ->  w_0 + w_1*x^1 + w_2*x^3 ...
  '''
  # device = x.device if isinstance(x, torch.Tensor) else None
  x = torch.as_tensor(x, dtype=torch.float)
  coeffs = torch.as_tensor(coeffs, dtype=torch.float)
  assert coeffs.ndim == 1
  assert terms in ('even', 'odd', 'all')
  
  coeffs = coeffs.flip(dims=(0,))
  one = torch.ones_like(x)
  y = coeffs[0] * one

  if terms == 'even':
    for c in coeffs[1:]:
      y = y * x**2 + c
  elif terms == 'odd':
    for c in coeffs[1:-1]:
      y = y * x**2 + c
    y = y * x + coeffs[-1]
  else:
    for c in coeffs[1:]:
      y = y * x + c

  return y


# even_poly_basis = lambda x, w: polyval(x, w, terms='even')

# odd_poly_basis = lambda x, w: polyval(x, w, terms='odd')

# poly_basis = lambda x, w: polyval(x, w, terms='all')

# x = 3
# print(polyval(x, [1, 1, 1]) == (1 + x**1 + x**2))
# print(polyval(x, [1, 1, 1], terms='odd') == (1 + x**1 + x**3))
# print(polyval(x, [1, 1, 1], terms='even') == (1 + x**2 + x**4))

# print(polyval(x**2, [1, 1, 1]) == (1**2 + (x**1)**2 + (x**2)**2))
# print(polyval(x**2, [1, 1, 1], terms='odd') == (1**2 + (x**1)**2 + (x**3)**2))
# print(polyval(x**2, [1, 1, 1], terms='even') == (1**2 + (x**2)**2 + (x**4)**2))

# n_parameters=3
# print(polyval(x**2, torch.ones(n_parameters)) == (1**2 + (x**1)**2 + (x**2)**2))
# print(polyval(x**2, torch.ones(n_parameters), terms='odd') == (1**2 + (x**1)**2 + (x**3)**2))
# print(polyval(x**2, torch.ones(n_parameters), terms='even') == (1**2 + (x**2)**2 + (x**4)**2))
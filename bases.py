import torch
from torch import nn, optim


def _init_nlm(dims, activation):
    assert len(dims) >= 2, 'Not enough layers for NLM'

    layers = []
    for i in range(len(dims)-2):
        n_in, n_out = dims[i], dims[i+1]
        layers += [
            nn.Linear(n_in, n_out),
            activation(),
        ]
    basis_block = nn.Sequential(*layers)
    output_layer = nn.Linear(dims[-2], dims[-1], bias=False)
    nlm = nn.Sequential(basis_block, output_layer)
    return nlm, basis_block


def _train(net, x, y, lr=0.01, l2=0.0, epochs=5000):
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    for _ in range(epochs):
        optimizer.zero_grad()
        l = loss(net(x), y)
        l.backward()
        optimizer.step()


def nlm_basis(dims, x, y, activation=nn.GELU):
    nlm, basis_block = _init_nlm(dims, activation)
    _train(nlm, x, y)

    phi = lambda x: basis_block(x.reshape(-1, 1)).detach()
    mu = nlm.state_dict()['1.weight'].flatten()
    return dims[-2], phi, mu


def bias_basis():
    def phi(x):
        x = x.reshape(-1, 1)
        ones = torch.ones_like(x)
        return torch.hstack((ones, x))
    return 2, phi


def poly_basis(deg, bias=True):
    exp = torch.arange(0 if bias else 1, deg + 1)
    phi = lambda x: x.reshape(-1, 1) ** exp
    return deg + 1 if bias else deg, phi

def identity_basis():
    return 1, lambda x: x.reshape(-1, 1)

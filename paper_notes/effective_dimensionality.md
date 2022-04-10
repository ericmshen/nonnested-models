# Effective Dimensionality

## Background: Deep Double Descent
Nakkiran et al., 2019.
* Performance (test error / train error) improves, then worsens, then improves again with increasing model size on many classes of deep learning models (CNNs, ResNets, Transformers)
    * Especially when noise is added
    * Qualitatively, described as classic bias-variance tradeoff predicting worsening performance vs. modern ML approach predicting better performance as model sizes increase
* Peak of worsening error occurs at a "critical regime" where the train set is barely fit
* Given a fixed size, double descent also occurs with an increasing number of epochs
* Hypothesis: At interpolation threshold there is only one model that adequately fits all data, which fails when noise is added; there are no good models that interpolate training set and perform well on test set. With larger sizes, there are more models that can do the above, and SGD can find them.

## Rethinking Parameter Counting in Deep Models: Effective Dimensionality Revisited
Maddox et al. 2020
* In addition to double descent, in settings where NN models have more parameters than training points, there is still good generalization, which goes against the intuition that overparameterization is bad.
* **Main Takeaway**: **Effective Dimensionality** measures the dimensionality of the parameter space determined by the data (the actual number of dimensions used by a model). This is a more insightful way to measure generalization ability than parameter counting. It embodies the number of parameters that have been learned from the data.
* For $A$ a $k\times k$ matrix with eigenvalues $\lambda_i$ and $z$ a regularization constant, effective dimensionality is defined as $N_{eff}(A,z)=\sum_{i=1}^k \frac{\lambda_i}{\lambda_i+z}$.
* Given model $f(x; \theta)$, parameters $\theta\in \mathbb{R}^k$, the Hessian is the $k\times k$ matrix of second derivatives of the loss, $\mathcal{H}_\theta = -\nabla \nabla_\theta \mathcal{L}(\theta, \mathcal{D})$. Usually use negative log loss, $\mathcal{L}=-\log p(\theta|\mathcal{D})$. 
    * The "increase in certainty" in the parameters under the posterior is the **posterior contraction**, the difference in traces of prior and posterior covariance.
    * For Bayesian linear regression $y\sim \mathcal{N}(f=\Phi \beta, \sigma^2 I)$, where $\Phi=\Phi(x)\in \mathbb{R}^{n\times k}$ is a feature map of $n$ data observations to $k>n$ parameters, and we have isotropic prior $\beta \sim \mathcal{N}(0, \alpha^2 I_k)$, the contraction is $\alpha^2 \sum_{i=1}^k \frac{\lambda_i}{\lambda_i+\alpha^{-2}}$, where $\lambda_i$ are the eigenvalues of $\Phi^T \Phi$. 
    * The posterior distribution of $\beta$ has a $k-n$-dimensional subspace where the variance is identical to the prior variance. Only $\min\{n,k\}$ parameters are determined.
* For BNNs, effective dimensionality is inversely proportional to the variance of the posterior distribution: The Hessian of the loss is the inverse of covariance matrix.
* For linear/generalized linear models with isotropic priors, $k$ parameters, and $n<k$ observations, the $k-n$ orthogonal eigenvectors of the Hessian of the loss with the smallest eigenvalues are such that perturbations in the data along these vectors lead to minimal differences in the predictions on training data.
    * Bayesian Linear Models: Given the setting above, the minimal eigenvectors of the Hessian define a $k-n$-dimensional subspace where perturbations in parameters do not change predictions in function space.
    * During training, some eigenvalues become large, while others vanish. Growth of eigenvalues means growth in the curvature of the loss - increased certainty about parameters.
    * Eigenvectors represent which dimensions in parameter space have been determined by the data (posterior has shifted significantly from prior).
* For overparameterized NNs, there are degenerate directions in parameter space from these eigenvectors which lead to function-space homogeneity on test and train data.
    * Degenerate directions do not contain additional functional information - described as **model compression**.
    * As these correspond to eigenvectors near zero, the **loss surface** remains essentially constant when parameters are perturbed in these directions.
    * Degenerate parameter directions also do not create diverse models: the loss remains constant, even if parameters span wide ranges in these directions. (E.g.: NN fully connected classifier for Swiss roll dataset: perturbation of magnitude 10 in 500 minimal eigenvectors results in the same classifier, while perturbation of 3 maximal eigenvectors results in significant differences.)
    * Even with many parameters, NNs can be described by a low-dimensional subspace.
* In terms of the Occam factor for a model $\mathcal{M}$, Evidence = Likelihood times Occam Factor: $p(\mathcal{D}|\mathcal{M})=p(\mathcal{D}|\theta_{MAP}, \mathcal{M})p(\theta_{MAP}|\mathcal{M})\det^{-1/2}(\mathcal{H}_\theta/2\pi)$. As effective dimensionality decreases, eigenvalues decay, so the Occam factor increases.
* **Effective Dimensionality and Model Selection**: Effective dimensionality can serve as a promising factor for model selection. Given two otherwise similar models, choose the one with lower effective dimensionality.
    * Lower effective dimensionality means the model uses a smaller space of parameters, providing better compression and likely generalization.
    * After low training loss is achieved, effective dimensionality is correlated with double descent. E.g.: CNNs with increasing widths actually display lower effective dimensionality, which corresponds to generalization performance
    * At the peak of the double descent phenomenon, effective dimensionality is high because the model has found a sensitive fit to the precise settings of the parameters. As the model grows there are more subspaces which provide more effective data compressions; flat regions of loss become more prominent and discoverable.
    * Effective dimensionality tracks double descent in not only test loss, but also test error, and with width and depth of NNs.
* (Section 7) Effective dimensionality is a better proxy for test loss and test error than PAC-Bayes and path-norm measures. It is also more numerically stable, relatively consistent, and more interpretable.
* Computational notes: Used GPyTorch to get eigenvalues. Only calculate the leading eigenvalues.
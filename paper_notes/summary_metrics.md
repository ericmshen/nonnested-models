# Nonnested Model Selection Criteria (Bruce and Peterson)
* General class of models based on minimizing random distance functions
  * Random distance function can be general: E.g. using log-likelihood (minimize KL distance)
* Proposes its own criteria which chooses the best model with least parameters
  * Choose consistently: inferior models are chosen with probability $\to 0$
  * In general, given distance function $\hat{Q}$, parameters $\beta$, model $f(y,\beta)$, must have form $\hat{Q}(\beta)-\dim(f,Q)\cdot C_T$, where $C_T$ is chosen depending on sample size $T$
  * In nested models need $C_T=o(T)$, $C_T\to \infty$. But for nonnested models, also need $\frac{C_T}{\sqrt{T}}\to \infty$, otherwise inconsistent.
  * Claim NIC, with $C_T=\sqrt{T}\log(T)$, is consistent for nested and nonnested models.
* Bayes factors choose models with best fit objective functions, but do not consistently select the most "parsimonious" model among the best-fit ones
  * The BIC can be characterized as above with $C_T=O(\log T)$, so is inconsistent among nonnested models. 
  * In nonnested models, by comparing BICs as above which are "equally good", the term with $C_T$ does not dominate, so instead the difference between the criteria is random: both models could be selected with positive probabilities. No clear "interpretation"
* Under technical ("stochastic equicontinuity") assumptions, posterior odds ratios and Bayes factors are essentially equivalent to BIC
  * Object function does not need to be smooth (could be simulated)
  * Discussion here is very technical.

# Bayes Factor Consistency (Chib and Kuffner)
* Explores consistency of Bayes factor asymptotically.
  * Consistency: under true probability distribution, $\frac{P(D|\mathcal{M}_1)}{P(D|\mathcal{M}_2)}\to \infty$ if $\mathcal{M}_1$ is the best, $\to 0$ if $\mathcal{M}_2$ is the best
* Decomposes natural log of ratio of marginal likelihoods into log ratios of (1) likelihoods (2) prior densities (3) posterior densities
* In strictly non-nested models there are no parameter settings which produce the same conditional density of observations. 
* Casella et al 2009: "As far as we know, a general consistency result for the Bayesian model selection procedure for non-nested models has not yet been established." Historically for non-nested models, authors have specified a larger encompassing model in which both models are nested
  * The BIC is generally not a consistent model selection criterion when selecting among non-nested models.
  * In nonnested linear regression with opposing sets of predictor variables, the log ratio of posterior densities converges depending on the correctly-specified model.
  * In the case of normal linear models and a wide range of priors, including "intrinsic priors", Bayes factors are consistent.
  * The Generalized Likelihood Ratio framework does not apply in the nonnested case; suggests using inequalities to bound log-likelihood ratio statistics.

# Non-Nested Hypothesis Testing (Pesaran and Weeks)
* Common examples of nonnested models
  * Log-normal vs exponential distributions
  * Linear normal regression models with different sets of conditioning variables
  * Regression models where dependent variables are transformations of the data (e.g. linear vs. log-linear regression)
  * Probit vs. logit specifications
* While model selection plays a part in the decision-making process, it builds on statistical measures of fit, and so is closer to hypothesis testing than it actually is in principle.
  * Typically model selection treats all models symmetrically, whereas hypothesis testing starts out with a different status to the null/alternative hypotheses, treating the models asymetrically.
  * Hypothesis testing does not aim to make a decision, but looks for statistically significant evidence of departure from the null hypothesis (model).
  * Choice of null is important: in nested case the most parsimonious model would be chosen as the null. But for nonnested cases, must choose null on a priori grounds.

# Reconciling the Bayes Factor and Likelihood Ratio for Two Non-Nested Model Selection Problems (Ommen and Saunders)
* 
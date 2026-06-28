# Control Variates for a Bayesian GARCH Model

This project studies variance reduction for Markov Chain Monte Carlo (MCMC) estimators in a Bayesian GARCH(1,1) model. It implements Zero-Variance (ZV) control variates following Mira, Solgi & Imparato (2013), and compares ordinary MCMC estimates with ZV estimators based on OLS, Lasso, and Post-Lasso regression.

The project was developed for the **Simulation & Monte Carlo Methods** course at **ENSAE Paris**.

## Authors

- Baptiste Leloup  
- Mohamed Aziz Maghrebi  
- Alexandre Berlia  

## Project Objective

The goal is to estimate posterior expectations of GARCH parameters more accurately without increasing the number of MCMC iterations.

For a target function \(f(\omega)\), the standard MCMC estimator is:

\[
\hat{\mu}_{MC} = \frac{1}{N}\sum_{i=1}^{N} f(\omega^{(i)}).
\]

This estimator is unbiased asymptotically but can have high Monte Carlo variance. The Zero-Variance approach replaces \(f\) by a corrected function \(\tilde{f}\) such that:

\[
\mathbb{E}_{\pi}[\tilde{f}] = \mathbb{E}_{\pi}[f],
\qquad
\mathrm{Var}_{\pi}(\tilde{f}) \ll \mathrm{Var}_{\pi}(f).
\]

The correction is computed as a post-processing step on the same MCMC chain, so no additional posterior sampling is required.

## Model

The project uses a Gaussian GARCH(1,1) model for log-returns:

\[
r_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0,h_t),
\]

with conditional variance:

\[
h_t = \omega_1 + \omega_2 r_{t-1}^2 + \omega_3 h_{t-1}.
\]

The parameter vector is:

\[
\omega = (\omega_1,\omega_2,\omega_3).
\]

The admissible parameter space is:

\[
\omega_1 > 0,\quad \omega_2 \geq 0,\quad \omega_3 \geq 0,\quad \omega_2+\omega_3 < 1.
\]

The quantity \(\omega_2+\omega_3\) is interpreted as the **persistence of volatility**.

## Methodology

The project is organised around four main questions.

### 1. Bayesian GARCH MCMC Baseline

A Random-Walk Metropolis-Hastings sampler is implemented to sample from the posterior distribution of the GARCH parameters.

Main steps:

1. Simulate a GARCH(1,1) return series.
2. Define the log-likelihood and log-posterior.
3. Implement the Random-Walk Metropolis algorithm.
4. Tune the proposal covariance using the Roberts-Gelman-Gilks scaling rule.
5. Validate the sampler using trace plots, posterior densities, effective sample size, and acceptance rates.
6. Apply the sampler to real EUR/USD daily log-returns.

### 2. First-Order Zero-Variance Control Variates

The first-order ZV correction uses a degree-1 polynomial:

\[
P(\omega)=a^\top \omega.
\]

The corrected function is:

\[
\tilde{f}(\omega)=f(\omega)+a^\top z(\omega),
\]

where:

\[
z(\omega)=-\frac{1}{2}\nabla \log \pi(\omega \mid r).
\]

The optimal coefficient is estimated by OLS regression of the target function on the posterior-score controls.

### 3. Second-Order Controls: OLS, Lasso and Post-Lasso

The control space is enlarged using a degree-2 polynomial:

\[
P(\omega)=a^\top \omega+\frac{1}{2}\omega^\top B\omega.
\]

For a three-dimensional GARCH parameter vector, this produces 9 control variates.

The project compares:

- **ZV-OLS**: full ordinary least squares on all controls.
- **ZV-Lasso**: Lasso regression with cross-validation.
- **ZV-Post-Lasso**: Lasso for variable selection, then OLS refit on the selected variables.

### 4. Regression on Autocorrelated MCMC Chains

Since MCMC draws are autocorrelated, the project investigates whether the regression step should be adjusted.

Three strategies are compared:

- raw post-burn-in chain,
- thinning,
- block averaging.

The empirical result is that using the full raw chain is generally more stable than thinning or block averaging, because the latter reduce the sample size available for regression.

## Main Results

### Simulated GARCH Data

For simulated data with known true parameters, the Random-Walk Metropolis sampler gives stable posterior estimates and good effective sample sizes.

Reference chain example:

```text
Posterior mean: (0.067, 0.097, 0.835)
Posterior sd:   (0.014, 0.012, 0.022)
Acceptance rate: 0.40
Persistence: 0.93
```

### First-Order ZV Variance Reduction

Using degree-1 controls, the variance reduction factors are approximately:

| Target | Variance Reduction Factor |
|---|---:|
| \(\omega_1\) | 13.5 |
| \(\omega_2\) | 49.9 |
| \(\omega_3\) | 24.8 |
| \(\omega_2+\omega_3\) | 18.6 |

### Second-Order ZV Variance Reduction

Using degree-2 controls, ZV-OLS and ZV-Post-Lasso produce much stronger variance reduction:

| Target | ZV-OLS | ZV-Lasso | ZV-Post-Lasso |
|---|---:|---:|---:|
| \(\omega_1\) | 1,612 | 39.9 | 1,612 |
| \(\omega_2\) | 10,394 | 103.8 | 10,394 |
| \(\omega_3\) | 6,274 | 49.0 | 6,274 |
| \(\omega_2+\omega_3\) | 1,389 | 50.2 | 1,389 |

In this low-dimensional setting, OLS performs best because the degree-2 dictionary contains only 9 regressors. Lasso is less effective because shrinkage under-corrects the estimator.

### Real EUR/USD Data

The method is also applied to daily EUR/USD log-returns from 2019 to 2024.

After retuning the MCMC proposal, the sampler reaches an acceptance rate around 0.31. The fitted GARCH model captures volatility clustering in the EUR/USD series.

On real data, degree-2 ZV-OLS reduces variance by roughly **80x to 350x**, smaller than in the simulated setting but still substantial.

## Repository Structure

```text
.
├── projet_garch2_3_corrected.ipynb   # Main notebook with implementation and experiments
├── rapport_garch.pdf                 # Full written report
├── presentation_montecarlo.pdf       # Presentation slides
└── README.md                         # Project documentation
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
# Windows
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib numba scikit-learn yfinance jupyter
```

## How to Run

Open the notebook:

```bash
jupyter notebook projet_garch2_3_corrected.ipynb
```

Then run the cells in order.

The notebook includes:

1. GARCH simulation,
2. posterior log-density and gradient,
3. Random-Walk Metropolis sampler,
4. adaptive proposal calibration,
5. Zero-Variance control variate construction,
6. OLS, Lasso and Post-Lasso estimators,
7. repeated-chain experiments,
8. real EUR/USD application,
9. autocorrelation, thinning and block-averaging checks.

## Important Notes

- The EUR/USD experiment uses `yfinance`, so it requires an internet connection.
- The simulated-data experiments can be run without external data.
- The Zero-Variance estimator is a post-processing method: it improves estimator precision but does not fix a poorly mixing MCMC chain.
- Good MCMC calibration remains necessary before applying ZV controls.
- Boundary effects may matter because the GARCH posterior is constrained by positivity and stationarity conditions.

## Key Functions

The notebook contains the following main functions:

| Function | Purpose |
|---|---|
| `simulate_garch` | Simulate a GARCH(1,1) return series |
| `compute_h` | Compute the conditional variance path |
| `log_posterior` | Evaluate the Bayesian log-posterior |
| `grad_log_posterior` | Compute the posterior score |
| `rwm` | Run Random-Walk Metropolis |
| `adaptive_rwm` | Tune and run the MCMC sampler |
| `build_controls` | Build degree-1 or degree-2 ZV control matrices |
| `zv_estimator_ols` | Compute ZV-OLS estimator |
| `zv_estimator_lasso` | Compute ZV-Lasso estimator |
| `zv_estimator_post_lasso` | Compute ZV-Post-Lasso estimator |
| `effective_sample_size` | Estimate MCMC effective sample size |

## References

- Mira, A., Solgi, R., & Imparato, D. (2013). *Zero variance Markov chain Monte Carlo for Bayesian estimators*. Statistics and Computing, 23(5), 653–662.
- South, L. F., Oates, C. J., Mira, A., & Drovandi, C. (2022). *Regularized zero-variance control variates*. Bayesian Analysis.
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity*. Journal of Econometrics, 31(3), 307–327.
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). *Weak convergence and optimal scaling of random walk Metropolis algorithms*. Annals of Applied Probability.

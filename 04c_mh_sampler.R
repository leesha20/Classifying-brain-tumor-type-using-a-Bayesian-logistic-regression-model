# ============================================================================
# 04c_mh_sampler.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
# ----------------------------------------------------------------------------
# Purpose: Custom Metropolis-Hastings sampler for the posterior distribution
#          of the coefficients β in a multinomial logistic regression model.
#          Fits TWO models with different priors for comparison:
#            Model A: Normal(0, 10) — weakly informative
#            Model B: Normal(0, 2)  — moderately regularizing
#
# Each model: 4 chains × 10,000 iterations, first 2,000 burn-in, thin by 5.
# Warm start from MLE estimates; adaptive proposal tuning during burn-in.
#
# Input:   output/features/prepared_data.rds
#          output/models/frequentist_baseline.rds
# Output:  output/models/bayesian_model_A.rds
#          output/models/bayesian_model_B.rds
# ============================================================================

# --- 1. Setup ----------------------------------------------------------------
rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(mvtnorm)  # multivariate normal proposals

# --- 2. Config ---------------------------------------------------------------
PREPARED_PATH    <- file.path(PROJECT_ROOT, "output", "features", "prepared_data.rds")
FREQ_MODEL_PATH  <- file.path(PROJECT_ROOT, "output", "models", "frequentist_baseline.rds")
MODEL_A_PATH     <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_A.rds")
MODEL_B_PATH     <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_B.rds")

# MCMC settings
N_ITER       <- 10000   # total iterations per chain
N_BURNIN     <- 2000    # burn-in samples to discard
N_THIN       <- 5       # keep every 5th sample
N_CHAINS     <- 4       # number of independent chains
ADAPT_UNTIL  <- 1000    # adapt proposal scale during first 1000 iterations
TARGET_ACCEPT <- 0.234  # target acceptance rate (optimal for high-dim MH)

# --- 3. Load data ------------------------------------------------------------
prepared <- readRDS(PREPARED_PATH)
train_X  <- prepared$train_X
train_y  <- prepared$train_y
feature_names <- prepared$feature_names
class_levels  <- prepared$class_levels

freq_fit <- readRDS(FREQ_MODEL_PATH)

N <- nrow(train_X)
P <- ncol(train_X)            # number of features (11)
K <- length(class_levels)     # number of classes (4)
K1 <- K - 1                   # number of non-reference classes (3)

# Add intercept column: design matrix X is (N × (P+1))
X <- cbind(1, train_X)
colnames(X)[1] <- "(Intercept)"
D <- ncol(X)                  # = P + 1 = 12

# y as integer 1..K (1 = reference = no_tumor)
y_int <- as.integer(train_y)

cat("Data dimensions:\n")
cat("  N (obs):        ", N, "\n")
cat("  D (coefs/class):", D, "\n")
cat("  K (classes):    ", K, "\n")
cat("  K-1 × D (total params):", K1 * D, "\n\n")

# Total number of parameters = (K-1) × (P+1) = 3 × 12 = 36
n_params <- K1 * D

# --- 4. Log-likelihood function ---------------------------------------------
# beta_vec: length n_params = 36, organized as c(beta_class2, beta_class3, beta_class4)
# Each sub-block has length D = 12 (1 intercept + 11 feature coefs)
#
# Multinomial logit:
#   P(y=k | x) = exp(x β_k) / (1 + Σ_{j≠ref} exp(x β_j))   for k ≠ ref
#   P(y=ref | x) = 1 / (1 + Σ_{j≠ref} exp(x β_j))

log_likelihood <- function(beta_vec, X, y_int, K1, D) {
  # Reshape to (D × K1) matrix: each column = coefs for one non-reference class
  B <- matrix(beta_vec, nrow = D, ncol = K1)
  
  # Linear predictors for non-reference classes: eta is N × K1
  eta <- X %*% B
  
  # For each row, softmax denominator = 1 + sum(exp(eta))
  # For numerical stability, we use log-sum-exp trick:
  #   log(1 + Σ exp(eta_k)) = log(exp(0) + Σ exp(eta_k))
  max_eta <- pmax(0, apply(eta, 1, max))  # row-wise max including 0 (for reference class)
  stable_logsumexp <- max_eta + log(exp(-max_eta) + rowSums(exp(eta - max_eta)))
  
  # Log-probability for observed class
  log_probs <- numeric(nrow(X))
  
  # Reference class (y_int == 1): log P = 0 - logsumexp
  ref_idx <- y_int == 1
  log_probs[ref_idx] <- -stable_logsumexp[ref_idx]
  
  # Non-reference classes (y_int in 2..K): log P = eta_{k-1} - logsumexp
  for (k in 2:(K1 + 1)) {
    k_idx <- y_int == k
    log_probs[k_idx] <- eta[k_idx, k - 1] - stable_logsumexp[k_idx]
  }
  
  sum(log_probs)
}

# --- 5. Log-prior function --------------------------------------------------
# Independent Normal(0, sigma^2) prior on every coefficient
log_prior <- function(beta_vec, sigma) {
  sum(dnorm(beta_vec, mean = 0, sd = sigma, log = TRUE))
}

# --- 6. Log-posterior -------------------------------------------------------
log_posterior <- function(beta_vec, X, y_int, K1, D, sigma) {
  log_likelihood(beta_vec, X, y_int, K1, D) +
    log_prior(beta_vec, sigma)
}

# --- 7. Warm-start: initialize chains near MLE -------------------------------
# Frequentist coef matrix has shape (K1 × D). Convert to vector of length 36.
freq_coef <- freq_fit$coef_matrix  # rows = classes 2..K, cols = intercept + features
freq_beta_vec <- as.vector(t(freq_coef))  # flattened in (class-major, feature-minor) order
# Note: t() so the vector goes beta_class2_[intercept, feat1, ..., feat11],
#                                 beta_class3_[intercept, feat1, ..., feat11], ...

# Sanity check: does our log-likelihood at MLE give approximately -deviance/2?
ll_at_mle <- log_likelihood(freq_beta_vec, X, y_int, K1, D)
cat("Sanity check — log-likelihood at MLE:\n")
cat("  Our computation:     ", round(ll_at_mle, 2), "\n")
cat("  Expected (-dev/2):   ", round(-freq_fit$model$deviance / 2, 2), "\n")
cat("  Difference:          ", round(ll_at_mle + freq_fit$model$deviance / 2, 4), "\n")
if (abs(ll_at_mle + freq_fit$model$deviance / 2) > 1) {
  warning("Log-likelihood disagrees with MLE — check parameter ordering!")
} else {
  cat("  ✓ Log-likelihood matches MLE within tolerance.\n")
}
cat("\n")

# --- 8. Core Metropolis-Hastings sampler ------------------------------------
# One chain of MH with adaptive proposal during burn-in phase
run_mh_chain <- function(chain_id, start_beta, sigma_prior,
                         n_iter, n_burnin, n_thin, adapt_until,
                         target_accept, X, y_int, K1, D, n_params,
                         init_prop_sd = 0.05) {
  
  # Storage for post-burnin, post-thinning samples
  n_keep <- floor((n_iter - n_burnin) / n_thin)
  samples <- matrix(NA_real_, nrow = n_keep, ncol = n_params)
  
  # Current state
  beta_curr <- start_beta
  lp_curr   <- log_posterior(beta_curr, X, y_int, K1, D, sigma_prior)
  
  # Adaptive proposal: scalar multiplier on diagonal MVN
  prop_sd    <- init_prop_sd
  n_accept_recent <- 0
  accept_total    <- 0
  total_proposals <- 0
  
  keep_idx  <- 1
  adapt_window <- 50   # check acceptance every 50 iterations during adaptation
  
  start_time <- Sys.time()
  
  for (iter in 1:n_iter) {
    # Propose new beta: current + multivariate normal perturbation
    prop_cov <- (prop_sd^2) * diag(n_params)
    beta_prop <- as.vector(beta_curr + rmvnorm(1, sigma = prop_cov))
    
    # Evaluate log-posterior at proposal
    lp_prop <- log_posterior(beta_prop, X, y_int, K1, D, sigma_prior)
    
    # Metropolis accept/reject
    log_ratio <- lp_prop - lp_curr
    if (log(runif(1)) < log_ratio) {
      beta_curr <- beta_prop
      lp_curr   <- lp_prop
      accept_total    <- accept_total + 1
      n_accept_recent <- n_accept_recent + 1
    }
    total_proposals <- total_proposals + 1
    
    # Adaptive tuning during burn-in: scale proposal SD up if acceptance > target,
    # down if < target. Rule: multiply by exp((rate - target) / 4).
    if (iter <= adapt_until && iter %% adapt_window == 0) {
      recent_rate <- n_accept_recent / adapt_window
      adjustment <- exp((recent_rate - target_accept) / 4)
      prop_sd <- prop_sd * adjustment
      prop_sd <- max(0.001, min(prop_sd, 1.0))  # clamp
      n_accept_recent <- 0
    }
    
    # Store post-burnin samples with thinning
    if (iter > n_burnin && ((iter - n_burnin) %% n_thin == 0)) {
      samples[keep_idx, ] <- beta_curr
      keep_idx <- keep_idx + 1
    }
    
    # Progress print every 1000 iterations
    if (iter %% 1000 == 0) {
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
      rate <- accept_total / total_proposals
      cat(sprintf("  [chain %d]  iter %d/%d  elapsed: %.1f min  accept: %.2f  prop_sd: %.4f\n",
                  chain_id, iter, n_iter, elapsed, rate, prop_sd))
    }
  }
  
  list(
    samples       = samples,
    final_prop_sd = prop_sd,
    accept_rate   = accept_total / total_proposals,
    chain_id      = chain_id
  )
}

# --- 9. Multi-chain wrapper --------------------------------------------------
# Runs N_CHAINS independent chains, each starting from slightly different
# perturbations of the MLE. Different seeds per chain.
fit_bayesian_model <- function(sigma_prior, model_name) {
  cat(sprintf("\n======================================================\n"))
  cat(sprintf("  FITTING MODEL: %s   (prior: Normal(0, %.1f))\n",
              model_name, sigma_prior))
  cat(sprintf("======================================================\n\n"))
  
  chains <- vector("list", N_CHAINS)
  
  for (c in 1:N_CHAINS) {
    set.seed(1000 + c)
    
    # Perturb MLE start to ensure chains begin from different places
    start_beta <- freq_beta_vec + rnorm(n_params, sd = 0.5)
    
    cat(sprintf("--- Chain %d/%d ---\n", c, N_CHAINS))
    chains[[c]] <- run_mh_chain(
      chain_id      = c,
      start_beta    = start_beta,
      sigma_prior   = sigma_prior,
      n_iter        = N_ITER,
      n_burnin      = N_BURNIN,
      n_thin        = N_THIN,
      adapt_until   = ADAPT_UNTIL,
      target_accept = TARGET_ACCEPT,
      X             = X,
      y_int         = y_int,
      K1            = K1,
      D             = D,
      n_params      = n_params
    )
    cat(sprintf("  Chain %d done. Accept rate: %.3f\n\n",
                c, chains[[c]]$accept_rate))
  }
  
  # Package results
  list(
    model_name    = model_name,
    sigma_prior   = sigma_prior,
    chains        = chains,
    feature_names = feature_names,
    class_levels  = class_levels,
    mcmc_settings = list(
      n_iter = N_ITER, n_burnin = N_BURNIN, n_thin = N_THIN,
      n_chains = N_CHAINS, target_accept = TARGET_ACCEPT
    ),
    param_layout  = list(
      K1 = K1, D = D, n_params = n_params,
      non_ref_classes = class_levels[-1],
      coef_names = c("(Intercept)", feature_names)
    )
  )
}

# --- 10. Run both models ----------------------------------------------------
cat("\n===========================================================\n")
cat("STARTING BAYESIAN MODEL FITS (2 models × 4 chains each)\n")
cat("Total: ~45-90 minutes depending on machine\n")
cat("===========================================================\n")

t0 <- Sys.time()

# Model A: weakly informative prior
model_A <- fit_bayesian_model(sigma_prior = 10, model_name = "A_weak_prior")
saveRDS(model_A, MODEL_A_PATH)
cat("Model A saved to:", MODEL_A_PATH, "\n")

# Model B: regularizing prior
model_B <- fit_bayesian_model(sigma_prior = 2, model_name = "B_regularizing_prior")
saveRDS(model_B, MODEL_B_PATH)
cat("Model B saved to:", MODEL_B_PATH, "\n")

t1 <- Sys.time()
cat(sprintf("\n=== TOTAL RUNTIME: %.1f minutes ===\n",
            as.numeric(difftime(t1, t0, units = "mins"))))

# --- 11. Quick acceptance rate summary --------------------------------------
cat("\n=== ACCEPTANCE RATE SUMMARY ===\n")
for (mdl in list(model_A, model_B)) {
  cat(sprintf("\nModel: %s (sigma=%.1f)\n", mdl$model_name, mdl$sigma_prior))
  for (c in seq_along(mdl$chains)) {
    cat(sprintf("  Chain %d: %.3f\n", c, mdl$chains[[c]]$accept_rate))
  }
}
cat("\nTarget was 0.234. Anywhere from 0.15 to 0.35 is acceptable.\n")
cat("If way off, we'll re-run with different initial proposal scale.\n")

cat("\n=== PHASE 4c COMPLETE ===\n")
cat("Next: Phase 4d (convergence diagnostics — R-hat, ESS, trace plots)\n")
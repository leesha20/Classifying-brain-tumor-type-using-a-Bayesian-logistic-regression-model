# ============================================================================
# 04c_mh_sampler_v2.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
# ----------------------------------------------------------------------------
# V2 REVISIONS (after v1 diagnostics showed R-hat > 2 across most params):
#   1. Drop 'dissimilarity' feature (r=0.94 with contrast — multicollinearity)
#   2. Covariance-informed proposal: (2.38^2 / d) * Sigma_MLE
#      instead of scalar * diag(d). Rosenthal (2011) optimal scaling.
#   3. 50,000 iterations per chain (up from 10,000).
#   4. Burn-in 10,000, thin 10. Saves ~4,000 post-processing samples/chain.
#
# Still fits TWO models: Normal(0, 10) vs Normal(0, 2)
# ============================================================================

rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(mvtnorm)

# --- Paths ------------------------------------------------------------------
PREPARED_PATH   <- file.path(PROJECT_ROOT, "output", "features", "prepared_data.rds")
FREQ_MODEL_PATH <- file.path(PROJECT_ROOT, "output", "models", "frequentist_baseline.rds")
MODEL_A_PATH    <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_A.rds")
MODEL_B_PATH    <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_B.rds")

# --- MCMC settings (revised) ------------------------------------------------
N_ITER        <- 50000    # up from 10,000
N_BURNIN      <- 10000    # up from 2,000
N_THIN        <- 10       # up from 5
N_CHAINS      <- 4
ADAPT_UNTIL   <- 5000     # longer adaptation window
TARGET_ACCEPT <- 0.234    # Roberts-Gelman-Gilks optimal for high-dim
DROP_FEATURE  <- "dissimilarity"   # remove this feature

# --- Load & prep data -------------------------------------------------------
prepared <- readRDS(PREPARED_PATH)
feature_names <- prepared$feature_names
class_levels  <- prepared$class_levels

# Drop dissimilarity column
drop_idx <- which(feature_names == DROP_FEATURE)
stopifnot(length(drop_idx) == 1)

train_X <- prepared$train_X[, -drop_idx]
test_X  <- prepared$test_X[,  -drop_idx]
train_y <- prepared$train_y
test_y  <- prepared$test_y
feature_names <- feature_names[-drop_idx]

cat("Dropped feature:", DROP_FEATURE, "\n")
cat("Remaining features (", length(feature_names), "):", paste(feature_names, collapse = ", "), "\n\n")

# Design matrix
N <- nrow(train_X)
P <- ncol(train_X)
K <- length(class_levels)
K1 <- K - 1
X <- cbind(1, train_X)
colnames(X)[1] <- "(Intercept)"
D <- ncol(X)
y_int <- as.integer(train_y)
n_params <- K1 * D

cat("Dimensions:  N =", N, "  D =", D, "  K =", K, "  n_params =", n_params, "\n\n")

# --- Re-fit frequentist with reduced feature set (for warm-start + cov) -----
# We need MLE coefficients and their covariance matrix under the 10-feature setup.
# Easiest: re-fit with nnet::multinom and extract vcov.
library(nnet)
train_df <- data.frame(class = train_y, train_X)
set.seed(42)
freq_fit_v2 <- multinom(class ~ ., data = train_df, maxit = 500, trace = FALSE, Hess = TRUE)

freq_coef_v2 <- coef(freq_fit_v2)   # (K1 x D)
freq_vcov_v2 <- vcov(freq_fit_v2)   # (n_params x n_params)

# The MLE vector in the same ordering as our likelihood expects
freq_beta_vec <- as.vector(t(freq_coef_v2))

# Check that vcov matrix dimensions match our parameter count
stopifnot(ncol(freq_vcov_v2) == n_params)

# --- Log-likelihood (same as v1) --------------------------------------------
log_likelihood <- function(beta_vec, X, y_int, K1, D) {
  B <- matrix(beta_vec, nrow = D, ncol = K1)
  eta <- X %*% B
  max_eta <- pmax(0, apply(eta, 1, max))
  stable_logsumexp <- max_eta + log(exp(-max_eta) + rowSums(exp(eta - max_eta)))
  log_probs <- numeric(nrow(X))
  ref_idx <- y_int == 1
  log_probs[ref_idx] <- -stable_logsumexp[ref_idx]
  for (k in 2:(K1 + 1)) {
    k_idx <- y_int == k
    log_probs[k_idx] <- eta[k_idx, k - 1] - stable_logsumexp[k_idx]
  }
  sum(log_probs)
}

log_prior <- function(beta_vec, sigma) {
  sum(dnorm(beta_vec, 0, sigma, log = TRUE))
}

log_posterior <- function(beta_vec, X, y_int, K1, D, sigma) {
  log_likelihood(beta_vec, X, y_int, K1, D) +
    log_prior(beta_vec, sigma)
}

# Sanity check
ll_at_mle <- log_likelihood(freq_beta_vec, X, y_int, K1, D)
cat("Sanity check — log-likelihood at MLE (v2):\n")
cat("  Ours:               ", round(ll_at_mle, 2), "\n")
cat("  -deviance/2:        ", round(-freq_fit_v2$deviance / 2, 2), "\n")
cat("  Difference:         ", round(ll_at_mle + freq_fit_v2$deviance / 2, 4), "\n")
if (abs(ll_at_mle + freq_fit_v2$deviance / 2) > 1) {
  stop("Log-likelihood disagrees with MLE — halting.")
}
cat("  ✓ Passed.\n\n")

# --- Cholesky-factored proposal covariance ----------------------------------
# Rosenthal (2011): optimal scale = 2.38^2 / d. Use MLE vcov as shape.
base_prop_cov <- (2.38^2 / n_params) * freq_vcov_v2

# Pre-compute Cholesky factor for fast MVN sampling
prop_chol <- chol(base_prop_cov + diag(1e-8, n_params))  # tiny regularization
# So proposal step = prop_sd_scalar * t(prop_chol) %*% z where z ~ N(0, I)

# --- MH sampler with covariance-informed proposal ---------------------------
run_mh_chain <- function(chain_id, start_beta, sigma_prior,
                         n_iter, n_burnin, n_thin, adapt_until,
                         target_accept, X, y_int, K1, D, n_params,
                         prop_chol_base,
                         init_scale = 1.0) {
  
  n_keep <- floor((n_iter - n_burnin) / n_thin)
  samples <- matrix(NA_real_, nrow = n_keep, ncol = n_params)
  
  beta_curr <- start_beta
  lp_curr   <- log_posterior(beta_curr, X, y_int, K1, D, sigma_prior)
  
  prop_scale <- init_scale
  n_accept_recent <- 0
  accept_total    <- 0
  total_proposals <- 0
  keep_idx        <- 1
  adapt_window    <- 100
  
  start_time <- Sys.time()
  
  for (iter in 1:n_iter) {
    # Draw proposal: beta' = beta + scale * L %*% z where LL^T = base_prop_cov
    z <- rnorm(n_params)
    step <- prop_scale * as.vector(crossprod(prop_chol_base, z))
    beta_prop <- beta_curr + step
    
    lp_prop <- log_posterior(beta_prop, X, y_int, K1, D, sigma_prior)
    log_ratio <- lp_prop - lp_curr
    
    if (log(runif(1)) < log_ratio) {
      beta_curr <- beta_prop
      lp_curr   <- lp_prop
      accept_total    <- accept_total + 1
      n_accept_recent <- n_accept_recent + 1
    }
    total_proposals <- total_proposals + 1
    
    # Adaptive: adjust scalar multiplier on proposal covariance
    if (iter <= adapt_until && iter %% adapt_window == 0) {
      recent_rate <- n_accept_recent / adapt_window
      adjustment <- exp((recent_rate - target_accept) / 4)
      prop_scale <- prop_scale * adjustment
      prop_scale <- max(0.05, min(prop_scale, 10))
      n_accept_recent <- 0
    }
    
    if (iter > n_burnin && ((iter - n_burnin) %% n_thin == 0)) {
      samples[keep_idx, ] <- beta_curr
      keep_idx <- keep_idx + 1
    }
    
    if (iter %% 5000 == 0) {
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
      rate <- accept_total / total_proposals
      cat(sprintf("  [chain %d]  iter %d/%d  elapsed: %.1f min  accept: %.2f  scale: %.3f\n",
                  chain_id, iter, n_iter, elapsed, rate, prop_scale))
    }
  }
  
  list(samples = samples,
       final_prop_scale = prop_scale,
       accept_rate = accept_total / total_proposals,
       chain_id = chain_id)
}

# --- Multi-chain wrapper ----------------------------------------------------
fit_bayesian_model <- function(sigma_prior, model_name) {
  cat(sprintf("\n======================================================\n"))
  cat(sprintf("  FITTING MODEL: %s   (prior: Normal(0, %.1f))\n",
              model_name, sigma_prior))
  cat(sprintf("======================================================\n\n"))
  
  chains <- vector("list", N_CHAINS)
  for (c in 1:N_CHAINS) {
    set.seed(2000 + c)
    # Start each chain from a perturbation of MLE
    start_beta <- freq_beta_vec + rnorm(n_params, sd = 0.3)
    cat(sprintf("--- Chain %d/%d ---\n", c, N_CHAINS))
    chains[[c]] <- run_mh_chain(
      chain_id = c, start_beta = start_beta, sigma_prior = sigma_prior,
      n_iter = N_ITER, n_burnin = N_BURNIN, n_thin = N_THIN,
      adapt_until = ADAPT_UNTIL, target_accept = TARGET_ACCEPT,
      X = X, y_int = y_int, K1 = K1, D = D, n_params = n_params,
      prop_chol_base = prop_chol
    )
    cat(sprintf("  Chain %d done. Accept rate: %.3f\n\n", c, chains[[c]]$accept_rate))
  }
  
  list(
    model_name = model_name,
    sigma_prior = sigma_prior,
    chains = chains,
    feature_names = feature_names,
    class_levels = class_levels,
    mcmc_settings = list(
      n_iter = N_ITER, n_burnin = N_BURNIN, n_thin = N_THIN,
      n_chains = N_CHAINS, target_accept = TARGET_ACCEPT,
      proposal_type = "covariance-informed (2.38^2/d * Sigma_MLE)"
    ),
    param_layout = list(
      K1 = K1, D = D, n_params = n_params,
      non_ref_classes = class_levels[-1],
      coef_names = c("(Intercept)", feature_names)
    ),
    dropped_feature = DROP_FEATURE,
    freq_fit_v2 = freq_fit_v2   # keep for Phase 4d/4e convenience
  )
}

# --- Run --------------------------------------------------------------------
cat("\n===========================================================\n")
cat("V2 SAMPLER: covariance proposal, 50k iter/chain, drop", DROP_FEATURE, "\n")
cat("Expected runtime: ~15-25 minutes\n")
cat("===========================================================\n")

t0 <- Sys.time()

model_A <- fit_bayesian_model(sigma_prior = 10, model_name = "A_weak_prior")
saveRDS(model_A, MODEL_A_PATH)
cat("Model A saved to:", MODEL_A_PATH, "\n")

model_B <- fit_bayesian_model(sigma_prior = 2, model_name = "B_regularizing_prior")
saveRDS(model_B, MODEL_B_PATH)
cat("Model B saved to:", MODEL_B_PATH, "\n")

t1 <- Sys.time()
cat(sprintf("\n=== TOTAL RUNTIME: %.1f minutes ===\n",
            as.numeric(difftime(t1, t0, units = "mins"))))

cat("\n=== ACCEPTANCE RATE SUMMARY ===\n")
for (mdl in list(model_A, model_B)) {
  cat(sprintf("\n%s (sigma=%.1f)\n", mdl$model_name, mdl$sigma_prior))
  for (c in seq_along(mdl$chains)) {
    cat(sprintf("  Chain %d: %.3f\n", c, mdl$chains[[c]]$accept_rate))
  }
}

cat("\n=== PHASE 4c V2 COMPLETE ===\n")
cat("Next: re-run 04d_diagnostics.R (after fixing the printf error)\n")
# ============================================================================
# 04e_evaluate.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
# ----------------------------------------------------------------------------
# Purpose: Evaluate both Bayesian models on held-out test set.
#          Compute posterior predictive class probabilities, point predictions
#          via argmax of mean posterior prob, classification metrics, and
#          predictive uncertainty (entropy of predictive distribution).
#          Compare against frequentist MLE baseline.
#
# Input:   output/features/prepared_data.rds
#          output/models/bayesian_model_A.rds
#          output/models/bayesian_model_B.rds
# Output:  output/models/evaluation_results.rds
#          output/models/evaluation_summary.txt
# ============================================================================

rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(nnet)

# --- Paths ------------------------------------------------------------------
PREPARED_PATH <- file.path(PROJECT_ROOT, "output", "features", "prepared_data.rds")
MODEL_A_PATH  <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_A.rds")
MODEL_B_PATH  <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_B.rds")
EVAL_RDS_PATH <- file.path(PROJECT_ROOT, "output", "models", "evaluation_results.rds")
EVAL_TXT_PATH <- file.path(PROJECT_ROOT, "output", "models", "evaluation_summary.txt")

# --- Load data and models ---------------------------------------------------
prepared <- readRDS(PREPARED_PATH)
model_A  <- readRDS(MODEL_A_PATH)
model_B  <- readRDS(MODEL_B_PATH)

# We need the REDUCED feature set (without dissimilarity)
feature_names_v2 <- model_A$feature_names
drop_feat <- model_A$dropped_feature

train_X <- prepared$train_X[, feature_names_v2]
test_X  <- prepared$test_X[,  feature_names_v2]
train_y <- prepared$train_y
test_y  <- prepared$test_y

class_levels <- prepared$class_levels
K  <- length(class_levels)
K1 <- K - 1

# Design matrices (with intercept)
X_test  <- cbind(1, test_X)
colnames(X_test)[1] <- "(Intercept)"
D <- ncol(X_test)
N_test <- nrow(X_test)

cat("Test set:", N_test, "obs,", D, "design cols,", K, "classes\n\n")

# --- Helper: predict class probabilities from a single beta vector ----------
# Returns N_test x K matrix of predicted probabilities (reference class first)
predict_probs <- function(beta_vec, X, K1, D) {
  B <- matrix(beta_vec, nrow = D, ncol = K1)  # D x K1
  eta <- X %*% B                              # N x K1
  
  # Softmax with reference class 0-logit (first column)
  # P(k) = exp(eta_k) / (1 + sum_j exp(eta_j)) for k > ref
  # P(ref) = 1 / (1 + sum_j exp(eta_j))
  max_eta <- pmax(0, apply(eta, 1, max))
  denom   <- exp(-max_eta) + rowSums(exp(eta - max_eta))
  
  probs <- matrix(NA_real_, nrow = nrow(X), ncol = K1 + 1)
  probs[, 1] <- exp(-max_eta) / denom
  for (k in 1:K1) {
    probs[, k + 1] <- exp(eta[, k] - max_eta) / denom
  }
  probs
}

# --- Posterior predictive: average probabilities over all MCMC samples -----
posterior_predict <- function(model_obj, X_test, K1, D) {
  # Pool all chains into one big matrix of samples (S x n_params)
  all_samples <- do.call(rbind, lapply(model_obj$chains, function(c) c$samples))
  S <- nrow(all_samples)
  cat("  Using", S, "posterior samples\n")
  
  # Accumulate mean probabilities across samples
  # For efficiency, we work in batches (though not strictly necessary here)
  mean_probs <- matrix(0, nrow = nrow(X_test), ncol = K1 + 1)
  
  for (s in seq_len(S)) {
    p_s <- predict_probs(all_samples[s, ], X_test, K1, D)
    mean_probs <- mean_probs + p_s
  }
  mean_probs <- mean_probs / S
  
  # For each test obs, also compute prediction uncertainty (entropy of mean dist)
  # AND posterior predictive variance (variability in predicted probs across samples)
  # For computational tractability, compute variance by sampling
  set.seed(123)
  n_sub <- min(500, S)
  sub_idx <- sample.int(S, n_sub)
  prob_cube <- array(NA_real_, dim = c(nrow(X_test), K1 + 1, n_sub))
  for (i in seq_along(sub_idx)) {
    prob_cube[, , i] <- predict_probs(all_samples[sub_idx[i], ], X_test, K1, D)
  }
  
  # Posterior SD of predicted probability for each (test obs, class)
  prob_sd <- apply(prob_cube, c(1, 2), sd)
  
  # Entropy of MEAN predictive distribution per test obs (information-theoretic uncertainty)
  entropy <- apply(mean_probs, 1, function(p) {
    p <- p[p > 0]
    -sum(p * log2(p))
  })
  
  list(
    mean_probs  = mean_probs,
    prob_sd     = prob_sd,
    entropy     = entropy,
    n_samples   = S
  )
}

# --- Predict with each Bayesian model --------------------------------------
cat("Posterior predictive: Model A\n")
pred_A <- posterior_predict(model_A, X_test, K1, D)

cat("Posterior predictive: Model B\n")
pred_B <- posterior_predict(model_B, X_test, K1, D)

# --- Frequentist baseline on SAME reduced feature set ----------------------
# Use the pre-fit v2 frequentist stored inside model_A
freq_fit_v2 <- model_A$freq_fit_v2
test_df <- data.frame(class = test_y, test_X)
freq_probs <- predict(freq_fit_v2, newdata = test_df, type = "probs")
# nnet::multinom returns probs in alphabetic/factor order; check alignment
# Our class_levels = c("no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor")
# Reorder columns to match
freq_probs <- freq_probs[, class_levels]

# --- Point predictions via argmax of mean predictive probability ----------
pred_class <- function(prob_mat, class_levels) {
  idx <- max.col(prob_mat, ties.method = "first")
  factor(class_levels[idx], levels = class_levels)
}

pred_A_class <- pred_class(pred_A$mean_probs, class_levels)
pred_B_class <- pred_class(pred_B$mean_probs, class_levels)
pred_F_class <- pred_class(freq_probs,        class_levels)

# --- Evaluation metrics ----------------------------------------------------
compute_metrics <- function(y_true, y_pred, class_levels) {
  cm <- table(Actual = y_true, Predicted = y_pred)
  acc <- sum(diag(cm)) / sum(cm)
  
  metrics <- data.frame(
    class = class_levels,
    precision = NA_real_, recall = NA_real_, f1 = NA_real_, support = NA_integer_
  )
  for (k in seq_along(class_levels)) {
    cls <- class_levels[k]
    tp <- sum(y_true == cls & y_pred == cls)
    fp <- sum(y_true != cls & y_pred == cls)
    fn <- sum(y_true == cls & y_pred != cls)
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA
    rec  <- if ((tp + fn) > 0) tp / (tp + fn) else NA
    f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0)
      2 * prec * rec / (prec + rec) else NA
    metrics$precision[k] <- round(prec, 3)
    metrics$recall[k]    <- round(rec, 3)
    metrics$f1[k]        <- round(f1, 3)
    metrics$support[k]   <- sum(y_true == cls)
  }
  list(cm = cm, accuracy = acc, per_class = metrics,
       macro_f1 = mean(metrics$f1, na.rm = TRUE))
}

metrics_F <- compute_metrics(test_y, pred_F_class, class_levels)
metrics_A <- compute_metrics(test_y, pred_A_class, class_levels)
metrics_B <- compute_metrics(test_y, pred_B_class, class_levels)

# --- Reporting -------------------------------------------------------------
print_report <- function(name, m, pred_info = NULL) {
  cat(sprintf("\n--- %s ---\n", name))
  cat("Confusion matrix:\n"); print(m$cm)
  cat(sprintf("\nAccuracy: %.2f%% (%d / %d)\n",
              m$accuracy * 100, sum(diag(m$cm)), sum(m$cm)))
  cat("Per-class metrics:\n"); print(m$per_class)
  cat(sprintf("Macro F1: %.3f\n", m$macro_f1))
  if (!is.null(pred_info)) {
    cat(sprintf("Mean predictive entropy: %.3f  (max 2.0 = maximal uncertainty)\n",
                mean(pred_info$entropy)))
    cat(sprintf("Mean posterior SD of predicted probs: %.3f\n",
                mean(pred_info$prob_sd)))
  }
}

cat("\n================ TEST SET EVALUATION ================\n")
print_report("FREQUENTIST MLE (baseline)", metrics_F)
print_report("BAYESIAN Model A (Normal(0, 10) prior)", metrics_A, pred_A)
print_report("BAYESIAN Model B (Normal(0, 2) prior)",  metrics_B, pred_B)

# --- Head-to-head summary --------------------------------------------------
cat("\n============ MODEL COMPARISON ============\n")
comp <- data.frame(
  Model       = c("Frequentist MLE (v2)", "Bayesian A (N(0,10))", "Bayesian B (N(0,2))"),
  Accuracy    = sprintf("%.2f%%", 100 * c(metrics_F$accuracy, metrics_A$accuracy, metrics_B$accuracy)),
  Macro_F1    = round(c(metrics_F$macro_f1, metrics_A$macro_f1, metrics_B$macro_f1), 3),
  Glioma_F1   = c(metrics_F$per_class$f1[2], metrics_A$per_class$f1[2], metrics_B$per_class$f1[2]),
  NoTumor_F1  = c(metrics_F$per_class$f1[1], metrics_A$per_class$f1[1], metrics_B$per_class$f1[1]),
  stringsAsFactors = FALSE
)
print(comp, row.names = FALSE)

# --- Save everything -------------------------------------------------------
eval_obj <- list(
  test_y         = test_y,
  class_levels   = class_levels,
  frequentist    = list(probs = freq_probs, pred_class = pred_F_class, metrics = metrics_F),
  model_A        = list(probs = pred_A$mean_probs, prob_sd = pred_A$prob_sd,
                        entropy = pred_A$entropy, pred_class = pred_A_class,
                        metrics = metrics_A, n_samples = pred_A$n_samples),
  model_B        = list(probs = pred_B$mean_probs, prob_sd = pred_B$prob_sd,
                        entropy = pred_B$entropy, pred_class = pred_B_class,
                        metrics = metrics_B, n_samples = pred_B$n_samples),
  comparison_tbl = comp
)
saveRDS(eval_obj, EVAL_RDS_PATH)

sink(EVAL_TXT_PATH)
cat("TEST SET EVALUATION — STAT 5353 FINAL PROJECT\n")
cat(rep("=", 70), sep = ""); cat("\n\n")
print_report("FREQUENTIST MLE (baseline)", metrics_F)
print_report("BAYESIAN Model A (Normal(0, 10) prior)", metrics_A, pred_A)
print_report("BAYESIAN Model B (Normal(0, 2) prior)",  metrics_B, pred_B)
cat("\n============ MODEL COMPARISON ============\n")
print(comp, row.names = FALSE)
sink()

cat("\n=== PHASE 4e COMPLETE ===\n")
cat("Evaluation RDS:  ", EVAL_RDS_PATH, "\n")
cat("Evaluation text: ", EVAL_TXT_PATH, "\n")
# ============================================================================
# 04b_frequentist_baseline.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
# ----------------------------------------------------------------------------
# Purpose: Fit a frequentist multinomial logistic regression via MLE as a
#          baseline for comparison with the Bayesian model. Establishes
#          expected classification performance and coefficient scale.
#
# Input:   output/features/prepared_data.rds
# Output:  output/models/frequentist_baseline.rds
#          output/models/frequentist_results.txt (human-readable summary)
# ============================================================================

# --- 1. Setup ----------------------------------------------------------------
rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(nnet)     # for multinom()
library(dplyr)

# --- 2. Config ---------------------------------------------------------------
PREPARED_PATH  <- file.path(PROJECT_ROOT, "output", "features", "prepared_data.rds")
MODELS_DIR     <- file.path(PROJECT_ROOT, "output", "models")
MODEL_PATH     <- file.path(MODELS_DIR, "frequentist_baseline.rds")
RESULTS_TXT    <- file.path(MODELS_DIR, "frequentist_results.txt")

dir.create(MODELS_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 3. Load prepared data ---------------------------------------------------
prepared <- readRDS(PREPARED_PATH)
train_X       <- prepared$train_X
train_y       <- prepared$train_y
test_X        <- prepared$test_X
test_y        <- prepared$test_y
feature_names <- prepared$feature_names
class_levels  <- prepared$class_levels

cat("Training set:  ", nrow(train_X), "obs,", ncol(train_X), "features\n")
cat("Test set:      ", nrow(test_X),  "obs,", ncol(test_X),  "features\n")
cat("Feature names: ", paste(feature_names, collapse = ", "), "\n")
cat("Class levels:  ", paste(class_levels,  collapse = ", "), "\n\n")

# --- 4. Fit multinomial logistic regression via MLE -------------------------
# nnet::multinom() uses the no_tumor class as reference (first level of factor)
# and returns coefficients for the other 3 classes relative to no_tumor.

train_df <- data.frame(class = train_y, train_X)

cat("Fitting frequentist multinomial logistic regression...\n")
set.seed(42)
freq_fit <- multinom(class ~ ., data = train_df, maxit = 500, trace = FALSE)

cat("Converged:", freq_fit$convergence == 0, "\n")
cat("Final deviance:", round(freq_fit$deviance, 2), "\n")
cat("AIC:           ", round(freq_fit$AIC, 2),      "\n\n")

# --- 5. Extract coefficients and standard errors ----------------------------
coef_mat <- coef(freq_fit)  # rows = classes (excluding reference), cols = intercept + features
se_mat   <- summary(freq_fit)$standard.errors

cat("=== COEFFICIENT ESTIMATES (log-odds vs no_tumor) ===\n")
for (cls in rownames(coef_mat)) {
  cat("\n--- Class:", cls, "---\n")
  out <- data.frame(
    estimate = round(coef_mat[cls, ], 3),
    std_err  = round(se_mat[cls, ],  3),
    z_value  = round(coef_mat[cls, ] / se_mat[cls, ], 2)
  )
  print(out)
}

# Coefficient scale — useful for prior specification in Phase 4c
abs_coefs <- abs(coef_mat[, -1])  # exclude intercepts
cat("\n=== COEFFICIENT MAGNITUDE (informs Bayesian prior choice) ===\n")
cat("Absolute value stats of slope coefficients:\n")
cat("  min:    ", round(min(abs_coefs), 2), "\n")
cat("  median: ", round(median(abs_coefs), 2), "\n")
cat("  max:    ", round(max(abs_coefs), 2), "\n")

# --- 6. Predict on test set --------------------------------------------------
test_df <- data.frame(class = test_y, test_X)
test_pred_prob <- predict(freq_fit, newdata = test_df, type = "probs")
test_pred      <- predict(freq_fit, newdata = test_df, type = "class")

# --- 7. Evaluation metrics --------------------------------------------------
cat("\n=== TEST SET EVALUATION ===\n")

# Confusion matrix
cm <- table(Actual = test_y, Predicted = test_pred)
cat("\nConfusion matrix:\n")
print(cm)

# Overall accuracy
accuracy <- sum(diag(cm)) / sum(cm)
cat(sprintf("\nOverall accuracy: %.2f%% (%d / %d correct)\n",
            accuracy * 100, sum(diag(cm)), sum(cm)))

# Per-class precision, recall, F1
cat("\nPer-class metrics:\n")
metrics <- data.frame(
  class     = class_levels,
  precision = rep(NA, length(class_levels)),
  recall    = rep(NA, length(class_levels)),
  f1        = rep(NA, length(class_levels)),
  support   = rep(NA, length(class_levels))
)

for (k in seq_along(class_levels)) {
  cls <- class_levels[k]
  tp <- sum(test_y == cls & test_pred == cls)
  fp <- sum(test_y != cls & test_pred == cls)
  fn <- sum(test_y == cls & test_pred != cls)
  
  prec <- if (tp + fp > 0) tp / (tp + fp) else NA
  rec  <- if (tp + fn > 0) tp / (tp + fn) else NA
  f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) {
    2 * prec * rec / (prec + rec)
  } else NA
  
  metrics$precision[k] <- round(prec, 3)
  metrics$recall[k]    <- round(rec,  3)
  metrics$f1[k]        <- round(f1,   3)
  metrics$support[k]   <- sum(test_y == cls)
}
print(metrics)

macro_f1 <- mean(metrics$f1, na.rm = TRUE)
cat(sprintf("\nMacro-averaged F1: %.3f\n", macro_f1))

# --- 8. Save model and results ---------------------------------------------
baseline_results <- list(
  model           = freq_fit,
  coef_matrix     = coef_mat,
  se_matrix       = se_mat,
  test_pred_prob  = test_pred_prob,
  test_pred_class = test_pred,
  test_y          = test_y,
  confusion_matrix = cm,
  accuracy        = accuracy,
  per_class_metrics = metrics,
  macro_f1        = macro_f1,
  coef_magnitude_stats = c(
    min    = min(abs_coefs),
    median = median(abs_coefs),
    max    = max(abs_coefs)
  )
)

saveRDS(baseline_results, MODEL_PATH)

# Write a human-readable summary too
sink(RESULTS_TXT)
cat("FREQUENTIST BASELINE — Multinomial Logistic Regression via MLE\n")
cat(rep("=", 70), sep = ""); cat("\n\n")
cat("Training set:", nrow(train_X), "observations,", ncol(train_X), "features\n")
cat("Test set:    ", nrow(test_X),  "observations\n\n")
cat("Converged:", freq_fit$convergence == 0, "\n")
cat("Deviance: ", round(freq_fit$deviance, 2), "\n")
cat("AIC:      ", round(freq_fit$AIC, 2), "\n\n")
cat("Test accuracy: ", sprintf("%.2f%%", accuracy * 100), "\n")
cat("Macro F1:      ", round(macro_f1, 3), "\n\n")
cat("Confusion matrix:\n"); print(cm)
cat("\nPer-class metrics:\n"); print(metrics)
cat("\nCoefficient estimates:\n"); print(round(coef_mat, 3))
sink()

cat("\n=== PHASE 4b COMPLETE ===\n")
cat("Model saved to:  ", MODEL_PATH, "\n")
cat("Summary saved to:", RESULTS_TXT, "\n")
cat("\nThis accuracy is your TARGET — Bayesian model should match or exceed it.\n")
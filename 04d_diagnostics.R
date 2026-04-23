# ============================================================================
# 04d_diagnostics.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
# ----------------------------------------------------------------------------
# Purpose: MCMC convergence diagnostics and posterior summaries for both
#          Bayesian models. Produces R-hat, effective sample size, trace
#          plots, posterior means/SDs/credible intervals, and a side-by-side
#          comparison with the frequentist MLE.
#
# Input:   output/models/bayesian_model_A.rds
#          output/models/bayesian_model_B.rds
#          output/models/frequentist_baseline.rds
# Output:  output/figures/trace_plots_A.png
#          output/figures/trace_plots_B.png
#          output/models/posterior_summary_A.csv
#          output/models/posterior_summary_B.csv
#          output/models/diagnostics_report.txt
# ============================================================================

# --- 1. Setup ----------------------------------------------------------------
rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(coda)       # gelman.diag, effectiveSize
library(dplyr)

# --- 2. Config ---------------------------------------------------------------
MODEL_A_PATH    <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_A.rds")
MODEL_B_PATH    <- file.path(PROJECT_ROOT, "output", "models", "bayesian_model_B.rds")
FREQ_PATH       <- file.path(PROJECT_ROOT, "output", "models", "frequentist_baseline.rds")
FIGURES_DIR     <- file.path(PROJECT_ROOT, "output", "figures")
REPORT_PATH     <- file.path(PROJECT_ROOT, "output", "models", "diagnostics_report.txt")

dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 3. Load ---------------------------------------------------------------
model_A  <- readRDS(MODEL_A_PATH)
model_B  <- readRDS(MODEL_B_PATH)
freq_fit <- readRDS(FREQ_PATH)

# Build parameter names: "class : coefname"
param_names <- character(0)
for (cls in model_A$param_layout$non_ref_classes) {
  for (coef in model_A$param_layout$coef_names) {
    param_names <- c(param_names, paste0(cls, " : ", coef))
  }
}
cat("Parameter count:", length(param_names), "\n\n")

# --- 4. Helper: convert chains to coda mcmc.list ----------------------------
# Each chain's samples is a matrix (n_keep × n_params). coda expects mcmc objects.
to_mcmc_list <- function(model_obj) {
  chains <- lapply(model_obj$chains, function(ch) {
    m <- as.matrix(ch$samples)
    colnames(m) <- param_names
    mcmc(m)
  })
  mcmc.list(chains)
}

mcmc_A <- to_mcmc_list(model_A)
mcmc_B <- to_mcmc_list(model_B)

cat("Chain dimensions:\n")
cat("  Model A: ", N_A <- nrow(mcmc_A[[1]]), "samples ×",
    length(mcmc_A), "chains\n")
cat("  Model B: ", N_B <- nrow(mcmc_B[[1]]), "samples ×",
    length(mcmc_B), "chains\n\n")

# --- 5. Compute R-hat and ESS ----------------------------------------------
compute_diagnostics <- function(mcmc_list, model_name) {
  cat("Computing diagnostics for", model_name, "...\n")
  
  # Gelman-Rubin R-hat (multivariate = FALSE gives per-parameter)
  gr <- gelman.diag(mcmc_list, multivariate = FALSE, autoburnin = FALSE)
  rhat <- gr$psrf[, "Point est."]
  rhat_upper <- gr$psrf[, "Upper C.I."]
  
  # Effective sample size (sum across chains)
  ess <- effectiveSize(mcmc_list)
  
  data.frame(
    parameter   = param_names,
    r_hat       = round(rhat, 4),
    r_hat_upper = round(rhat_upper, 4),
    ess         = round(ess, 0),
    stringsAsFactors = FALSE
  )
}

diag_A <- compute_diagnostics(mcmc_A, "Model A")
diag_B <- compute_diagnostics(mcmc_B, "Model B")

# --- 6. Posterior summary statistics ---------------------------------------
summarize_posterior <- function(mcmc_list, model_name) {
  # Combine all chains for summary statistics
  combined <- do.call(rbind, mcmc_list)
  
  data.frame(
    parameter  = param_names,
    post_mean  = round(apply(combined, 2, mean), 4),
    post_sd    = round(apply(combined, 2, sd),   4),
    ci_lower   = round(apply(combined, 2, quantile, 0.025), 4),
    ci_upper   = round(apply(combined, 2, quantile, 0.975), 4),
    stringsAsFactors = FALSE
  )
}

post_A <- summarize_posterior(mcmc_A, "Model A")
post_B <- summarize_posterior(mcmc_B, "Model B")

# --- 7. Merge diagnostics + posterior into one table -----------------------
summary_A <- merge(post_A, diag_A, by = "parameter", sort = FALSE)
summary_B <- merge(post_B, diag_B, by = "parameter", sort = FALSE)

write.csv(summary_A,
          file.path(PROJECT_ROOT, "output", "models", "posterior_summary_A.csv"),
          row.names = FALSE)
write.csv(summary_B,
          file.path(PROJECT_ROOT, "output", "models", "posterior_summary_B.csv"),
          row.names = FALSE)

# --- 8. Convergence flags ---------------------------------------------------
flag_convergence <- function(diag_df, model_name) {
  n_bad_rhat <- sum(diag_df$r_hat > 1.1)
  n_marginal <- sum(diag_df$r_hat > 1.01 & diag_df$r_hat <= 1.1)
  n_low_ess  <- sum(diag_df$ess < 400)
  
  cat(sprintf("\n--- %s convergence summary ---\n", model_name))
  cat(sprintf("  R-hat > 1.1  (re-run needed): %d / %d params\n",
              n_bad_rhat, nrow(diag_df)))
  cat(sprintf("  R-hat 1.01-1.1 (marginal):    %d / %d params\n",
              n_marginal, nrow(diag_df)))
  cat(sprintf("  ESS < 400      (low info):    %d / %d params\n",
              n_low_ess, nrow(diag_df)))
  cat(sprintf("  Max R-hat:     %.4f\n", max(diag_df$r_hat)))
  cat(sprintf("  Min ESS:       %d\n", min(diag_df$ess)))
  cat(sprintf("  Median ESS:    %.0f\n", median(diag_df$ess)))
  
  if (n_bad_rhat > 0) {
    cat("\n  Parameters with R-hat > 1.1:\n")
    bad <- diag_df[diag_df$r_hat > 1.1, c("parameter", "r_hat", "ess")]
    print(bad)
  }
}

flag_convergence(diag_A, "Model A (Normal(0, 10) prior)")
flag_convergence(diag_B, "Model B (Normal(0, 2) prior)")

# --- 9. Trace plots ---------------------------------------------------------
# 36 parameters × 4 chains = a lot. Plot in a 6x6 grid.
plot_traces <- function(mcmc_list, model_name, out_path) {
  n_params <- ncol(mcmc_list[[1]])
  png(out_path, width = 1400, height = 1800, res = 120)
  par(mfrow = c(9, 4), mar = c(2, 2.5, 1.8, 0.5), oma = c(0, 0, 2, 0),
      cex.main = 0.75, cex.axis = 0.6)
  
  chain_colors <- c("black", "red", "blue", "darkgreen")
  
  for (p in 1:n_params) {
    # Build plot range across all chains for this parameter
    ylim_p <- range(sapply(mcmc_list, function(m) range(m[, p])))
    plot(mcmc_list[[1]][, p], type = "l", col = chain_colors[1],
         ylim = ylim_p, main = param_names[p],
         xlab = "", ylab = "", lwd = 0.5)
    for (c in 2:length(mcmc_list)) {
      lines(mcmc_list[[c]][, p], col = chain_colors[c], lwd = 0.5)
    }
  }
  mtext(paste("Trace plots —", model_name), outer = TRUE, cex = 1.1, font = 2)
  dev.off()
  cat("  Saved trace plots to:", out_path, "\n")
}

cat("\nGenerating trace plots...\n")
plot_traces(mcmc_A, "Model A (Normal(0, 10) prior)",
            file.path(FIGURES_DIR, "trace_plots_A.png"))
plot_traces(mcmc_B, "Model B (Normal(0, 2) prior)",
            file.path(FIGURES_DIR, "trace_plots_B.png"))

# --- 10. Prior impact comparison: MLE vs Model A vs Model B ----------------
# Use the re-fit frequentist from v2 (without dissimilarity), which was saved
# inside model_A and model_B objects.
freq_coef_v2_mat <- coef(model_A$freq_fit_v2)
freq_coef_vec    <- as.vector(t(freq_coef_v2_mat))

stopifnot(length(freq_coef_vec) == length(param_names))

compare_df <- data.frame(
  parameter  = param_names,
  mle        = round(freq_coef_vec, 3),
  post_A     = post_A$post_mean,
  post_B     = post_B$post_mean,
  shrinkage_A = round(post_A$post_mean - freq_coef_vec, 3),
  shrinkage_B = round(post_B$post_mean - freq_coef_vec, 3),
  stringsAsFactors = FALSE
)

cat("\n=== PRIOR IMPACT COMPARISON (top 10 by |MLE magnitude|) ===\n")
ord <- order(-abs(compare_df$mle))
print(compare_df[ord[1:10], ])

cat("\nShrinkage magnitude summary:\n")
cat(sprintf("  Model A (sigma=10):  mean |shrinkage| = %.3f, max = %.3f\n",
            mean(abs(compare_df$shrinkage_A)), max(abs(compare_df$shrinkage_A))))
cat(sprintf("  Model B (sigma=2):   mean |shrinkage| = %.3f, max = %.3f\n",
            mean(abs(compare_df$shrinkage_B)), max(abs(compare_df$shrinkage_B))))

write.csv(compare_df,
          file.path(PROJECT_ROOT, "output", "models", "prior_impact_comparison.csv"),
          row.names = FALSE)

# --- 11. Write comprehensive report to text file ---------------------------
sink(REPORT_PATH)
cat("MCMC DIAGNOSTICS REPORT\n")
cat(rep("=", 70), sep = ""); cat("\n\n")

cat("MCMC settings:\n")
cat("  Iterations/chain:", model_A$mcmc_settings$n_iter, "\n")
cat("  Burn-in:         ", model_A$mcmc_settings$n_burnin, "\n")
cat("  Thinning:        ", model_A$mcmc_settings$n_thin, "\n")
cat("  Chains:          ", model_A$mcmc_settings$n_chains, "\n")
cat("  Samples/chain after thinning:", N_A, "\n\n")

for (name_mdl in c("Model A (Normal(0, 10))", "Model B (Normal(0, 2))")) {
  cat(rep("-", 70), sep = ""); cat("\n")
  cat(name_mdl, "\n")
  cat(rep("-", 70), sep = ""); cat("\n")
  
  d <- if (grepl("A ", name_mdl)) diag_A else diag_B
  p <- if (grepl("A ", name_mdl)) post_A else post_B
  ch_rates <- if (grepl("A ", name_mdl)) {
    sapply(model_A$chains, function(c) c$accept_rate)
  } else {
    sapply(model_B$chains, function(c) c$accept_rate)
  }
  
  cat("\nAcceptance rates: ", paste(round(ch_rates, 3), collapse = ", "), "\n")
  cat("Max R-hat:        ", round(max(d$r_hat), 4), "\n")
  cat("Min ESS:          ", min(d$ess), "\n")
  cat("Median ESS:       ", median(d$ess), "\n")
  cat("Params w/ R-hat > 1.1:", sum(d$r_hat > 1.1), "/", nrow(d), "\n")
  cat("Params w/ ESS < 400:  ", sum(d$ess < 400),   "/", nrow(d), "\n\n")
  
  cat("Posterior summary (all parameters):\n")
  full <- merge(p, d, by = "parameter", sort = FALSE)
  print(full, row.names = FALSE)
  cat("\n\n")
}

cat(rep("-", 70), sep = ""); cat("\n")
cat("PRIOR IMPACT: MLE vs Model A vs Model B\n")
cat(rep("-", 70), sep = ""); cat("\n")
print(compare_df, row.names = FALSE)
sink()

cat("\n=== PHASE 4d COMPLETE ===\n")
cat("Diagnostics report: ", REPORT_PATH, "\n")
cat("Trace plots:        ", FIGURES_DIR, "/trace_plots_{A,B}.png\n")
cat("Posterior summaries:", PROJECT_ROOT, "/output/models/posterior_summary_{A,B}.csv\n")
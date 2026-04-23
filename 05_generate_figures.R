# ============================================================================
# 05_generate_figures.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
# ----------------------------------------------------------------------------
# Purpose: Generate all presentation-ready figures from saved model results.
#          Every figure is saved as a high-resolution PNG in output/figures/.
#
# Figures produced:
#   1. Class distribution (train vs test) bar chart
#   2. Sample MRI images — one per class
#   3. Preprocessing pipeline illustration (1 image through 4 stages)
#   4. Feature correlation heatmap
#   5. MCMC trace plots for top 6 coefficients (clean subset of the full 33)
#   6. Posterior density plots for top 6 coefficients
#   7. Coefficient forest plot — MLE vs Model A vs Model B with 95% CIs
#   8. Confusion matrices — side-by-side for all 3 models
#   9. Model comparison bar chart (accuracy, macro F1)
#  10. Per-class F1 grouped bar chart
#  11. Predictive entropy distribution per class
#  12. R-hat / ESS summary table as figure
# ============================================================================

rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(ggplot2)
library(dplyr)
library(EBImage)
library(coda)
library(nnet)


FIGURES_DIR <- file.path(PROJECT_ROOT, "output", "figures")
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# --- Load everything --------------------------------------------------------
manifest   <- read.csv(file.path(PROJECT_ROOT, "output", "image_manifest.csv"),
                       stringsAsFactors = FALSE)
features   <- read.csv(file.path(PROJECT_ROOT, "output", "features",
                                 "radiomic_features.csv"), stringsAsFactors = FALSE)
prepared   <- readRDS(file.path(PROJECT_ROOT, "output", "features", "prepared_data.rds"))
model_A    <- readRDS(file.path(PROJECT_ROOT, "output", "models", "bayesian_model_A.rds"))
model_B    <- readRDS(file.path(PROJECT_ROOT, "output", "models", "bayesian_model_B.rds"))
eval_obj   <- readRDS(file.path(PROJECT_ROOT, "output", "models", "evaluation_results.rds"))
post_A     <- read.csv(file.path(PROJECT_ROOT, "output", "models", "posterior_summary_A.csv"),
                       stringsAsFactors = FALSE)
post_B     <- read.csv(file.path(PROJECT_ROOT, "output", "models", "posterior_summary_B.csv"),
                       stringsAsFactors = FALSE)

# Friendly class names for plots
class_pretty <- c(
  no_tumor         = "No Tumor",
  glioma_tumor     = "Glioma",
  meningioma_tumor = "Meningioma",
  pituitary_tumor  = "Pituitary"
)
class_colors <- c(
  "No Tumor"   = "#6C757D",
  "Glioma"     = "#E63946",
  "Meningioma" = "#457B9D",
  "Pituitary"  = "#2A9D8F"
)

# Enforce consistent class ordering across every figure
CLASS_ORDER_PRETTY <- c("No Tumor", "Glioma", "Meningioma", "Pituitary")
CLASS_ORDER_RAW    <- c("no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor")

# Shared ggplot theme — clean academic style
academic_theme <- theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11),
    legend.position = "top",
    legend.title = element_blank()
  )

# Helper: save a ggplot
save_plot <- function(p, name, w = 8, h = 5, dpi = 200) {
  ggsave(file.path(FIGURES_DIR, name), p,
         width = w, height = h, dpi = dpi, bg = "white")
  cat("  Saved:", name, "\n")
}

# ============================================================================
# Figure 1: Class distribution (train vs test)
# ============================================================================
cat("\nFigure 1: Class distribution\n")
dist_df <- as.data.frame(table(manifest$split, manifest$class))
names(dist_df) <- c("Split", "Class", "Count")
dist_df$Class <- class_pretty[as.character(dist_df$Class)]
dist_df$Split <- ifelse(dist_df$Split == "train", "Training", "Test")
dist_df$Class <- factor(dist_df$Class, levels = CLASS_ORDER_PRETTY)

p1 <- ggplot(dist_df, aes(x = Class, y = Count, fill = Class)) +
  geom_col() +
  geom_text(aes(label = Count), vjust = -0.3, size = 3.5) +
  facet_wrap(~Split, scales = "free_y") +
  scale_fill_manual(values = class_colors) +
  labs(title = "Dataset class distribution",
       subtitle = "Bhuvaji Brain Tumor Classification (MRI), 3,264 images total",
       x = NULL, y = "Number of images") +
  academic_theme +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, hjust = 1))
save_plot(p1, "fig01_class_distribution.png", w = 9, h = 4.5)

# ============================================================================
# Figure 2: Sample MRI images — one per class (side by side grid)
# ============================================================================
cat("Figure 2: Sample MRI images\n")
set.seed(7)
sample_per_class <- do.call(rbind, lapply(
  CLASS_ORDER_RAW,
  function(cl) manifest[manifest$class == cl & manifest$split == "train", ][
    sample(sum(manifest$class == cl & manifest$split == "train"), 1), ]
))

png(file.path(FIGURES_DIR, "fig02_sample_images.png"),
    width = 1800, height = 500, res = 200, bg = "white")
par(mfrow = c(1, 4), mar = c(2, 1, 3, 1))
for (i in seq_len(nrow(sample_per_class))) {
  row <- sample_per_class[i, ]
  img <- readImage(row$file_path)
  if (length(dim(img)) == 3) img <- (img[,,1] + img[,,2] + img[,,3]) / 3
  image(t(img)[, nrow(img):1], col = gray.colors(256),
        main = class_pretty[row$class], axes = FALSE, cex.main = 1.4)
}
dev.off()
cat("  Saved: fig02_sample_images.png\n")

# ============================================================================
# Figure 3: Preprocessing pipeline (raw -> grayscale -> resize -> normalized)
# ============================================================================
cat("Figure 3: Preprocessing pipeline\n")
sample_row <- sample_per_class[1, ]  # use glioma example
raw_img  <- readImage(sample_row$file_path)
gray_img <- if (length(dim(raw_img)) == 3)
  (raw_img[,,1] + raw_img[,,2] + raw_img[,,3]) / 3 else raw_img
resized  <- resize(Image(gray_img, colormode = Grayscale), w = 128, h = 128)
resized_mat <- imageData(resized)
thresh   <- otsu(Image(resized_mat, colormode = Grayscale))
mask     <- resized_mat > thresh
brain_vals <- resized_mat[mask]
mu <- mean(brain_vals); sigma <- sd(brain_vals)
normalized <- (resized_mat - mu) / sigma
normalized[!mask] <- 0

png(file.path(FIGURES_DIR, "fig03_preprocessing_pipeline.png"),
    width = 2000, height = 520, res = 200, bg = "white")
par(mfrow = c(1, 4), mar = c(1, 1, 3, 1))
image(t(gray_img)[, nrow(gray_img):1],    col = gray.colors(256), main = "1. Raw grayscale",  axes = FALSE, cex.main = 1.3)
image(t(resized_mat)[, nrow(resized_mat):1], col = gray.colors(256), main = "2. Resized 128x128", axes = FALSE, cex.main = 1.3)
image(t(mask)[, nrow(mask):1],            col = c("black", "white"), main = "3. Otsu mask",    axes = FALSE, cex.main = 1.3)
image(t(normalized)[, nrow(normalized):1], col = gray.colors(256), main = "4. Z-score normalized", axes = FALSE, cex.main = 1.3)
dev.off()
cat("  Saved: fig03_preprocessing_pipeline.png\n")

# ============================================================================
# Figure 4: Feature correlation heatmap
# ============================================================================
cat("Figure 4: Feature correlation heatmap\n")
feat_cols <- model_A$feature_names
cor_mat <- cor(features[features$split == "train", feat_cols])
cor_long <- expand.grid(Feature1 = feat_cols, Feature2 = feat_cols, stringsAsFactors = FALSE)
cor_long$Correlation <- as.vector(cor_mat)

p4 <- ggplot(cor_long, aes(x = Feature1, y = Feature2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", Correlation)), size = 2.8) +
  scale_fill_gradient2(low = "#457B9D", mid = "white", high = "#E63946",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Feature correlation structure (training set)",
       subtitle = "After dropping 'dissimilarity' (r = 0.94 with contrast)",
       x = NULL, y = NULL) +
  academic_theme +
  theme(axis.text.x = element_text(angle = 40, hjust = 1))
save_plot(p4, "fig04_correlation_heatmap.png", w = 9, h = 7)

# ============================================================================
# Figure 5: MCMC trace plots for top 6 coefficients (curated, not all 33)
# ============================================================================
cat("Figure 5: Trace plots (top 6)\n")
# Pick the 6 parameters with largest |posterior mean| in Model A
top6_idx <- order(-abs(post_A$post_mean))[1:6]
top6_names <- post_A$parameter[top6_idx]

# Extract samples for these
to_mcmc_list <- function(model_obj, param_names_full) {
  chains <- lapply(model_obj$chains, function(ch) {
    m <- as.matrix(ch$samples); colnames(m) <- param_names_full
    mcmc(m)
  })
  mcmc.list(chains)
}
mcmc_A <- to_mcmc_list(model_A, post_A$parameter)

png(file.path(FIGURES_DIR, "fig05_trace_plots_top6.png"),
    width = 1600, height = 1200, res = 180, bg = "white")
par(mfrow = c(3, 2), mar = c(3.5, 3.5, 3, 1), mgp = c(2, 0.6, 0))
chain_cols <- c("#1d3557", "#E63946", "#457B9D", "#2A9D8F")
for (p in top6_idx) {
  ylim_p <- range(sapply(mcmc_A, function(m) range(m[, p])))
  plot(mcmc_A[[1]][, p], type = "l", col = chain_cols[1],
       ylim = ylim_p, main = top6_names[which(top6_idx == p)],
       xlab = "Iteration (post burn-in, thinned)", ylab = "Coefficient value",
       lwd = 0.6)
  for (c in 2:length(mcmc_A)) lines(mcmc_A[[c]][, p], col = chain_cols[c], lwd = 0.6)
}
dev.off()
cat("  Saved: fig05_trace_plots_top6.png\n")

# ============================================================================
# Figure 6: Posterior density plots for top 6 coefficients (Model A vs B)
# ============================================================================
cat("Figure 6: Posterior densities (top 6)\n")
mcmc_B <- to_mcmc_list(model_B, post_B$parameter)
combined_A <- do.call(rbind, mcmc_A)
combined_B <- do.call(rbind, mcmc_B)

dens_df <- do.call(rbind, lapply(top6_idx, function(p) {
  rbind(
    data.frame(param = top6_names[which(top6_idx == p)],
               model = "Model A (N(0,10))", value = combined_A[, p]),
    data.frame(param = top6_names[which(top6_idx == p)],
               model = "Model B (N(0,2))",  value = combined_B[, p])
  )
}))

p6 <- ggplot(dens_df, aes(x = value, fill = model, color = model)) +
  geom_density(alpha = 0.35, linewidth = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  facet_wrap(~param, scales = "free", ncol = 2) +
  scale_fill_manual(values  = c("Model A (N(0,10))" = "#1d3557",
                                "Model B (N(0,2))"  = "#E63946")) +
  scale_color_manual(values = c("Model A (N(0,10))" = "#1d3557",
                                "Model B (N(0,2))"  = "#E63946")) +
  labs(title = "Posterior densities for top coefficients",
       subtitle = "Comparing weakly informative vs. regularizing prior",
       x = "Coefficient value (log-odds)", y = "Posterior density") +
  academic_theme
save_plot(p6, "fig06_posterior_densities.png", w = 10, h = 7)

# ============================================================================
# Figure 7: Coefficient forest plot — MLE vs Model A vs Model B, top 10
# ============================================================================
cat("Figure 7: Forest plot (top 10 coefficients)\n")
# coef() may return matrix (with nnet loaded) or named vector; handle both.
freq_coef_raw <- coef(model_A$freq_fit_v2)
if (is.matrix(freq_coef_raw)) {
  freq_coef_vec <- as.vector(t(freq_coef_raw))
} else {
  # Vector returned; it's already in the right order (class-major, feature-minor)
  freq_coef_vec <- as.vector(freq_coef_raw)
}

# Take top 10 parameters by |Model A posterior mean|, excluding intercepts
slope_idx <- !grepl("Intercept", post_A$parameter)
top10_idx <- which(slope_idx)[order(-abs(post_A$post_mean[slope_idx]))[1:10]]

forest_df <- data.frame(
  parameter = rep(post_A$parameter[top10_idx], 3),
  model     = rep(c("Frequentist MLE", "Bayesian A (N(0,10))", "Bayesian B (N(0,2))"),
                  each = 10),
  estimate  = c(freq_coef_vec[top10_idx],
                post_A$post_mean[top10_idx],
                post_B$post_mean[top10_idx]),
  lower     = c(freq_coef_vec[top10_idx],     # MLE: no CI shown
                post_A$ci_lower[top10_idx],
                post_B$ci_lower[top10_idx]),
  upper     = c(freq_coef_vec[top10_idx],
                post_A$ci_upper[top10_idx],
                post_B$ci_upper[top10_idx])
)
forest_df$parameter <- factor(forest_df$parameter,
                              levels = rev(post_A$parameter[top10_idx]))
forest_df$model <- factor(forest_df$model,
                          levels = c("Frequentist MLE", "Bayesian A (N(0,10))", "Bayesian B (N(0,2))"))

p7 <- ggplot(forest_df, aes(x = estimate, y = parameter, color = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.3,
                 position = position_dodge(width = 0.55), linewidth = 0.5) +
  geom_point(position = position_dodge(width = 0.55), size = 2.5) +
  scale_color_manual(values = c("Frequentist MLE"       = "gray30",
                                "Bayesian A (N(0,10))"  = "#1d3557",
                                "Bayesian B (N(0,2))"   = "#E63946")) +
  labs(title = "Top 10 coefficients: MLE vs Bayesian posteriors",
       subtitle = "Horizontal bars = 95% posterior credible intervals",
       x = "Coefficient (log-odds)", y = NULL) +
  academic_theme +
  theme(panel.grid.major.y = element_line(color = "gray92"))
save_plot(p7, "fig07_coefficient_forest.png", w = 10, h = 6.5)

# ============================================================================
# Figure 8: Confusion matrices side-by-side
# ============================================================================
cat("Figure 8: Confusion matrices\n")
cm_to_df <- function(cm_mat, model_name) {
  df <- as.data.frame(as.table(cm_mat))
  names(df) <- c("Actual", "Predicted", "Count")
  df$Actual    <- class_pretty[as.character(df$Actual)]
  df$Predicted <- class_pretty[as.character(df$Predicted)]
  df$Model <- model_name
  df
}
cm_df <- rbind(
  cm_to_df(eval_obj$frequentist$metrics$cm, "Frequentist MLE"),
  cm_to_df(eval_obj$model_A$metrics$cm,     "Bayesian A (N(0,10))"),
  cm_to_df(eval_obj$model_B$metrics$cm,     "Bayesian B (N(0,2))")
)
cm_df$Model <- factor(cm_df$Model,
                      levels = c("Frequentist MLE", "Bayesian A (N(0,10))", "Bayesian B (N(0,2))"))

p8 <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = Count), size = 3.8, fontface = "bold") +
  facet_wrap(~Model) +
  scale_fill_gradient(low = "white", high = "#1d3557") +
  labs(title = "Confusion matrices on held-out test set (N = 394)",
       subtitle = "Rows = actual class, Columns = predicted class",
       x = "Predicted", y = "Actual") +
  academic_theme +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none")
save_plot(p8, "fig08_confusion_matrices.png", w = 13, h = 5)

# ============================================================================
# Figure 9: Model comparison — accuracy and macro F1
# ============================================================================
cat("Figure 9: Accuracy / Macro F1 comparison\n")
comp_df <- data.frame(
  Model = rep(c("Frequentist\nMLE", "Bayesian A\nN(0,10)", "Bayesian B\nN(0,2)"), 2),
  Metric = rep(c("Accuracy", "Macro F1"), each = 3),
  Value  = c(eval_obj$frequentist$metrics$accuracy,
             eval_obj$model_A$metrics$accuracy,
             eval_obj$model_B$metrics$accuracy,
             eval_obj$frequentist$metrics$macro_f1,
             eval_obj$model_A$metrics$macro_f1,
             eval_obj$model_B$metrics$macro_f1)
)

p9 <- ggplot(comp_df, aes(x = Model, y = Value, fill = Model)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", Value)), vjust = -0.4, size = 3.5) +
  facet_wrap(~Metric, scales = "free_y") +
  scale_fill_manual(values = c("Frequentist\nMLE"   = "gray50",
                               "Bayesian A\nN(0,10)" = "#1d3557",
                               "Bayesian B\nN(0,2)"  = "#E63946")) +
  labs(title = "Overall model performance on test set",
       x = NULL, y = NULL) +
  academic_theme +
  theme(legend.position = "none")
save_plot(p9, "fig09_accuracy_f1_comparison.png", w = 9, h = 4.5)

# ============================================================================
# Figure 10: Per-class F1 grouped bar chart
# ============================================================================
cat("Figure 10: Per-class F1\n")
per_class_df <- rbind(
  data.frame(model = "Frequentist MLE",
             class = eval_obj$frequentist$metrics$per_class$class,
             f1    = eval_obj$frequentist$metrics$per_class$f1),
  data.frame(model = "Bayesian A (N(0,10))",
             class = eval_obj$model_A$metrics$per_class$class,
             f1    = eval_obj$model_A$metrics$per_class$f1),
  data.frame(model = "Bayesian B (N(0,2))",
             class = eval_obj$model_B$metrics$per_class$class,
             f1    = eval_obj$model_B$metrics$per_class$f1)
)
per_class_df$class <- class_pretty[as.character(per_class_df$class)]
per_class_df$class <- factor(per_class_df$class,
                             levels = c("No Tumor", "Glioma", "Meningioma", "Pituitary"))
per_class_df$model <- factor(per_class_df$model,
                             levels = c("Frequentist MLE", "Bayesian A (N(0,10))", "Bayesian B (N(0,2))"))

p10 <- ggplot(per_class_df, aes(x = class, y = f1, fill = model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", f1)),
            position = position_dodge(width = 0.75), vjust = -0.3, size = 3) +
  scale_fill_manual(values = c("Frequentist MLE"       = "gray50",
                               "Bayesian A (N(0,10))"  = "#1d3557",
                               "Bayesian B (N(0,2))"   = "#E63946")) +
  labs(title = "F1 score by tumor class",
       subtitle = "Bayesian B shows modest improvements on the harder classes",
       x = NULL, y = "F1 score") +
  academic_theme +
  ylim(0, 0.7)
save_plot(p10, "fig10_per_class_f1.png", w = 10, h = 5)

# ============================================================================
# Figure 11: Predictive entropy distribution per class (Bayesian uncertainty)
# ============================================================================
cat("Figure 11: Predictive entropy distribution\n")
entropy_df <- data.frame(
  actual  = eval_obj$test_y,
  entropy = eval_obj$model_B$entropy,
  correct = eval_obj$test_y == eval_obj$model_B$pred_class
)
entropy_df$actual <- class_pretty[as.character(entropy_df$actual)]
entropy_df$actual <- factor(entropy_df$actual,
                            levels = c("No Tumor", "Glioma", "Meningioma", "Pituitary"))

p11 <- ggplot(entropy_df, aes(x = actual, y = entropy, fill = actual)) +
  geom_boxplot(width = 0.5, alpha = 0.75, outlier.size = 0.8) +
  scale_fill_manual(values = class_colors) +
  labs(title = "Predictive uncertainty per class (Bayesian Model B)",
       subtitle = "Entropy of mean posterior predictive distribution (0 = certain, 2 = max confusion)",
       x = NULL, y = "Entropy (bits)") +
  academic_theme +
  theme(legend.position = "none")
save_plot(p11, "fig11_predictive_entropy.png", w = 9, h = 5)

# ============================================================================
# Figure 12: Convergence diagnostics summary (R-hat and ESS)
# ============================================================================
cat("Figure 12: Convergence diagnostics\n")
diag_df <- rbind(
  data.frame(model = "Model A (N(0,10))",
             r_hat = post_A$r_hat, ess = post_A$ess),
  data.frame(model = "Model B (N(0,2))",
             r_hat = post_B$r_hat, ess = post_B$ess)
)

p12a <- ggplot(diag_df, aes(x = r_hat, fill = model)) +
  geom_histogram(bins = 12, alpha = 0.75, color = "white", position = "identity") +
  geom_vline(xintercept = 1.01, linetype = "dashed", color = "darkgreen", linewidth = 0.5) +
  geom_vline(xintercept = 1.10, linetype = "dashed", color = "red", linewidth = 0.5) +
  annotate("text", x = 1.008, y = Inf, label = "R-hat = 1.01\ngold standard", vjust = 2, size = 3) +
  scale_fill_manual(values = c("Model A (N(0,10))" = "#1d3557",
                               "Model B (N(0,2))"  = "#E63946")) +
  labs(title = "Gelman-Rubin R-hat distribution (33 parameters)",
       subtitle = "All parameters well below the 1.01 gold-standard threshold",
       x = "R-hat", y = "Number of parameters") +
  academic_theme
save_plot(p12a, "fig12a_rhat_distribution.png", w = 9, h = 4.5)

p12b <- ggplot(diag_df, aes(x = ess, fill = model)) +
  geom_histogram(bins = 12, alpha = 0.75, color = "white", position = "identity") +
  geom_vline(xintercept = 400, linetype = "dashed", color = "red", linewidth = 0.5) +
  annotate("text", x = 450, y = Inf, label = "ESS = 400\nminimum", vjust = 2, size = 3) +
  scale_fill_manual(values = c("Model A (N(0,10))" = "#1d3557",
                               "Model B (N(0,2))"  = "#E63946")) +
  labs(title = "Effective sample size (ESS) distribution",
       subtitle = "All parameters far exceed the minimum threshold",
       x = "Effective sample size", y = "Number of parameters") +
  academic_theme
save_plot(p12b, "fig12b_ess_distribution.png", w = 9, h = 4.5)

cat("\n=== ALL FIGURES GENERATED ===\n")
cat("Output directory:", FIGURES_DIR, "\n")
cat("Files produced:\n")
print(list.files(FIGURES_DIR, pattern = "^fig.*\\.png$"))
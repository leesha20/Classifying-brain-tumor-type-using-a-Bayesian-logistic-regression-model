# ============================================================================
# 04a_feature_prep.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
# ----------------------------------------------------------------------------
# Purpose: Clean and prepare the radiomic feature matrix for modeling:
#          drop uninformative features, drop redundant correlated features,
#          standardize, and produce train/test matrices.
#
# Input:   output/features/radiomic_features.csv
# Output:  output/features/prepared_data.rds  (list with train/test X, y, plus metadata)
# ============================================================================

# --- 1. Setup ----------------------------------------------------------------
rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(dplyr)

# --- 2. Config ---------------------------------------------------------------
FEATURES_PATH  <- file.path(PROJECT_ROOT, "output", "features", "radiomic_features.csv")
PREPARED_PATH  <- file.path(PROJECT_ROOT, "output", "features", "prepared_data.rds")

# Correlation threshold for dropping redundant features
CORRELATION_THRESHOLD <- 0.95

# --- 3. Load features --------------------------------------------------------
features <- read.csv(FEATURES_PATH, stringsAsFactors = FALSE)
cat("Loaded features:", nrow(features), "rows x", ncol(features), "cols\n")

# Remove any rows that failed feature extraction (there should be none)
features <- features[is.na(features$extraction_error), ]
cat("After removing extraction errors:", nrow(features), "rows\n\n")

# --- 4. Drop uninformative features ------------------------------------------
# mean_intensity and variance are constant across all images (z-score artifact)
all_feature_cols <- c("mean_intensity", "variance", "skewness", "kurtosis",
                      "energy", "entropy", "contrast", "correlation",
                      "glcm_energy", "homogeneity", "dissimilarity",
                      "tumor_area", "aspect_ratio", "perimeter")

# Check which columns have zero variance (constant)
feature_variances <- sapply(features[, all_feature_cols], var, na.rm = TRUE)
constant_features <- names(feature_variances)[feature_variances < 1e-10]

cat("Constant features to drop:", paste(constant_features, collapse = ", "), "\n")

informative_features <- setdiff(all_feature_cols, constant_features)
cat("Informative features remaining:", length(informative_features), "\n\n")

# --- 5. Build train/test splits ---------------------------------------------
train_df <- features[features$split == "train", ]
test_df  <- features[features$split == "test", ]

cat("Train rows:", nrow(train_df), "\n")
cat("Test rows: ", nrow(test_df), "\n\n")

# --- 6. Drop highly correlated features (based on TRAINING set only) --------
train_feat_matrix <- as.matrix(train_df[, informative_features])
cor_mat <- cor(train_feat_matrix)

cat("Feature correlation matrix (training set):\n")
print(round(cor_mat, 2))

# Find pairs with |correlation| > threshold, drop the second of each pair
to_drop <- character(0)
for (i in 1:(ncol(cor_mat) - 1)) {
  for (j in (i + 1):ncol(cor_mat)) {
    if (abs(cor_mat[i, j]) > CORRELATION_THRESHOLD) {
      # Drop the one with lower variance (less discriminative) — usually
      # both are fine, but this is a deterministic rule
      feat_j <- colnames(cor_mat)[j]
      if (!(feat_j %in% to_drop)) {
        to_drop <- c(to_drop, feat_j)
      }
    }
  }
}

cat("\nHighly correlated features to drop (|r| >", CORRELATION_THRESHOLD, "):",
    if (length(to_drop) == 0) "none" else paste(to_drop, collapse = ", "), "\n")

final_features <- setdiff(informative_features, to_drop)
cat("Final feature set:", length(final_features), "features\n")
cat("  ", paste(final_features, collapse = ", "), "\n\n")

# --- 7. Standardize (z-score) using TRAINING statistics ---------------------
train_X_raw <- as.matrix(train_df[, final_features])
test_X_raw  <- as.matrix(test_df[,  final_features])

# Compute mean and sd from training data only
feature_means <- colMeans(train_X_raw)
feature_sds   <- apply(train_X_raw, 2, sd)

# Apply standardization
train_X <- sweep(train_X_raw, 2, feature_means, "-")
train_X <- sweep(train_X,     2, feature_sds,   "/")

test_X <- sweep(test_X_raw, 2, feature_means, "-")
test_X <- sweep(test_X,     2, feature_sds,   "/")

# Sanity check: training features should now have mean ~0, sd ~1
cat("Post-standardization check (training set):\n")
cat("  Column means (should be ~0):", round(colMeans(train_X), 3), "\n")
cat("  Column SDs   (should be ~1):", round(apply(train_X, 2, sd), 3), "\n\n")

# --- 8. Encode class labels as factors (consistent levels across train/test) -
class_levels <- c("no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor")
# Reference class = no_tumor. Other classes get log-odds relative to this.

train_y <- factor(train_df$class, levels = class_levels)
test_y  <- factor(test_df$class,  levels = class_levels)

cat("Class distribution (train):\n")
print(table(train_y))
cat("\nClass distribution (test):\n")
print(table(test_y))

# --- 9. Assemble and save ---------------------------------------------------
prepared <- list(
  train_X        = train_X,
  train_y        = train_y,
  test_X         = test_X,
  test_y         = test_y,
  feature_names  = final_features,
  feature_means  = feature_means,   # needed if we ever want to predict on new data
  feature_sds    = feature_sds,
  class_levels   = class_levels,
  dropped_constant = constant_features,
  dropped_correlated = to_drop,
  correlation_matrix = cor_mat
)

saveRDS(prepared, PREPARED_PATH)

cat("\n=== PHASE 4a COMPLETE ===\n")
cat("Prepared data saved to:", PREPARED_PATH, "\n")
cat("Ready for Phase 4b (frequentist baseline) and 4c (Bayesian MH sampler).\n")
# ============================================================================
# 03_features.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
#                           for Brain Tumor MRI Classification
# ----------------------------------------------------------------------------
# Purpose: Extract radiomic features from preprocessed brain images. Each
#          image produces one feature vector. All features are computed on
#          the brain region only (background is masked out).
#
# Features extracted (14 total):
#
#   First-order intensity statistics (6):
#     - mean, variance, skewness, kurtosis, energy, entropy
#
#   GLCM texture features (5):  [32 gray levels, 4 directions averaged]
#     - contrast, correlation, glcm_energy, homogeneity, dissimilarity
#
#   Shape features (3):
#     - tumor_area (brain mask pixel count)
#     - aspect_ratio (bounding box width / height)
#     - perimeter (estimated boundary pixel count)
#
# Input:   data/processed/<image_id>.rds
#          output/image_manifest.csv
# Output:  output/features/radiomic_features.csv
# ============================================================================

# --- 1. Setup ----------------------------------------------------------------
rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(EBImage)
library(dplyr)

# --- 2. Config ---------------------------------------------------------------
N_GRAY_LEVELS   <- 32     # GLCM quantization: 32 gray levels
GLCM_DIRECTIONS <- list(  # 4 directions, each as (dy, dx) offset
  "0deg"   = c( 0,  1),   # horizontal
  "45deg"  = c(-1,  1),   # diagonal up-right
  "90deg"  = c(-1,  0),   # vertical
  "135deg" = c(-1, -1)    # diagonal up-left
)

PROCESSED_DIR  <- file.path(PROJECT_ROOT, "data", "processed")
MANIFEST_PATH  <- file.path(PROJECT_ROOT, "output", "image_manifest.csv")
FEATURES_DIR   <- file.path(PROJECT_ROOT, "output", "features")
FEATURES_PATH  <- file.path(FEATURES_DIR, "radiomic_features.csv")

dir.create(FEATURES_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 3. Load manifest --------------------------------------------------------
manifest <- read.csv(MANIFEST_PATH, stringsAsFactors = FALSE)
cat("Loaded manifest:", nrow(manifest), "images\n\n")

# ============================================================================
# --- 4. FEATURE COMPUTATION FUNCTIONS ---------------------------------------
# ============================================================================

# ----------------------------------------------------------------------------
# 4a. First-order statistics
#     Operates on a vector of pixel intensities (brain pixels only)
# ----------------------------------------------------------------------------
first_order_features <- function(pixels) {
  n <- length(pixels)
  if (n < 2) {
    return(list(
      mean_intensity = NA, variance = NA, skewness = NA,
      kurtosis = NA, energy = NA, entropy = NA
    ))
  }
  
  mu    <- mean(pixels)
  sigma <- sd(pixels)
  
  # Skewness and kurtosis — manual computation (avoid extra package dependency)
  centered <- pixels - mu
  if (sigma > 0) {
    skew <- mean(centered^3) / sigma^3
    kurt <- mean(centered^4) / sigma^4 - 3  # excess kurtosis
  } else {
    skew <- 0
    kurt <- 0
  }
  
  # Energy = sum of squared intensities
  energy <- sum(pixels^2)
  
  # Entropy — compute from a histogram of pixel values
  h <- hist(pixels, breaks = 64, plot = FALSE)
  p <- h$counts / sum(h$counts)
  p <- p[p > 0]
  entropy <- -sum(p * log2(p))
  
  list(
    mean_intensity = mu,
    variance       = sigma^2,
    skewness       = skew,
    kurtosis       = kurt,
    energy         = energy,
    entropy        = entropy
  )
}

# ----------------------------------------------------------------------------
# 4b. Quantize image to N gray levels
#     Input: a matrix of (z-scored) pixel intensities + a brain mask
#     Output: integer matrix with values in 1..N_GRAY_LEVELS, or 0 for background
# ----------------------------------------------------------------------------
quantize_image <- function(img_mat, mask, n_levels = N_GRAY_LEVELS) {
  q <- matrix(0L, nrow = nrow(img_mat), ncol = ncol(img_mat))
  brain_vals <- img_mat[mask]
  if (length(brain_vals) == 0) return(q)
  
  # Min-max scale the brain pixels to [1, n_levels]
  mn <- min(brain_vals)
  mx <- max(brain_vals)
  if (mx - mn < 1e-8) {
    # All pixels identical — put them all in level 1
    q[mask] <- 1L
  } else {
    scaled <- (brain_vals - mn) / (mx - mn)
    # Map to 1..n_levels (ceiling of scaled * n_levels, clamped)
    levels <- pmin(pmax(ceiling(scaled * n_levels), 1L), n_levels)
    q[mask] <- as.integer(levels)
  }
  q
}

# ----------------------------------------------------------------------------
# 4c. Compute GLCM for one direction
#     Builds an N x N co-occurrence matrix counting pixel-pair frequencies.
#     Only counts pairs where BOTH pixels are in the brain (nonzero quantized).
# ----------------------------------------------------------------------------
compute_glcm <- function(q_img, offset, n_levels = N_GRAY_LEVELS) {
  dy <- offset[1]
  dx <- offset[2]
  nr <- nrow(q_img)
  nc <- ncol(q_img)
  
  glcm <- matrix(0, nrow = n_levels, ncol = n_levels)
  
  # Define the valid range of (i, j) such that (i+dy, j+dx) is also in bounds
  row_start <- max(1, 1 - dy)
  row_end   <- min(nr, nr - dy)
  col_start <- max(1, 1 - dx)
  col_end   <- min(nc, nc - dx)
  
  if (row_start > row_end || col_start > col_end) return(glcm)
  
  # Pull out the two pixel arrays (reference and neighbor)
  ref_vals  <- q_img[row_start:row_end,       col_start:col_end]
  neigh_vals<- q_img[(row_start + dy):(row_end + dy),
                     (col_start + dx):(col_end + dx)]
  
  # Keep only pairs where both pixels are brain tissue (level > 0)
  both_valid <- ref_vals > 0 & neigh_vals > 0
  if (!any(both_valid)) return(glcm)
  
  r <- ref_vals[both_valid]
  n <- neigh_vals[both_valid]
  
  # Fast bivariate tabulation: each (r, n) pair increments glcm[r, n]
  # Use table() with fixed levels to ensure n_levels x n_levels output
  tab <- table(factor(r, levels = 1:n_levels),
               factor(n, levels = 1:n_levels))
  glcm <- as.matrix(tab) + t(as.matrix(tab))  # symmetric GLCM
  
  # Normalize to probabilities
  total <- sum(glcm)
  if (total > 0) glcm <- glcm / total
  glcm
}

# ----------------------------------------------------------------------------
# 4d. Extract texture features from a GLCM matrix
#     Follows standard Haralick (1973) definitions
# ----------------------------------------------------------------------------
glcm_features <- function(glcm) {
  n <- nrow(glcm)
  if (sum(glcm) < 1e-12) {
    return(list(contrast = NA, correlation = NA, glcm_energy = NA,
                homogeneity = NA, dissimilarity = NA))
  }
  
  # Coordinate grids
  i_grid <- matrix(rep(1:n, n), nrow = n)
  j_grid <- t(i_grid)
  
  # Marginal distributions
  p_i <- rowSums(glcm)
  p_j <- colSums(glcm)
  
  mu_i    <- sum((1:n) * p_i)
  mu_j    <- sum((1:n) * p_j)
  sigma_i <- sqrt(sum(((1:n) - mu_i)^2 * p_i))
  sigma_j <- sqrt(sum(((1:n) - mu_j)^2 * p_j))
  
  # Contrast:     sum of (i - j)^2 * P(i, j)
  contrast <- sum((i_grid - j_grid)^2 * glcm)
  
  # Correlation:  normalized covariance of (i, j) under GLCM
  if (sigma_i > 0 && sigma_j > 0) {
    correlation <- sum((i_grid - mu_i) * (j_grid - mu_j) * glcm) / (sigma_i * sigma_j)
  } else {
    correlation <- 0
  }
  
  # GLCM energy (Angular Second Moment): sum of P(i, j)^2
  glcm_energy <- sum(glcm^2)
  
  # Homogeneity (Inverse Difference Moment):  sum P(i,j) / (1 + (i-j)^2)
  homogeneity <- sum(glcm / (1 + (i_grid - j_grid)^2))
  
  # Dissimilarity: sum |i - j| * P(i, j)
  dissimilarity <- sum(abs(i_grid - j_grid) * glcm)
  
  list(
    contrast      = contrast,
    correlation   = correlation,
    glcm_energy   = glcm_energy,
    homogeneity   = homogeneity,
    dissimilarity = dissimilarity
  )
}

# ----------------------------------------------------------------------------
# 4e. Shape features from brain mask
# ----------------------------------------------------------------------------
shape_features <- function(mask) {
  n_pixels <- sum(mask)
  if (n_pixels < 10) {
    return(list(tumor_area = n_pixels, aspect_ratio = NA, perimeter = NA))
  }
  
  # Bounding box
  rows_with_px <- which(rowSums(mask) > 0)
  cols_with_px <- which(colSums(mask) > 0)
  bb_height <- diff(range(rows_with_px)) + 1
  bb_width  <- diff(range(cols_with_px)) + 1
  aspect_ratio <- bb_width / bb_height
  
  # Perimeter: count mask pixels that have at least one non-mask neighbor
  # Use a simple shift-compare approach
  pad <- matrix(FALSE, nrow = nrow(mask) + 2, ncol = ncol(mask) + 2)
  pad[2:(nrow(mask) + 1), 2:(ncol(mask) + 1)] <- mask
  
  up    <- pad[1:nrow(mask),       2:(ncol(mask) + 1)]
  down  <- pad[3:(nrow(mask) + 2), 2:(ncol(mask) + 1)]
  left  <- pad[2:(nrow(mask) + 1), 1:ncol(mask)]
  right <- pad[2:(nrow(mask) + 1), 3:(ncol(mask) + 2)]
  
  is_boundary <- mask & !(up & down & left & right)
  perimeter <- sum(is_boundary)
  
  list(
    tumor_area   = n_pixels,
    aspect_ratio = aspect_ratio,
    perimeter    = perimeter
  )
}

# ----------------------------------------------------------------------------
# 4f. Full feature extraction for one image
# ----------------------------------------------------------------------------
extract_features <- function(processed_rds_path) {
  obj <- readRDS(processed_rds_path)
  img  <- obj$image
  mask <- obj$mask
  
  # First-order on brain pixels only
  fo <- first_order_features(img[mask])
  
  # GLCM: quantize then average across 4 directions
  q <- quantize_image(img, mask, n_levels = N_GRAY_LEVELS)
  glcm_list <- lapply(GLCM_DIRECTIONS, function(off) compute_glcm(q, off))
  # Average the 4 GLCMs (rotation invariance)
  glcm_avg <- Reduce("+", glcm_list) / length(glcm_list)
  tx <- glcm_features(glcm_avg)
  
  # Shape
  sh <- shape_features(mask)
  
  # Combine all 14 features
  c(fo, tx, sh)
}

# ============================================================================
# --- 5. EXTRACT FEATURES FOR ALL IMAGES -------------------------------------
# ============================================================================

n_images <- nrow(manifest)
feature_rows <- vector("list", n_images)

cat("Starting feature extraction for", n_images, "images...\n")
cat("GLCM: ", N_GRAY_LEVELS, "gray levels,", length(GLCM_DIRECTIONS), "directions averaged\n")
cat("Progress prints every 200 images.\n\n")

start_time <- Sys.time()

for (i in seq_len(n_images)) {
  row <- manifest[i, ]
  rds_path <- file.path(PROCESSED_DIR, paste0(row$image_id, ".rds"))
  
  feats <- tryCatch(
    extract_features(rds_path),
    error = function(e) list(error = conditionMessage(e))
  )
  
  if (!is.null(feats$error)) {
    # Failed row — fill with NAs + error flag
    feature_rows[[i]] <- data.frame(
      image_id       = row$image_id,
      split          = row$split,
      class          = row$class,
      mean_intensity = NA, variance = NA, skewness = NA, kurtosis = NA,
      energy = NA, entropy = NA,
      contrast = NA, correlation = NA, glcm_energy = NA,
      homogeneity = NA, dissimilarity = NA,
      tumor_area = NA, aspect_ratio = NA, perimeter = NA,
      extraction_error = feats$error,
      stringsAsFactors = FALSE
    )
  } else {
    feature_rows[[i]] <- data.frame(
      image_id       = row$image_id,
      split          = row$split,
      class          = row$class,
      mean_intensity = feats$mean_intensity,
      variance       = feats$variance,
      skewness       = feats$skewness,
      kurtosis       = feats$kurtosis,
      energy         = feats$energy,
      entropy        = feats$entropy,
      contrast       = feats$contrast,
      correlation    = feats$correlation,
      glcm_energy    = feats$glcm_energy,
      homogeneity    = feats$homogeneity,
      dissimilarity  = feats$dissimilarity,
      tumor_area     = feats$tumor_area,
      aspect_ratio   = feats$aspect_ratio,
      perimeter      = feats$perimeter,
      extraction_error = NA_character_,
      stringsAsFactors = FALSE
    )
  }
  
  if (i %% 200 == 0 || i == n_images) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    rate    <- i / elapsed
    eta     <- (n_images - i) / rate
    cat(sprintf("  [%d / %d]  elapsed: %.1f min  rate: %.0f img/min  ETA: %.1f min\n",
                i, n_images, elapsed, rate, eta))
  }
}

# --- 6. Combine and save -----------------------------------------------------
features_df <- do.call(rbind, feature_rows)
write.csv(features_df, FEATURES_PATH, row.names = FALSE)

# --- 7. Summary --------------------------------------------------------------
cat("\n=== FEATURE EXTRACTION SUMMARY ===\n")
cat("Total images:     ", nrow(features_df), "\n")
cat("Successful:       ", sum(is.na(features_df$extraction_error)), "\n")
cat("Errors:           ", sum(!is.na(features_df$extraction_error)), "\n")

cat("\nFeature summary (successful extractions):\n")
ok <- features_df[is.na(features_df$extraction_error), ]
feature_cols <- c("mean_intensity", "variance", "skewness", "kurtosis",
                  "energy", "entropy", "contrast", "correlation",
                  "glcm_energy", "homogeneity", "dissimilarity",
                  "tumor_area", "aspect_ratio", "perimeter")
print(round(sapply(ok[, feature_cols], function(x) {
  c(min = min(x, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    mean = mean(x, na.rm = TRUE),
    max = max(x, na.rm = TRUE),
    n_na = sum(is.na(x)))
}), 3))

cat("\nFeatures saved to:", FEATURES_PATH, "\n")
cat("\nPhase 3 complete.\n")
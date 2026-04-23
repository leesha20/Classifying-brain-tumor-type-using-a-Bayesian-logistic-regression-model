# ============================================================================
# 02_preprocess.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
#                           for Brain Tumor MRI Classification
# ----------------------------------------------------------------------------
# Purpose: Preprocess raw MRI images into a standardized form ready for
#          feature extraction.
#
# Pipeline per image:
#   1. Load the raw JPG
#   2. Convert to grayscale (average RGB channels)
#   3. Resize to TARGET_SIZE x TARGET_SIZE (default 128x128)
#   4. Background removal via Otsu thresholding (remove black padding)
#   5. Intensity normalization (z-score within brain region)
#   6. Save as a compact RDS file (one matrix per image)
#
# Input:   output/image_manifest.csv
#          data/raw/.../*.jpg (via file_path column)
# Output:  data/processed/<image_id>.rds  (one per image)
#          output/preprocessing_log.csv   (status per image)
# ============================================================================

# --- 1. Setup ----------------------------------------------------------------
rm(list = ls())

PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

library(EBImage)
library(dplyr)

# --- 2. Config ---------------------------------------------------------------
TARGET_SIZE     <- 128    # final image dimension (pixels)
PROCESSED_DIR   <- file.path(PROJECT_ROOT, "data", "processed")
MANIFEST_PATH   <- file.path(PROJECT_ROOT, "output", "image_manifest.csv")
LOG_PATH        <- file.path(PROJECT_ROOT, "output", "preprocessing_log.csv")

# Create output directory
dir.create(PROCESSED_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 3. Load manifest --------------------------------------------------------
manifest <- read.csv(MANIFEST_PATH, stringsAsFactors = FALSE)
cat("Loaded manifest:", nrow(manifest), "images\n")

# --- 4. Core preprocessing function ------------------------------------------
# Takes a file path, returns a normalized grayscale matrix
preprocess_image <- function(file_path, target_size = TARGET_SIZE) {
  # Load
  img <- readImage(file_path)
  
  # Convert to grayscale:
  # EBImage stores color as a 3D array (w, h, 3). If it's already grayscale
  # (2D), just keep it. If color, average the channels.
  if (length(dim(img)) == 3 && dim(img)[3] >= 3) {
    gray <- (img[,,1] + img[,,2] + img[,,3]) / 3
  } else if (length(dim(img)) == 3 && dim(img)[3] == 1) {
    gray <- img[,,1]
  } else {
    gray <- img
  }
  
  # Resize — EBImage::resize works on a 2D matrix
  gray_img <- Image(gray, colormode = Grayscale)
  resized  <- resize(gray_img, w = target_size, h = target_size)
  mat      <- imageData(resized)  # pull out the underlying matrix
  
  # Background removal via Otsu threshold:
  # MRI images have a lot of black border padding. Otsu finds a threshold
  # that separates dark background from bright brain tissue.
  thresh <- otsu(Image(mat, colormode = Grayscale))
  brain_mask <- mat > thresh
  
  # Intensity normalization: z-score WITHIN the brain region only.
  # This matters — normalizing over the whole image (including black padding)
  # would make every image's stats dominated by how much padding it has.
  brain_pixels <- mat[brain_mask]
  if (length(brain_pixels) > 10) {  # sanity: at least 10 brain pixels
    mu    <- mean(brain_pixels)
    sigma <- sd(brain_pixels)
    if (sigma > 0) {
      mat_norm <- (mat - mu) / sigma
    } else {
      mat_norm <- mat - mu
    }
  } else {
    # If the mask failed (e.g. all-black image), return raw normalized
    mat_norm <- (mat - mean(mat)) / (sd(mat) + 1e-8)
  }
  
  # Zero out background so it doesn't contaminate feature extraction later
  mat_norm[!brain_mask] <- 0
  
  list(
    image       = mat_norm,
    brain_mask  = brain_mask,
    n_brain_px  = sum(brain_mask),
    mean_brain  = if (exists("mu")) mu else NA,
    sd_brain    = if (exists("sigma")) sigma else NA
  )
}

# --- 5. Process all images with progress tracking ---------------------------
n_images <- nrow(manifest)
log_rows <- vector("list", n_images)

cat("Starting preprocessing of", n_images, "images...\n")
cat("Target size:", TARGET_SIZE, "x", TARGET_SIZE, "\n")
cat("Progress prints every 200 images.\n\n")

start_time <- Sys.time()

for (i in seq_len(n_images)) {
  row <- manifest[i, ]
  
  result <- tryCatch({
    preprocess_image(row$file_path)
  }, error = function(e) {
    list(error = conditionMessage(e))
  })
  
  if (!is.null(result$error)) {
    log_rows[[i]] <- data.frame(
      image_id    = row$image_id,
      status      = "ERROR",
      n_brain_px  = NA,
      mean_brain  = NA,
      sd_brain    = NA,
      error_msg   = result$error,
      stringsAsFactors = FALSE
    )
  } else {
    # Save the processed image as RDS (fast to reload later)
    out_path <- file.path(PROCESSED_DIR, paste0(row$image_id, ".rds"))
    saveRDS(list(image = result$image, mask = result$brain_mask), out_path)
    
    log_rows[[i]] <- data.frame(
      image_id    = row$image_id,
      status      = "OK",
      n_brain_px  = result$n_brain_px,
      mean_brain  = result$mean_brain,
      sd_brain    = result$sd_brain,
      error_msg   = NA,
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

# --- 6. Save the log ---------------------------------------------------------
log_df <- do.call(rbind, log_rows)
write.csv(log_df, LOG_PATH, row.names = FALSE)

# --- 7. Summary report -------------------------------------------------------
cat("\n=== PREPROCESSING SUMMARY ===\n")
cat("Total images:", n_images, "\n")
cat("Successful:  ", sum(log_df$status == "OK"), "\n")
cat("Errors:      ", sum(log_df$status == "ERROR"), "\n")

if (sum(log_df$status == "ERROR") > 0) {
  cat("\nFirst few errors:\n")
  print(head(log_df[log_df$status == "ERROR", c("image_id", "error_msg")]))
}

cat("\nBrain pixel count distribution (% of 128x128 = 16384):\n")
ok_log <- log_df[log_df$status == "OK", ]
print(summary(ok_log$n_brain_px / (TARGET_SIZE^2) * 100))

cat("\nLog saved to:", LOG_PATH, "\n")
cat("Processed images saved to:", PROCESSED_DIR, "\n")

# --- 8. Visual spot check: display 4 random processed images (one per class) -
# This opens a plot in RStudio so you can eyeball the preprocessing
cat("\nGenerating spot-check plot...\n")

set.seed(42)
spot_check <- do.call(rbind, lapply(
  unique(manifest$class),
  function(cl) manifest[manifest$class == cl, ][sample(sum(manifest$class == cl), 1), ]
))

par(mfrow = c(2, 4), mar = c(2, 2, 2, 1))
for (i in seq_len(nrow(spot_check))) {
  row <- spot_check[i, ]
  # Original
  orig <- readImage(row$file_path)
  if (length(dim(orig)) == 3) orig <- (orig[,,1] + orig[,,2] + orig[,,3]) / 3
  image(t(orig)[, nrow(orig):1], col = gray.colors(256),
        main = paste("Original\n", row$class), axes = FALSE)
  
  # Processed
  proc <- readRDS(file.path(PROCESSED_DIR, paste0(row$image_id, ".rds")))$image
  image(t(proc)[, nrow(proc):1], col = gray.colors(256),
        main = "Processed", axes = FALSE)
}
par(mfrow = c(1, 1))

cat("\nPhase 2 complete.\n")
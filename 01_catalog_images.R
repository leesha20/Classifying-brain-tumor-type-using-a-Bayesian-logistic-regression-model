# ============================================================================
# 01_catalog_images.R
# STAT 5353 Final Project — Bayesian Multinomial Logistic Regression
#                           for Brain Tumor MRI Classification
# ----------------------------------------------------------------------------
# Purpose: Walk the raw data folders and build a manifest CSV with one row per
#          image. This manifest will be the "source of truth" for every later
#          phase (preprocessing, feature extraction, modeling).
#
# Input:   data/raw/Training/<class>/*.jpg
#          data/raw/Testing/<class>/*.jpg
# Output:  output/image_manifest.csv
# ============================================================================

# --- 1. Setup -----------------------------------------------------------------
# Clear workspace so we start fresh every time this script runs
rm(list = ls())

# Set the project root. Change this path if you moved your project folder.
PROJECT_ROOT <- "~/Documents/stat5353_final"
setwd(PROJECT_ROOT)

# Load libraries
library(dplyr)     # for data frame manipulation
library(EBImage)   # we'll use this later, but load now to catch install issues

# --- 2. Define paths ---------------------------------------------------------
raw_dir       <- file.path(PROJECT_ROOT, "data", "raw")
training_dir  <- file.path(raw_dir, "Training")
testing_dir   <- file.path(raw_dir, "Testing")

# Sanity check: do these folders exist?
stopifnot(dir.exists(training_dir))
stopifnot(dir.exists(testing_dir))

cat("Training folder:", training_dir, "\n")
cat("Testing folder:",  testing_dir, "\n\n")

# --- 3. Catalog function -----------------------------------------------------
# Given a split directory (Training or Testing), return a data frame with
# one row per image file found inside its class subfolders.
catalog_split <- function(split_dir, split_name) {
  class_dirs <- list.dirs(split_dir, recursive = FALSE)
  
  # Collect all image file paths per class
  out <- do.call(rbind, lapply(class_dirs, function(cd) {
    files <- list.files(cd, pattern = "\\.(jpg|jpeg|png)$",
                        full.names = TRUE, ignore.case = TRUE)
    if (length(files) == 0) return(NULL)
    data.frame(
      file_path  = files,
      class      = basename(cd),
      split      = split_name,
      stringsAsFactors = FALSE
    )
  }))
  
  return(out)
}

# --- 4. Build the manifest ---------------------------------------------------
train_manifest <- catalog_split(training_dir, "train")
test_manifest  <- catalog_split(testing_dir,  "test")

manifest <- rbind(train_manifest, test_manifest)

# Add a unique image_id (image_00001, image_00002, ...)
manifest$image_id <- sprintf("image_%05d", seq_len(nrow(manifest)))

# Reorder columns for readability
manifest <- manifest %>%
  select(image_id, split, class, file_path)

# --- 5. Summary report -------------------------------------------------------
cat("=== MANIFEST SUMMARY ===\n")
cat("Total images:", nrow(manifest), "\n\n")

cat("By split:\n")
print(table(manifest$split))

cat("\nBy class (train):\n")
print(table(manifest$class[manifest$split == "train"]))

cat("\nBy class (test):\n")
print(table(manifest$class[manifest$split == "test"]))

# --- 6. Save ----------------------------------------------------------------
output_path <- file.path(PROJECT_ROOT, "output", "image_manifest.csv")
write.csv(manifest, output_path, row.names = FALSE)
cat("\nManifest saved to:", output_path, "\n")

# --- 7. Spot check: load one random image to confirm EBImage works ----------
set.seed(42)
sample_row <- manifest[sample(nrow(manifest), 1), ]
cat("\nSpot-checking random image:\n")
cat("  ID:    ", sample_row$image_id, "\n")
cat("  Class: ", sample_row$class, "\n")
cat("  Path:  ", sample_row$file_path, "\n")

img <- readImage(sample_row$file_path)
cat("  Dimensions:", paste(dim(img), collapse = " x "), "\n")
cat("  Color mode:", colorMode(img), "  (0=grayscale, 2=color)\n")
cat("  Intensity range:", round(range(img), 3), "\n")

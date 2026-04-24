# Classifying Brain Tumor Type Using a Bayesian Logistic Regression Model

**Author:** Sriya Leesha Gourammagari
**Course:** STAT 5353 — Final Project

## Project Overview

This project applies Bayesian multinomial logistic regression to classify four categories of brain MRI scans (glioma, meningioma, pituitary tumor, and no tumor) using radiomic features extracted from the Bhuvaji Brain Tumor Classification dataset (3,264 images). The analysis compares frequentist maximum likelihood estimation against two Bayesian models with different prior specifications, and evaluates the practical value of uncertainty quantification in medical image classification.

## Research Questions

1. Can a Bayesian logistic regression model classify brain tumor type from MRI-derived radiomic features?
  2. Which features are the most significant predictors of tumor classification?
  3. How does the Bayesian model compare in accuracy to the frequentist MLE approach?
  4. How do different prior specifications affect posterior parameter estimates and predictive performance?
  
  ## Methodology
  
  **Data:** 3,264 T1-weighted MRI images from [Bhuvaji et al. (2020)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri), split into 2,870 training and 394 test images across four classes.

**Preprocessing:** Each image was converted to grayscale, resized to 128×128 pixels, skull-stripped using Otsu thresholding, and intensity-normalized via z-scoring within the brain region.

**Feature extraction:** 14 radiomic features were extracted per image (first-order statistics, Gray-Level Co-occurrence Matrix texture features averaged across 4 directions at 32 gray levels, and shape descriptors). After removing zero-variance and highly correlated features (|r| > 0.95), 10 features were retained for modeling.

**Modeling:** A custom Metropolis-Hastings sampler was implemented for a 4-class multinomial logistic regression, with the reference class set to "no tumor." Two priors were compared:
  - **Model A:** Normal(0, 10) — weakly informative
- **Model B:** Normal(0, 2) — moderately regularizing

Each model was fit with 4 chains of 50,000 iterations (10,000 burn-in, thinned by 10), using a covariance-informed proposal (Rosenthal 2011 optimal scaling: 2.38²/d × Σ_MLE).

**Evaluation:** Compared to a frequentist multinomial logit baseline (`nnet::multinom`) on the held-out test set. Classification metrics include accuracy, per-class precision/recall/F1, and macro-averaged F1. Bayesian-specific outputs include posterior credible intervals and predictive entropy.

## Results

| Model | Accuracy | Macro F1 | Glioma F1 |
  |---|---|---|---|
  | Frequentist MLE | 41.37% | 0.404 | 0.232 |
  | Bayesian A (σ=10) | 41.37% | 0.404 | 0.232 |
  | Bayesian B (σ=2) | 41.88% | 0.411 | 0.257 |
  
  **MCMC convergence:** All 33 parameters achieved Gelman-Rubin R-hat < 1.01 and effective sample size > 1,400 across both models, indicating reliable convergence.

**Key finding:** With sufficient training data, Bayesian and frequentist point predictions converged to nearly identical values. The Bayesian framework's practical value in this setting lies in uncertainty quantification — credible intervals for coefficients and predictive entropy for individual test cases — rather than improved classification accuracy.

## Repository Structure
.
├── 01_catalog_images.R            Image manifest construction
├── 02_preprocess.R                Grayscale, resize, Otsu, z-score
├── 03_features.R                  Radiomic feature extraction
├── 04a_feature_prep.R             Feature standardization & pruning
├── 04b_frequentist_baseline.R     MLE multinomial logit baseline
├── 04c_mh_sampler_v2.R            Custom Metropolis-Hastings sampler
├── 04d_diagnostics.R              MCMC convergence diagnostics
├── 04e_evaluate.R                 Test set evaluation
├── 05_generate_figures.R          Presentation figures
└── output/
├── features/                  Extracted feature CSVs
├── figures/                   Generated PNG figures
└── models/                    Fitted model RDS files

## Reproducibility

1. Clone this repository.
2. Download the Bhuvaji MRI dataset from Kaggle and extract into `data/raw/` (contains `Training/` and `Testing/` folders with class subdirectories).
3. Install required R packages: `EBImage` (Bioconductor), `coda`, `nnet`, `mvtnorm`, `ggplot2`, `dplyr`.
4. Run scripts in numerical order (`01_catalog_images.R` through `05_generate_figures.R`).

## Limitations

- Tumor segmentation used whole-brain radiomics rather than lesion-specific ROIs.
- Leave-one-out cross-validation was not performed due to compute constraints; held-out test set evaluation was used instead.
- Classical radiomic features substantially underperform deep learning benchmarks (~90%) for this task; this project focuses on Bayesian methodology rather than state-of-the-art classification accuracy.

## References

- Bhuvaji, S., Kadam, A., Bhumkar, P., Dedge, S., & Kanchan, S. (2020). *Brain Tumor Classification (MRI) Dataset.* Kaggle.
- Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, 3(6), 610–621.
- Rosenthal, J. S. (2011). Optimal proposal distributions and adaptive MCMC. *Handbook of Markov Chain Monte Carlo*, 93–112.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
# EEG feature boxplots for selected inter-feature correlation variables

########################## Install packages ##########################
# install.packages("ggplot2")

########################## Activate packages #########################
library(ggplot2)

##################### CTS & MANUAL UPDATES ###########################
TIMESTAMP <- "APR_15_2026"
ROOT_DIR <- "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/"
DATABASE_PATH <- file.path(ROOT_DIR, "DATA/OUTPUTS/Clustered")
STATS_PATH <- file.path(ROOT_DIR, "DATA/OUTPUTS/Stats")
database_filepath <- file.path(DATABASE_PATH, "clustered_SOM_Q1K_CHU_MHC_BC_DATA_MAR_09_2026.csv")

########################## Import dataset ############################
db <- read.csv(database_filepath)

# Only these columns enter the inter-feature correlation (edit here; independent of eeg_features)
corr_features <- c(
  "hurst_2s",
  "pow_per_delta_2s",
  "pow_per_low_gamma_2s",
  "pow_per_high_gamma_2s",
  "higuchi_fd_5s"
)

# Set reference cluster for LM models (used to keep cluster ordering consistent)
reference_cluster <- 3

# Keep only features that exist in the dataset
present_features <- corr_features[corr_features %in% names(db)]
missing_features <- setdiff(corr_features, present_features)

if (length(missing_features) > 0) {
  warning(
    paste0(
      "These requested features are missing from the dataset and will be skipped: ",
      paste(missing_features, collapse = ", ")
    )
  )
}

if (!"cluster" %in% names(db)) {
  stop("The input dataset does not contain a 'cluster' column.")
}

if (length(present_features) == 0) {
  stop("None of the requested corr_features were found in the dataset.")
}

# Required columns for prepare_feature_data logic used in LM section
required_cols <- c("participant_id", "cluster", "age_at_test", "sex")
missing_required_cols <- setdiff(required_cols, names(db))
if (length(missing_required_cols) > 0) {
  stop(
    paste0(
      "The input dataset is missing required columns: ",
      paste(missing_required_cols, collapse = ", ")
    )
  )
}

df <- db
df$cluster <- factor(df$cluster)
df$sex <- factor(df$sex)

# Cluster color palette (raw clusters 0-3 displayed as Cluster 1-4)
cluster_colors <- c(
  "0" = "#6BAED6", # Light blue
  "1" = "#08519C", # Dark blue
  "2" = "#74C476", # Light green
  "3" = "#238B45"  # Dark green
)

# ---- Helper function to prepare data for analysis with participant exclusions ----#
prepare_feature_data <- function(df, feat, verbose = TRUE) {
  # Select relevant columns including participant_id
  dat_feat <- df[, c("participant_id", "cluster", "age_at_test", "sex", feat)]
  dat_feat <- na.omit(dat_feat) # Remove rows with NA

  # Exclude participants based on feature type (2s vs 5s)
  if (grepl("_2s$", feat)) {
    # For 2s features: exclude Q1K_HSJ_1525-1012_P
    n_before <- nrow(dat_feat)
    dat_feat <- dat_feat[dat_feat$participant_id != "Q1K_HSJ_1525-1012_P", ]
    if (verbose && nrow(dat_feat) < n_before) {
      cat("Excluded participant Q1K_HSJ_1525-1012_P for 2s feature\n")
    }
  } else if (grepl("_5s$", feat)) {
    # For 5s features: exclude Q1K_HSJ_1525-1012_P and Q1K_HSJ_1525-1083_P
    n_before <- nrow(dat_feat)
    dat_feat <- dat_feat[!dat_feat$participant_id %in% c("Q1K_HSJ_1525-1012_P", "Q1K_HSJ_1525-1083_P"), ]
    if (verbose && nrow(dat_feat) < n_before) {
      cat("Excluded participants Q1K_HSJ_1525-1012_P and Q1K_HSJ_1525-1083_P for 5s feature\n")
    }
  }

  # Remove participant_id column before analysis
  dat_feat <- dat_feat[, c("cluster", "age_at_test", "sex", feat)]

  return(dat_feat)
}

# Pretty display labels for feature names in plots
pretty_feature_name <- function(feat) {
  custom_labels <- c(
    "husrt_2s" = "Hurst",
    "hurst_2s" = "Hurst",
    "pow_per_delta_2s" = "Delta (1 - 4 Hz)",
    "pow_per_low_gamma_2s" = "Low Gamma (30 - 57 Hz)",
    "pow_per_high_gamma_2s" = "High Gamma (63 - 80 Hz)",
    "higuchi_fd_5s" = "Higuchi FD"
  )

  if (feat %in% names(custom_labels)) {
    return(custom_labels[[feat]])
  }

  label <- gsub("_", " ", feat)
  label <- gsub("\\b[25]s\\b", "", label)
  label <- gsub("\\s+", " ", label)
  trimws(label)
}

############################ Boxplots ################################
output_pdf <- file.path(STATS_PATH, paste0("eeg_feature_boxplots_corr_features_", TIMESTAMP, ".pdf"))

pdf(output_pdf, width = 8, height = 6)

for (feat in present_features) {
  feat_data <- prepare_feature_data(df, feat, verbose = TRUE)
  feat_data$cluster <- droplevels(factor(feat_data$cluster))
  feat_data$sex <- droplevels(factor(feat_data$sex))

  if (nrow(feat_data) == 0 || nlevels(feat_data$cluster) < 2) {
    warning(paste0("Skipping feature due to insufficient data after exclusions: ", feat))
    next
  }

  # Keep the same reference cluster used in LM models (if present)
  if (reference_cluster %in% levels(feat_data$cluster)) {
    feat_data$cluster <- relevel(feat_data$cluster, ref = as.character(reference_cluster))
  }

  n_total <- nrow(feat_data)
  feat_label <- pretty_feature_name(feat)

  p <- ggplot(feat_data, aes(x = cluster, y = .data[[feat]])) +
    geom_boxplot(color = "black", fill = NA, linewidth = 0.8, outlier.shape = NA) +
    geom_jitter(aes(color = cluster), width = 0.15, height = 0, alpha = 0.6, size = 5) +
    labs(
      title = paste0("Boxplot by Cluster: ", feat_label, " (N = ", n_total, ")"),
      x = "Cluster",
      y = feat_label
    ) +
    scale_color_manual(values = cluster_colors, drop = FALSE) +
    scale_x_discrete(labels = function(x) paste("Cluster", as.integer(as.character(x)) + 1L)) +
    theme_bw() +
    theme(
      legend.position = "none",
      axis.text = element_text(size = 14),
      axis.title = element_text(size = 16),
      plot.title = element_text(size = 16, face = "bold")
    )

  print(p)
}

dev.off()

cat("Boxplots saved to:\n")
cat(output_pdf, "\n")

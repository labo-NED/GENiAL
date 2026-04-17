# Behavioral feature boxplots based on LM preprocessing pipeline

########################## Install packages ##########################
# install.packages("ggplot2")

########################## Activate packages #########################
library(ggplot2)

##################### CTS & MANUAL UPDATES ###########################
TIMESTAMP <- "APR_17_2026"
ROOT_DIR <- "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/"
DATABASE_PATH <- file.path(ROOT_DIR, "DATA/OUTPUTS/Clustered")
STATS_PATH <- file.path(ROOT_DIR, "DATA/OUTPUTS/Stats")
database_filepath <- file.path(DATABASE_PATH, "clustered_SOM_Q1K_CHU_MHC_BC_DATA_MAR_09_2026.csv")

########################## Import dataset ############################
db <- read.csv(database_filepath)

# Set reference cluster for LM models
reference_cluster <- 3

# ---- LM for each behavioral measure (same model as EEG: cluster vs ref + age + sex) ----#
behavioral_features <- c(
  "SRS_restrictive_repetitive_tscore",
  "SRS_social_communication_tscore",
  "SRS_social_cognition_tscore",
  "attention_deficit_hyperactivity_tscore",
  "nonverbal_iq"
)
behavioral_features <- behavioral_features[behavioral_features %in% names(db)]

if (!"cluster" %in% names(db)) {
  stop("The input dataset does not contain a 'cluster' column.")
}

if (length(behavioral_features) == 0) {
  stop("None of the requested behavioral features were found in the dataset.")
}

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
    n_before <- nrow(dat_feat)
    dat_feat <- dat_feat[dat_feat$participant_id != "Q1K_HSJ_1525-1012_P", ]
    if (verbose && nrow(dat_feat) < n_before) {
      cat("Excluded participant Q1K_HSJ_1525-1012_P for 2s feature\n")
    }
  } else if (grepl("_5s$", feat)) {
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

# Pretty display labels for behavioral features
pretty_behavioral_name <- function(feat) {
  custom_labels <- c(
    "SRS_restrictive_repetitive_tscore" = "Restrictive and Repetitive Behavior",
    "SRS_social_communication_tscore" = "Social Communication",
    "SRS_social_cognition_tscore" = "Social Cognition",
    "attention_deficit_hyperactivity_tscore" = "ADHD Traits",
    "nonverbal_iq" = "Nonverbal IQ"
  )

  if (feat %in% names(custom_labels)) {
    return(custom_labels[[feat]])
  }

  label <- gsub("_", " ", feat)
  label <- gsub("\\s+", " ", label)
  trimws(label)
}

############################ Boxplots ################################
output_pdf <- file.path(STATS_PATH, paste0("behavioral_feature_boxplots_", TIMESTAMP, ".pdf"))

pdf(output_pdf, width = 8, height = 6)

for (feat in behavioral_features) {
  dat_feat <- prepare_feature_data(df, feat, verbose = TRUE)
  dat_feat$cluster <- droplevels(dat_feat$cluster)
  dat_feat$sex <- droplevels(dat_feat$sex)

  if (nrow(dat_feat) == 0 || nlevels(dat_feat$cluster) < 2) {
    warning(paste0("Skipping feature due to insufficient data after NA removal: ", feat))
    next
  }

  if (reference_cluster %in% levels(dat_feat$cluster)) {
    dat_feat$cluster <- relevel(dat_feat$cluster, ref = as.character(reference_cluster))
  }

  n_total <- nrow(dat_feat)
  feat_label <- pretty_behavioral_name(feat)

  p <- ggplot(dat_feat, aes(x = cluster, y = .data[[feat]])) +
    geom_boxplot(color = "black", fill = NA, linewidth = 0.8, outlier.shape = NA) +
    geom_jitter(aes(color = cluster), width = 0.15, height = 0, alpha = 0.6, size = 5) +
    labs(
      # title = paste0("Boxplot by Cluster: ", feat_label, " (N = ", n_total, ")"),
      x = NULL,
      y = feat_label
    ) +
    scale_y_continuous(limits = c(30, 100)) +
    scale_color_manual(values = cluster_colors, drop = FALSE) +
    scale_x_discrete(labels = function(x) paste("Cluster", as.integer(as.character(x)) + 1L)) +
    theme_bw() +
    theme(
      legend.position = "none",
      axis.text = element_text(size = 20),
      axis.title = element_text(size = 20),
      # plot.title = element_text(size = 16, face = "bold")
    )

  print(p)
}

dev.off()

cat("Behavioral boxplots saved to:\n")
cat(output_pdf, "\n")

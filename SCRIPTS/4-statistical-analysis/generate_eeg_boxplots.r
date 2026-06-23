# EEG feature boxplots for selected inter-feature correlation variables

########################## Install packages ##########################
# install.packages("ggplot2")

########################## Activate packages #########################
library(ggplot2)

##################### CTS & MANUAL UPDATES ###########################
TIMESTAMP <- "JUN_16_2026"
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
output_pdf <- file.path(
  STATS_PATH,
  paste0("eeg_feature_boxplots_corr_features_", TIMESTAMP, ".pdf")
)

pdf(output_pdf, width = 5, height = 4.5)

for (feat in present_features) {
  
  feat_data <- prepare_feature_data(df, feat, verbose = TRUE)
  feat_data$cluster <- droplevels(factor(feat_data$cluster))
  feat_data$sex <- droplevels(factor(feat_data$sex))
  
  if (nrow(feat_data) == 0 || nlevels(feat_data$cluster) < 2) {
    warning(paste0(
      "Skipping feature due to insufficient data after exclusions: ",
      feat
    ))
    next
  }
  
  # Keep same reference cluster used in LM models
  if (reference_cluster %in% levels(feat_data$cluster)) {
    feat_data$cluster <- relevel(
      feat_data$cluster,
      ref = as.character(reference_cluster)
    )
  }
  
  n_total <- nrow(feat_data)
  
  # Rescale very small power values for plotting only
  scale_factor <- 1
  
  if (feat == "pow_per_delta_2s") {
    # scale_factor <- 1e12
    feat_label <- "Delta Power"
  } else if (feat == "pow_per_low_gamma_2s") {
    # scale_factor <- 1e14
    feat_label <- "Low Gamma Power"
  } else if (feat == "pow_per_high_gamma_2s") {
    # scale_factor <- 1e14
    feat_label <- "High Gamma Power"
  } else {
    feat_label <- pretty_feature_name(feat)
  }
  
  plot_value <- paste0(feat, "_plot")
  feat_data[[plot_value]] <- feat_data[[feat]] * scale_factor
  
  # Calculate position of significance bar
  y_max <- max(feat_data[[plot_value]], na.rm = TRUE)
  y_min <- min(feat_data[[plot_value]], na.rm = TRUE)
  y_range <- y_max - y_min
  
  if (y_range == 0) {
    y_range <- abs(y_max) * 0.1
  }
  if (y_range == 0) {
    y_range <- 1
  }
  
  bar_y <- y_max + 0.08 * y_range
  tick_y <- y_max + 0.04 * y_range
  star_y <- y_max + 0.11 * y_range
  
  p <- ggplot(
    feat_data,
    aes(x = cluster, y = .data[[plot_value]])
  ) +
    
    geom_boxplot(
      color = "black",
      fill = NA,
      linewidth = 0.8,
      outlier.shape = NA
    ) +
    
    geom_jitter(
      aes(color = cluster),
      width = 0.15,
      height = 0,
      alpha = 0.6,
      size = 3.5
    ) +
    
    labs(
      x = NULL,
      y = feat_label
    ) +
    
    scale_color_manual(
      values = cluster_colors,
      drop = FALSE
    ) +
    
    scale_x_discrete(
      labels = function(x)
        paste(as.integer(as.character(x)) + 1L)
    ) +
    
    # Significance bar: Cluster 2 vs Cluster 4
    annotate(
      "segment",
      x = 1, xend = 3,
      y = bar_y, yend = bar_y,
      linewidth = 0.8
    ) +
    
    annotate(
      "segment",
      x = 1, xend = 1,
      y = tick_y, yend = bar_y,
      linewidth = 0.8
    ) +
    
    annotate(
      "segment",
      x = 3, xend = 3,
      y = tick_y, yend = bar_y,
      linewidth = 0.8
    ) +
    
    annotate(
      "text",
      x = 2,
      y = star_y,
      label = "*",
      size = 8
    ) +
    
    coord_cartesian(
      ylim = c(
        y_min,
        y_max + 0.18 * y_range
      )
    ) +
    
    theme_bw(base_size = 16) +
    
    theme(
      legend.position = "none",
      
      axis.text.x = element_text(
        size = 24,
        color = "black"
      ),
      
      axis.text.y = element_text(
        size = 18,
        color = "black"
      ),
      
      axis.title.y = element_text(
        size = 24,
        color = "black"
      ),
      
      axis.title.x = element_blank(),
      
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      plot.margin = margin(
        t = 8,
        r = 8,
        b = 8,
        l = 8
      )
    )
  
  print(p)
}

dev.off()

cat("Boxplots saved to:\n")
cat(output_pdf, "\n")

############################ Line Plot (Emmeans) ######################
library(emmeans)
library(patchwork)

output_pdf_line <- file.path(STATS_PATH, paste0("eeg_feature_lineplot_corr_features_", TIMESTAMP, ".pdf"))

# Poster-matching theme
theme_poster <- function() {
  theme_bw() +
    theme(
      legend.position = "none",
      axis.text = element_text(size = 14, color = "black"),
      axis.title.y = element_text(size = 14, color = "black"),
      axis.title.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", linewidth = 0.8),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )
}

# Same cluster colors as your boxplots
cluster_colors <- c(
  "Cluster 1" = "#08519C",  # Dark blue
  "Cluster 2" = "#74C476",  # Light green
  "Cluster 3" = "#238B45",  # Dark green
  "Cluster 4" = "#6BAED6"   # Light blue
)

plot_list <- list()

for (feat in present_features) {
  feat_data <- prepare_feature_data(df, feat, verbose = FALSE)
  feat_data$cluster <- droplevels(factor(feat_data$cluster))
  feat_data$sex <- droplevels(factor(feat_data$sex))
  
  if (nrow(feat_data) == 0 || nlevels(feat_data$cluster) < 2) next
  
  if (reference_cluster %in% levels(feat_data$cluster)) {
    feat_data$cluster <- relevel(feat_data$cluster, ref = as.character(reference_cluster))
  }
  
  formula <- as.formula(paste(feat, "~ cluster + age_at_test + sex"))
  model <- lm(formula, data = feat_data)
  
  emm <- as.data.frame(emmeans(model, ~ cluster))
  emm$cluster_label <- paste("Cluster", as.integer(as.character(emm$cluster)) + 1L)
  emm$cluster_label <- factor(emm$cluster_label, levels = paste("Cluster", 1:4))
  
  feat_label <- pretty_feature_name(feat)
  y_max   <- max(emm$upper.CL)
  y_range <- max(emm$upper.CL) - min(emm$lower.CL)
  step    <- y_range * 0.15
  
  p <- ggplot(emm, aes(x = cluster_label, y = emmean, group = 1, color = cluster_label)) +
    geom_line(linewidth = 0.9, color = "grey40", linetype = "solid") +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                  width = 0.12, linewidth = 0.8, color = "grey40") +
    geom_point(size = 5, shape = 21,
               aes(fill = cluster_label), color = "white", stroke = 1.2) +
    scale_fill_manual(values = cluster_colors) +
    scale_color_manual(values = cluster_colors) +
    scale_x_discrete(labels = c(
      "Cluster 1" = "Cl. 1",
      "Cluster 2" = "Cl. 2",
      "Cluster 3" = "Cl. 3",
      "Cluster 4" = "Cl. 4"
    )) +
    labs(x = NULL, y = feat_label) +
    theme_poster() +
    # Significance bar between Cluster 2 and Cluster 4
    annotate("segment",
             x = "Cluster 2", xend = "Cluster 4",
             y = y_max + step, yend = y_max + step,
             linewidth = 0.6, color = "black") +
    annotate("segment",
             x = "Cluster 2", xend = "Cluster 2",
             y = y_max + step, yend = y_max + step * 0.85,
             linewidth = 0.6, color = "black") +
    annotate("segment",
             x = "Cluster 4", xend = "Cluster 4",
             y = y_max + step, yend = y_max + step * 0.85,
             linewidth = 0.6, color = "black") +
    annotate("text",
             x = 3,
             y = y_max + step * 1.2,
             label = "*", size = 6, color = "black") +
    coord_cartesian(ylim = c(
      min(emm$lower.CL) - y_range * 0.05,
      y_max + step * 1.8
    ))
  
  plot_list[[feat]] <- p
}

# Combine into 3-top + 2-bottom layout with explicit feature order
combined <- (plot_list[["pow_per_delta_2s"]] | plot_list[["pow_per_low_gamma_2s"]] | plot_list[["pow_per_high_gamma_2s"]]) /
  (plot_list[["hurst_2s"]] | plot_list[["higuchi_fd_5s"]]) +
  plot_layout(heights = c(1, 1))

pdf(output_pdf_line, width = 12, height = 6)
print(combined)
dev.off()

cat("Line plots saved to:\n")
cat(output_pdf_line, "\n")

# Behavioral feature boxplots based on LM preprocessing pipeline

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
    "SRS_restrictive_repetitive_tscore" = "RRB",
    "SRS_social_communication_tscore" = "Social Communication",
    "SRS_social_cognition_tscore" = "Social Cognition",
    "attention_deficit_hyperactivity_tscore" = "ADHD Traits",
    "nonverbal_iq" = "NVIQ"
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

plot_order <- c("3", "0", "1", "2")

# Significant comparisons against reference Cluster 4
# Raw cluster coding: 0 = Cluster 1, 1 = Cluster 2, 2 = Cluster 3, 3 = Cluster 4
sig_pairs <- list(
  "SRS_restrictive_repetitive_tscore"      = list(c("0", "3"), c("1", "3"), c("2", "3")),
  "SRS_social_communication_tscore"        = list(c("0", "3"), c("1", "3"), c("2", "3")),
  "SRS_social_cognition_tscore"            = list(c("0", "3"), c("1", "3"), c("2", "3")),
  "attention_deficit_hyperactivity_tscore" = list(c("0", "3"), c("1", "3")),
  "nonverbal_iq"                           = list(c("1", "3"), c("2", "3"))
)

pdf(output_pdf, width = 5, height = 4.5)

for (feat in behavioral_features) {
  
  dat_feat <- prepare_feature_data(df, feat, verbose = TRUE)
  dat_feat$cluster <- droplevels(factor(dat_feat$cluster))
  dat_feat$sex <- droplevels(factor(dat_feat$sex))
  
  if (nrow(dat_feat) == 0 || nlevels(dat_feat$cluster) < 2) {
    warning(paste0("Skipping feature due to insufficient data after NA removal: ", feat))
    next
  }
  
  # Force plotting order: Cluster 1, 2, 3, 4
  dat_feat$cluster <- factor(
    dat_feat$cluster,
    levels = plot_order
  )
  
  n_total <- nrow(dat_feat)
  feat_label <- pretty_behavioral_name(feat)
  
  y_max <- max(dat_feat[[feat]], na.rm = TRUE)
  y_min <- min(dat_feat[[feat]], na.rm = TRUE)
  y_range <- y_max - y_min
  
  if (y_range == 0) {
    y_range <- abs(y_max) * 0.1
  }
  if (y_range == 0) {
    y_range <- 1
  }
  
  p <- ggplot(
    dat_feat,
    aes(x = cluster, y = .data[[feat]])
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
      limits = plot_order,
      labels = c(
        "3" = "4",
        "0" = "1",
        "1" = "2",
        "2" = "3"
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
  
  # Add significance bars
  if (feat %in% names(sig_pairs)) {
    
    pairs <- sig_pairs[[feat]]
    
    for (i in seq_along(pairs)) {
      pair <- pairs[[i]]
      
      bar_y <- y_max + (0.08 + (i - 1) * 0.08) * y_range
      tick_y <- bar_y - 0.03 * y_range
      star_y <- bar_y + 0.025 * y_range
      
      p <- p +
        annotate(
          "segment",
          x = pair[1], xend = pair[2],
          y = bar_y, yend = bar_y,
          linewidth = 0.8
        ) +
        
        annotate(
          "segment",
          x = pair[1], xend = pair[1],
          y = tick_y, yend = bar_y,
          linewidth = 0.8
        ) +
        
        annotate(
          "segment",
          x = pair[2], xend = pair[2],
          y = tick_y, yend = bar_y,
          linewidth = 0.8
        ) +
        
        annotate(
          "text",
          x = mean(match(pair, plot_order)),
          y = star_y,
          label = "***",
          size = 8
        )
    }
    
    ylim_top <- y_max + (0.18 + length(pairs) * 0.08) * y_range
  } else {
    ylim_top <- y_max + 0.10 * y_range
  }
  
  p <- p +
    coord_cartesian(
      ylim = c(y_min, ylim_top)
    )
  
  print(p)
}

dev.off()

cat("Behavioral boxplots saved to:\n")
cat(output_pdf, "\n")


############################ Behavioral Line Plot (Emmeans) ###########
library(emmeans)
library(patchwork)

output_pdf_behav <- file.path(STATS_PATH, paste0("behavioral_lineplot_", TIMESTAMP, ".pdf"))

theme_poster <- function() {
  theme_bw() +
    theme(
      axis.text = element_text(size = 12, color = "black"),
      axis.title.y = element_text(size = 12, color = "black"),
      axis.title.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", linewidth = 0.8),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      legend.title = element_blank(),
      legend.text = element_text(size = 10)
    )
}

cluster_colors <- c(
  "Cluster 1" = "#08519C",
  "Cluster 2" = "#74C476",
  "Cluster 3" = "#238B45",
  "Cluster 4" = "#6BAED6"
)

# T-score features (same scale — plot together)
tscore_features <- c(
  "SRS_restrictive_repetitive_tscore",
  "SRS_social_communication_tscore",
  "SRS_social_cognition_tscore",
  "attention_deficit_hyperactivity_tscore"
)

# Feature line colors and labels
feature_colors <- c(
  "SRS_restrictive_repetitive_tscore"     = "#C0392B",
  "SRS_social_communication_tscore"       = "#E67E22",
  "SRS_social_cognition_tscore"           = "#F1C40F",
  "attention_deficit_hyperactivity_tscore"= "#8E44AD"
)

feature_labels <- c(
  "SRS_restrictive_repetitive_tscore"     = "RRB",
  "SRS_social_communication_tscore"       = "Social Comm.",
  "SRS_social_cognition_tscore"           = "Social Cog.",
  "attention_deficit_hyperactivity_tscore"= "ADHD Traits"
)

# Collect emmeans for all t-score features
emm_all <- data.frame()

for (feat in tscore_features) {
  if (!feat %in% names(df)) next
  
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
  emm$feature <- feat
  emm_all <- rbind(emm_all, emm)
}

emm_all$cluster_label <- paste("Cluster", as.integer(as.character(emm_all$cluster)) + 1L)
emm_all$cluster_label <- factor(emm_all$cluster_label, levels = paste("Cluster", 1:4))
emm_all$feature_label <- feature_labels[emm_all$feature]
emm_all$feature_label <- factor(emm_all$feature_label, levels = feature_labels)

# --- Combined t-score line plot ---
p_tscore <- ggplot(emm_all, aes(x = cluster_label, y = emmean,
                                group = feature_label, color = feature_label)) +
  geom_line(linewidth = 0.9) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                width = 0.12, linewidth = 0.6) +
  geom_point(size = 4, shape = 21, fill = "white", stroke = 1.2) +
  scale_color_manual(values = setNames(feature_colors, feature_labels)) +
  scale_x_discrete(labels = c(
    "Cluster 1" = "Cl. 1", "Cluster 2" = "Cl. 2",
    "Cluster 3" = "Cl. 3", "Cluster 4" = "Cl. 4"
  )) +
  # Reference line at t-score mean
  geom_hline(yintercept = 50, linetype = "dashed",
             color = "grey60", linewidth = 0.5) +
  labs(x = NULL, y = "T-score") +
  theme_poster()

# --- Nonverbal IQ separate panel ---
feat_iq <- "nonverbal_iq"
emm_iq  <- NULL

if (feat_iq %in% names(df)) {
  feat_data_iq <- prepare_feature_data(df, feat_iq, verbose = FALSE)
  feat_data_iq$cluster <- droplevels(factor(feat_data_iq$cluster))
  feat_data_iq$sex <- droplevels(factor(feat_data_iq$sex))
  
  if (reference_cluster %in% levels(feat_data_iq$cluster)) {
    feat_data_iq$cluster <- relevel(feat_data_iq$cluster,
                                    ref = as.character(reference_cluster))
  }
  
  model_iq <- lm(nonverbal_iq ~ cluster + age_at_test + sex, data = feat_data_iq)
  emm_iq   <- as.data.frame(emmeans(model_iq, ~ cluster))
  emm_iq$cluster_label <- paste("Cluster", as.integer(as.character(emm_iq$cluster)) + 1L)
  emm_iq$cluster_label <- factor(emm_iq$cluster_label, levels = paste("Cluster", 1:4))
  
  p_iq <- ggplot(emm_iq, aes(x = cluster_label, y = emmean, group = 1)) +
    geom_line(linewidth = 0.9, color = "grey40") +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                  width = 0.12, linewidth = 0.6, color = "grey40") +
    geom_point(size = 4, shape = 21, fill = "#2C3E50",
               color = "white", stroke = 1.2) +
    scale_x_discrete(labels = c(
      "Cluster 1" = "Cl. 1", "Cluster 2" = "Cl. 2",
      "Cluster 3" = "Cl. 3", "Cluster 4" = "Cl. 4"
    )) +
    labs(x = NULL, y = "Nonverbal IQ") +
    theme_poster() +
    theme(legend.position = "none")
}

# --- Combine ---
pdf(output_pdf_behav, width = 12, height = 3)

if (!is.null(emm_iq)) {
  print(p_tscore + p_iq + plot_layout(widths = c(2.5, 1)))
} else {
  print(p_tscore)
}

dev.off()

cat("Behavioral line plots saved to:\n")
cat(output_pdf_behav, "\n")

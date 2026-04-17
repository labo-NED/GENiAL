# Stats analysis for GENIAL Project
# Based on combination of Q1K and Brain Canada Data
# Input database processed with python
# Author: Emmanuelle C. Nadeau (Nov 2025)

######################################################################
########################## Install packages ##########################
######################################################################
install.packages('readxl')
install.packages("sjmisc")
install.packages('jmv') # ANOVA
install.packages("haven")
install.packages("knitr")
install.packages("e1071")  # For skewness and kurtosis
install.packages("moments")  # Alternative for skewness/kurtosis
install.packages("dplyr")
install.packages("emmeans")
install.packages("ggsignif")

######################################################################
########################## Activate packages #########################
######################################################################
library(readxl)
library(sjmisc)
library(jmv)
library(haven)
library(e1071)
library(moments)
library(dplyr)
library(emmeans)
library(ggsignif)
library(ggplot2)


######################################################################
##################### CTS & MANUAL UPDATES ###########################
######################################################################
TIMESTAMP = 'APR_17_2026'
ROOT_DIR = '/Volumes/LaCie/Q1K-EMMA/' # '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/OUTPUTS/'
PLOTS_PATH = paste(ROOT_DIR, 'Plots/', sep='')
DATABASE_PATH = paste(ROOT_DIR, 'Database/', sep='')
database_filepath = paste(DATABASE_PATH, 'clustered_SOM_Q1K_CHU_MHC_BC_DATA_MAR_09_2026.csv', sep='')
isROIAnalysis = FALSE
analysis_label = if(isROIAnalysis) "ROI" else "GLOBAL"

base_eeg_features <- c('hurst_2s',
                       'pow_per_delta_2s', 'pow_per_theta_2s', 'pow_per_alpha_2s', 'pow_per_beta_2s', 'pow_per_gamma_2s', 'pow_per_low_gamma_2s', 'pow_per_high_gamma_2s',
                       'fooof_exponent_2s','fooof_offset_2s',
                       'higuchi_fd_5s', 'katz_fd_5s',
                       'samp_entropy_5s', 'CI_5s', 'CI_lowscale_5s', 'CI_highscale_5s')
base_eeg_2s_features <- c('hurst_2s', 
                          'fooof_offset_2s','fooof_exponent_2s',
                          'pow_delta_2s', 'pow_theta_2s', 'pow_alpha_2s', 'pow_beta_2s', 'pow_gamma_2s', 'pow_low_gamma_2s', 'pow_high_gamma_2s',
                          'pow_per_delta_2s', 'pow_per_theta_2s', 'pow_per_alpha_2s', 'pow_per_gamma_2s', 'pow_per_low_gamma_2s', 'pow_per_high_gamma_2s')

absolute_band_power_features <- c('pow_delta_2s', 'pow_theta_2s', 'pow_alpha_2s', 'pow_beta_2s', 'pow_gamma_2s', 'pow_low_gamma_2s', 'pow_high_gamma_2s')

# only 2s features
base_eeg_features <- base_eeg_2s_features


######################################################################
########################## Import dataset ############################
######################################################################
db <- read.csv(database_filepath)
db_copy <- db

# --- Update EEG Feature list (ROI or Global) --- #
if (isROIAnalysis){
  suffixes <- sub("^[^_]+_", "", base_eeg_features)
  suffixes
  eeg_features <- unlist(lapply(suffixes, function(suf) {
    grep(paste0("_", suf, "$"), names(db_copy), value = TRUE)
  }))
} else {
  eeg_features <- base_eeg_features
}

# ---- Basic checks ---- #
str(db_copy)
summary(db_copy)

table(db_copy$cluster) # Nb of participant / cluster

sapply(db_copy[eeg_features], function(x) sum(is.na(x))) # Check missing data for EEG features

## --- NOTE --- #
# Need to fetch EEG from orange drive again.
# About 30-40 missing EEGs

# Check covariate balance
db_copy |>
  group_by(cluster) |>
  summarise(
    n = n(),
    mean_age = mean(age_at_test, na.rm = TRUE),
    prop_male = mean(sex == "M", na.rm = TRUE)
  )



######################################################################
########################### Clean Up  ################################
######################################################################
# ---- Log transform power band features ----#
for (feat in absolute_band_power_features) {
  if (feat %in% names(db_copy)) {
    newname <- paste0("log_", feat)
    # Only log-transform positive values to avoid NaNs from zeros/negatives
    source_vals <- db_copy[[feat]]
    db_copy[[newname]] <- NA_real_
    valid_idx <- !is.na(source_vals) & source_vals > 0
    db_copy[[newname]][valid_idx] <- log(source_vals[valid_idx])
  }
}

# Update eeg_features list with new log features and remove original band power features
eeg_features <- c(eeg_features, paste0("log_", absolute_band_power_features))
eeg_features <- setdiff(eeg_features, absolute_band_power_features)

# ---- Recode ethnicities into new categories ---- #
db_copy$ethnicity_recoded <- dplyr::case_when(
  ## White / European
  db_copy$ethnicities %in% c(
    "White_Caucasian", "Caucasian", "Caucasien", "Caucasienne",
    "Caucasien/latin", "Caucasien/Latino",
    "CF", "French Canadian", "French Canadien", "French Candian",
    "Canadian Francais", "France/CF", "France/Qc", "Francaise",
    "CF/Espagne", "CF/portugal", "CF/Portugual",
    "Portugal/France", "Portugais", "Portugal",
    "Roumanie/Espagne", "Roumain/Européen Caucasien",
    "Italie/CF", "CF/Italie", "Grec/CF",
    "French Canadian/European", "Caucasian/Européen", "Caucasien/Européen",
    "Caucasian/European"
  ) ~ "White/European",
  
  ## Black / African / Caribbean
  db_copy$ethnicities %in% c(
    "Black", "Africain", "Africaine",
    "Haiti", "Haitien", "Haitienne",
    "Haitien/Canadian", "Haiti/Canadien",
    "Haïtien/CF", "Haïtienne/CF",
    "Caraibes", "Caribbean",
    "Ile Marice", "Ile Maurice"
  ) ~ "Black/African/Caribbean",
  
  ## Indigenous / First Nations
  db_copy$ethnicities %in% c(
    "Indigenous", "Indigenous, White_Caucasian",
    "Amérindien/CF", "Amérindienne", "Amérindien"
  ) ~ "Indigenous / First Nations",
  
  ## Latin American
  db_copy$ethnicities %in% c(
    "Latin_American",
    "Latin_American, Southeast_Asian, White_Caucasian",
    "Latino", "latino", "Latine",
    "Hispanic",
    "Chili", "Chili/Colombie",
    "Cuba/Canada", "Cuba/CF",
    "Colombia",
    "El Salvador", "Salvador/Chili",
    "Amerique Latine (Salvador)"
  ) ~ "Latin American",
  
  ## Asian (East, South, Southeast)
  db_copy$ethnicities %in% c(
    "Chine", "Chinese", "Chinoise",
    "Japanese", "Japanese", "Japonais",
    "South_Asian",
    "Filipino",
    "Southeast_Asian",
    "Viet Nam", "Vietnam",
    "Asian"
  ) ~ "Asian",
  
  ## Middle Eastern / West Asian (incl. North Africa in this group)
  db_copy$ethnicities %in% c(
    "Algérie", "Algerian",
    "Maroc", "MAroc",
    "Arab",
    "Moyen-Orient", "Moyen Orient",
    "Middle East",
    "West_Asian",
    "Nord-Africaine",
    "Liban", "Liban/Italie", "Liban/France", "USA/Liban", "Liban/USA"
  ) ~ "Middle Eastern / West Asian",
  
  ## Explicit mixed / "Other" codes
  db_copy$ethnicities %in% c(
    "Other_ethnicity",
    "White_Caucasian, Other_ethnicity",
    "Other_ethnicity, White_Caucasian",
    "Arab, White_Caucasian",
    "Black, White_Caucasian",
    "Latin_American, Southeast_Asian, White_Caucasian",
    "Indigenous, White_Caucasian",
    "Arab, Other_ethnicity",
    "CF/Haiti",
    "Caucasien/Middle Eastern",
    "Caucasien/latin", "Caucasien/Latino",
    "Latino-Moyen Orient",
    "Caucasien/Middle Eastern"
  ) ~ "Other / Mixed",
  
  ## Unknown / None
  db_copy$ethnicities %in% c("Unknown", "None") ~ "Unknown",
  
  ## Truly missing
  is.na(db_copy$ethnicities) | db_copy$ethnicities == "" ~ NA_character_,
  
  ## Fallback: keep original label (so you can see what was not recoded)
  TRUE ~ db_copy$ethnicities
)

db_copy$ethnicity_recoded <- factor(db_copy$ethnicity_recoded)

# save updated database
write.csv(db_copy, file.path(paste0(DATABASE_PATH, "merged_clustered_EEG_features_global_RSRio_", TIMESTAMP, "_logtransformed.csv")))



######################################################################
################## Normality & Distribution ##########################
######################################################################
columns_to_plot <- c("age_at_test","nonverbal_iq", "SRS_restrictive_repetitive_tscore",
                      "SRS_social_communication_tscore", "SRS_social_cognition_tscore", 
                      "attention_deficit_hyperactivity_tscore")


# --- DISTRIBUTION --- #
# Function to create scatter plot with mean - check for outliers
get_plot_range <- function(db, colname) {
  variable <- db[[colname]][!is.na(db[[colname]])] # Removing NA values from plots
  avg <- mean(variable)
  
  plot(variable, 
       main = paste("Nuage de points pour ", colname), 
       xlab = "Id", 
       ylab = "Values", 
       pch = 20, 
       col = "blue")
  
  abline(h = avg, col = "red", lty = 2) # Avg
  legend("topleft", legend = "Moyenne", col = "red", lty = 2)
}

pdf(file.path(paste0(PLOTS_PATH, "scatter_plots_", analysis_label, "_", TIMESTAMP, ".pdf")))
for (col in columns_to_plot) {
  if (col %in% names(db_copy)) {
    get_plot_range(db_copy, col)
  }
}
dev.off()

# ---- NORMALITY ----#
# Initialize results dataframe
normality_results <- data.frame(
  Variable = character(),
  N = integer(),
  Shapiro_W = numeric(),
  Shapiro_p = numeric(),
  Skewness = numeric(),
  Kurtosis = numeric(),
  Normal = character(),
  stringsAsFactors = FALSE
)

# Create PDF for histograms with normal curve overlay
pdf(file.path(paste0(PLOTS_PATH, "histograms_", analysis_label, "_", TIMESTAMP, ".pdf")), width = 10, height = 7)
par(mfrow = c(2, 3))  # 2 rows, 3 columns layout

for (var in columns_to_plot) {
  if (var %in% names(db_copy)) {
    # Remove NA values
    data <- db_copy[[var]][!is.na(db_copy[[var]])]
    n <- length(data)
    
    # Normality tests
    if (n > 3 && n <= 5000) {
      shapiro_test <- shapiro.test(data)
      shapiro_w <- shapiro_test$statistic
      shapiro_p <- shapiro_test$p.value
    } else {
      shapiro_w <- NA
      shapiro_p <- NA
    }
    
    # Skewness and Kurtosis
    skew <- skewness(data)
    kurt <- kurtosis(data) - 3  # Excess kurtosis (to match scipy)
    
    # Determine if normal
    is_normal <- ifelse(is.na(shapiro_p), "Unknown", ifelse(shapiro_p > 0.05, "Yes", "No"))
    
    # Store results
    normality_results <- rbind(normality_results, data.frame(
      Variable = var,
      N = n,
      Shapiro_W = shapiro_w,
      Shapiro_p = shapiro_p,
      Skewness = skew,
      Kurtosis = kurt,
      Normal = is_normal
    ))
    
    # Create histogram with density
    hist_obj <- hist(data, breaks = 30, plot = FALSE)
    hist(data, breaks = 30, freq = FALSE, 
         col = "lightblue", border = "black",
         main = sprintf("%s\nN=%d, Skew=%.2f, Kurt=%.2f\np=%.4f", 
                       var, n, skew, kurt, ifelse(is.na(shapiro_p), 0, shapiro_p)),
         xlab = var, ylab = "Density", cex.main = 0.9)
    
    # Overlay normal distribution
    mu <- mean(data)
    sigma <- sd(data)
    x_seq <- seq(min(data), max(data), length.out = 100)
    lines(x_seq, dnorm(x_seq, mean = mu, sd = sigma), col = "red", lwd = 2)
    
    # Overlay KDE
    dens <- density(data)
    lines(dens, col = "darkgreen", lwd = 2)
    
    # Add legend
    legend("topright", legend = c("Normal fit", "KDE"), 
           col = c("red", "darkgreen"), lwd = 2, cex = 0.7)
    
    # Print to console
    cat(sprintf("\n%s:\n", var))
    if (!is.na(shapiro_p)) {
      cat(sprintf("  Shapiro-Wilk: W=%.4f, p=%.4f\n", shapiro_w, shapiro_p))
    } else {
      cat("  Shapiro-Wilk: N/A (sample size issue)\n")
    }
    cat(sprintf("  Skewness: %.4f, Kurtosis: %.4f\n", skew, kurt))
    cat(sprintf("  Distribution: %s\n", ifelse(is_normal == "Yes", "NORMAL", "NON-NORMAL")))
  }
}

dev.off()
par(mfrow = c(1, 1))  # Reset layout

# Save normality results to CSV
write.csv(normality_results, 
          file.path(paste0(PLOTS_PATH, "normality_tests_", analysis_label, "_", TIMESTAMP, ".csv")),
          row.names = FALSE)


#################################################################
############################ LM #################################
#################################################################
df <- db_copy
df$cluster <- factor(df$cluster, levels = c(0, 1, 2, 3))
df$sex <- factor(df$sex)
covariates <- c("age_at_test", "sex")

# --- Choose reference cluster --- #
sink(file.path(paste0(PLOTS_PATH, "behavioral_scores_table_", TIMESTAMP, ".txt")))

# Compute average values by cluster for reference selection
cat("\n========================================\n")
cat("AVERAGE VALUES BY CLUSTER\n")
cat("(for reference cluster selection)\n")
cat("========================================\n\n")

cluster_means <- db_copy |>
  group_by(cluster) |>
  summarise(
    n = n(),
    # SRS variables
    SRS_restrictive_repetitive_mean = mean(SRS_restrictive_repetitive_tscore, na.rm = TRUE),
    SRS_restrictive_repetitive_sd = sd(SRS_restrictive_repetitive_tscore, na.rm = TRUE),
    SRS_social_communication_mean = mean(SRS_social_communication_tscore, na.rm = TRUE),
    SRS_social_communication_sd = sd(SRS_social_communication_tscore, na.rm = TRUE),
    SRS_social_cognition_mean = mean(SRS_social_cognition_tscore, na.rm = TRUE),
    SRS_social_cognition_sd = sd(SRS_social_cognition_tscore, na.rm = TRUE),
    # ADHD tscore
    ADHD_tscore_mean = mean(attention_deficit_hyperactivity_tscore, na.rm = TRUE),
    ADHD_tscore_sd = sd(attention_deficit_hyperactivity_tscore, na.rm = TRUE),
    # NVIQ
    NVIQ_mean = mean(nonverbal_iq, na.rm = TRUE),
    NVIQ_sd = sd(nonverbal_iq, na.rm = TRUE),
    .groups = 'drop'
  )

# Print formatted table
print(cluster_means)

# Also print in a more readable format
cat("\n--- Summary by Cluster ---\n")
for (i in seq_len(nrow(cluster_means))) {
  cat(sprintf("\nCluster %d (n = %d):\n", cluster_means$cluster[i], cluster_means$n[i]))
  cat(sprintf("  SRS Restrictive/Repetitive: %.2f (SD = %.2f)\n", 
              cluster_means$SRS_restrictive_repetitive_mean[i], 
              cluster_means$SRS_restrictive_repetitive_sd[i]))
  cat(sprintf("  SRS Social Communication:   %.2f (SD = %.2f)\n", 
              cluster_means$SRS_social_communication_mean[i], 
              cluster_means$SRS_social_communication_sd[i]))
  cat(sprintf("  SRS Social Cognition:      %.2f (SD = %.2f)\n", 
              cluster_means$SRS_social_cognition_mean[i], 
              cluster_means$SRS_social_cognition_sd[i]))
  cat(sprintf("  ADHD T-score:               %.2f (SD = %.2f)\n", 
              cluster_means$ADHD_tscore_mean[i], 
              cluster_means$ADHD_tscore_sd[i]))
  cat(sprintf("  NVIQ:                      %.2f (SD = %.2f)\n", 
              cluster_means$NVIQ_mean[i], 
              cluster_means$NVIQ_sd[i]))
}
cat("\n========================================\n\n")
sink()

# Set reference cluster for LM models
reference_cluster <- 3 # Most severe group

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

# ---- LM for each EEG feature ----#
sink(file.path(paste0(PLOTS_PATH, "EEG_features_statsmodel_summaries_", analysis_label, "_", TIMESTAMP, ".txt")))
results <- lapply(eeg_features, function(feat) {
  cat("\n======================\n")
  cat("Feature:", feat, "\n")
  cat("======================\n\n")
  
  dat_feat <- prepare_feature_data(df, feat, verbose = TRUE)
  dat_feat$cluster <- droplevels(dat_feat$cluster)
  dat_feat$sex     <- droplevels(dat_feat$sex)
  
  # If cluster has fewer than 2 levels, we cannot fit the full model
  if (nlevels(dat_feat$cluster) < 2) {
    cat("Skipped: cluster has fewer than 2 levels after NA removal\n")
    return(data.frame(feature = feat, p_value = NA_real_))
  }
  
  # Set reference cluster (only if the reference cluster exists in the data)
  if (reference_cluster %in% levels(dat_feat$cluster)) {
    dat_feat$cluster <- relevel(dat_feat$cluster, ref = as.character(reference_cluster))
    cat("Reference cluster set to:", reference_cluster, "\n")
  } else {
    cat("Warning: Reference cluster", reference_cluster, "not present in data, using default reference\n")
  }
  
  # If sex has only 1 level, drop sex from the model for this feature
  if (nlevels(dat_feat$sex) < 2) {
    cat("Warning: sex has only 1 level after NA removal, fitting model without sex\n")
    form <- as.formula(paste(feat, "~ cluster + age_at_test"))
  } else {
    form <- as.formula(paste(feat, "~ cluster + age_at_test + sex"))
  }
  
  # Fit LM
  fit <- lm(form, data = dat_feat)
  print(summary(fit))
  
})
sink()

# ---- LM for each behavioral measure (same model as EEG: cluster vs ref + age + sex) ----#
behavioral_features <- c(
  "SRS_restrictive_repetitive_tscore",
  "SRS_social_communication_tscore",
  "SRS_social_cognition_tscore",
  "attention_deficit_hyperactivity_tscore",
  "nonverbal_iq"
)
behavioral_features <- behavioral_features[behavioral_features %in% names(df)]

sink(file.path(paste0(PLOTS_PATH, "behavioral_measures_statsmodel_summaries_", analysis_label, "_", TIMESTAMP, ".txt")))
invisible(lapply(behavioral_features, function(feat) {
  cat("\n======================\n")
  cat("Behavioral measure:", feat, "\n")
  cat("======================\n\n")

  dat_feat <- prepare_feature_data(df, feat, verbose = TRUE)
  dat_feat$cluster <- droplevels(dat_feat$cluster)
  dat_feat$sex     <- droplevels(dat_feat$sex)

  if (nlevels(dat_feat$cluster) < 2) {
    cat("Skipped: cluster has fewer than 2 levels after NA removal\n")
    return(invisible(NULL))
  }

  if (reference_cluster %in% levels(dat_feat$cluster)) {
    dat_feat$cluster <- relevel(dat_feat$cluster, ref = as.character(reference_cluster))
    cat("Reference cluster set to:", reference_cluster, "\n")
  } else {
    cat("Warning: Reference cluster", reference_cluster, "not present in data, using default reference\n")
  }

  if (nlevels(dat_feat$sex) < 2) {
    cat("Warning: sex has only 1 level after NA removal, fitting model without sex\n")
    form <- as.formula(paste(feat, "~ cluster + age_at_test"))
  } else {
    form <- as.formula(paste(feat, "~ cluster + age_at_test + sex"))
  }

  fit <- lm(form, data = dat_feat)
  print(summary(fit))

  invisible(NULL)
}))
sink()


##################################################################
################## Pairwise comparisons ##########################
##################################################################

# ---- BOX PLOTS ---- #
out_dir <- PLOTS_PATH

# Open sink file for emmeans tables
sink(file.path(out_dir,
               paste0("emmeans_tables_all_features_", analysis_label, "_", TIMESTAMP, ".txt")))

pdf(file.path(out_dir,
              paste0("boxplots_all_features_with_stars_", analysis_label, "_", TIMESTAMP, ".pdf")),
    width = 7, height = 5)

for (feat in eeg_features) {
  
  dat_feat <- prepare_feature_data(df, feat, verbose = FALSE)
  dat_feat$cluster <- droplevels(factor(dat_feat$cluster))
  dat_feat$sex     <- droplevels(factor(dat_feat$sex))
  
  if (nrow(dat_feat) == 0 || nlevels(dat_feat$cluster) < 2) next
  
  # Set reference cluster (only if the reference cluster exists in the data)
  if (reference_cluster %in% levels(dat_feat$cluster)) {
    dat_feat$cluster <- relevel(dat_feat$cluster, ref = as.character(reference_cluster))
  }
  
  if (nlevels(dat_feat$sex) < 2) {
    form <- as.formula(paste(feat, "~ cluster + age_at_test"))
  } else {
    form <- as.formula(paste(feat, "~ cluster + age_at_test + sex"))
  }
  
  fit <- lm(form, data = dat_feat)
  
  emm   <- emmeans::emmeans(fit, "cluster")
  
  # Print emmeans table for this feature
  cat("\n======================\n")
  cat("Feature:", feat, "\n")
  cat("======================\n\n")
  cat("Estimated Marginal Means:\n")
  print(emm)
  cat("\n")
  
  pw    <- pairs(emm, adjust = "none")
  pw_df <- as.data.frame(pw)
  
  # Add significance stars to pairwise comparisons
  pw_df$stars <- sapply(pw_df$p.value, function(pv) {
    if (is.na(pv)) ""
    else if (pv < 0.001) "***"
    else if (pv < 0.01) "**"
    else if (pv < 0.05) "*"
    else ""
  })
  
  # Print pairwise comparisons with p-values and stars
  cat("Pairwise Comparisons:\n")
  print(pw_df[, c("contrast", "estimate", "SE", "df", "t.ratio", "p.value", "stars")], row.names = FALSE)
  cat("\n")
  
  sig_pw <- pw_df %>% dplyr::filter(p.value < 0.05)
  
  n_total <- nrow(dat_feat)
  
  y_max   <- max(dat_feat[[feat]], na.rm = TRUE)
  y_min   <- min(dat_feat[[feat]], na.rm = TRUE)
  y_range <- y_max - y_min
  if (y_range == 0) y_range <- 1
  
  # Add simple buffer above y-axis (30% of range)
  y_upper_limit <- y_max + y_range * 0.30
  
  p <- ggplot(dat_feat, aes(x = factor(cluster), y = .data[[feat]])) +
    geom_boxplot(outlier.shape = NA, alpha = 0.6) +
    geom_point(
      position = position_jitter(width = 0.15, height = 0),
      alpha    = 0.5,
      size     = 1.5,
      color    = "grey40"
    ) +
    stat_summary(
      fun   = mean,
      geom  = "point",
      shape = 23,
      size  = 2,
      fill  = "red",
      color = "red"
    ) +
    annotate(
      "text",
      x = -Inf, y = Inf,
      label = paste0("N = ", n_total),
      hjust = -0.1, vjust = 1.3,
      size = 3
    ) +
    labs(
      title = paste0("Feature: ", feat),
      x = "Cluster",
      y = feat
    ) +
    theme_bw() +
    coord_cartesian(ylim = c(y_min - y_range * 0.05, y_upper_limit))
  
  if (nrow(sig_pw) > 0) {
    parts  <- strsplit(as.character(sig_pw$contrast), " - ")
    g1_raw <- sapply(parts, `[`, 1)
    g2_raw <- sapply(parts, `[`, 2)
    
    g1_lab <- sub("^cluster", "", g1_raw)
    g2_lab <- sub("^cluster", "", g2_raw)
    
    y_base <- y_max + y_range * 0.05
    y_step <- y_range * 0.06
    y_vals <- y_base + (seq_along(g1_lab) - 1) * y_step
    
    stars <- sapply(sig_pw$p.value, function(pv) {
      if (pv < 0.001) "***"
      else if (pv < 0.01) "**"
      else "*"
    })
    
    ann_df <- data.frame(
      xmin        = g1_lab,
      xmax        = g2_lab,
      y_position  = y_vals,
      annotations = stars
    )
    
    p <- p +
      ggsignif::geom_signif(
        data   = ann_df,
        manual = TRUE,
        aes(xmin = xmin,
            xmax = xmax,
            annotations = annotations,
            y_position = y_position),
        tip_length = 0.01,
        textsize   = 4
      )
  }
  
  print(p)
}

dev.off()

# Close sink file for emmeans tables
sink()



######################################################################
####################### INTER-FEATURE CORRELATION ####################
######################################################################

cat("\n\n######################################################################\n")
cat("#################### EEG Feature Inter-Correlation ###################\n")
cat("######################################################################\n\n")

# Only these columns enter the inter-feature correlation (edit here; independent of eeg_features)
corr_features <- c(
  "hurst_2s",
  "pow_per_delta_2s",
  "pow_per_low_gamma_2s",
  "pow_per_high_gamma_2s",
  "higuchi_fd_5s"
)
corr_features <- corr_features[corr_features %in% names(db_copy)]

eeg_data_for_corr <- na.omit(db_copy[, corr_features, drop = FALSE])

cat(sprintf("Number of participants for correlation analysis: %d\n", nrow(eeg_data_for_corr)))

# Calculate the Pearson correlation matrix
correlation_matrix <- cor(eeg_data_for_corr, method = "pearson")

# Print the full correlation matrix
cat("\n--- Full Pearson Correlation Matrix (r values) ---\n")
print(round(correlation_matrix, 3))

# Optionally, save the correlation matrix to a CSV file
write.csv(round(correlation_matrix, 3),
          file.path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/OUTPUTS/Stats/", paste0("eeg_feature_correlation_matrix_", analysis_label, "_", TIMESTAMP, ".csv")),
          row.names = TRUE)

# --- Visualization: Correlogram (using 'corrplot' for better visualization) --- #
# Note: You may need to install the 'corrplot' package if you haven't already
# install.packages('corrplot')
library(corrplot)

pdf(file.path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/OUTPUTS/Stats/", paste0("correlogram_", analysis_label, "_", TIMESTAMP, ".pdf")),
    width = 10, height = 10)

corrplot(correlation_matrix,
         method = "circle", # Use colored circles to represent correlation magnitude
         type = "upper",    # Only show the upper triangle (matrix is symmetric)
         order = "hclust",  # Order variables by hierarchical clustering (groups similar features)
         tl.col = "black",  # Color of text labels
         tl.srt = 45,       # Text label rotation
         p.mat = cor.mtest(eeg_data_for_corr)$p, # Pass p-values for significance
         insig = "label_sig", # Add stars for significant correlations
         sig.level = c(0.001, 0.01, 0.05), # Significance levels for stars
         pch.cex = 0.9,
         diag = FALSE) # Do not show the diagonal

dev.off()
cat("\nCorrelogram saved to PDF: correlogram_", analysis_label, "_", TIMESTAMP, ".pdf\n")
cat("Correlation Matrix saved to CSV.\n")

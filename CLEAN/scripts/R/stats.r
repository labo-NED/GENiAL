########################## Install packages ##########################
install.packages('readxl')
install.packages("sjmisc")
install.packages('jmv') # ANOVA
install.packages("haven")
install.packages("knitr")

########################## Activate packages #########################
library(readxl)
library(sjmisc)
library(jmv)
library(haven)

########################## Import dataset ############################
# CLUSTERS WITH SOM + EEG features
original_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/merged_clustered_EEG_features_RSRio_NOV_24_2025.csv")

og_dataset_copy <- original_dataset # Make a copy for preprocessing

########################### Preprocessing ############################

#### Plausibilité des scores ###
# Check frequencies
sink(file="/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Clustered/RAW_freq_output.txt")
lapply(og_dataset_copy, sjmisc::frq)
sink()
sink(NULL) # Disable active sink

################### Convert to numeric/factor ########################

# Check classes (numeric, factor, or heaven_labelled)
lapply(og_dataset_copy, class) #No heaven labelled, can proceed.

### Recode highest_education_level into new categories
og_dataset_copy$highest_education_level_recoded <- dplyr::case_when(
  og_dataset_copy$highest_education_level %in% c("Master", "Master's degree", "Doctorate") ~ "Graduate studies",
  og_dataset_copy$highest_education_level %in% c("Bachelor", "Bachelor's degree") ~ "Bachelor",
  og_dataset_copy$highest_education_level %in% c(
    "College, CEGEP or other non-university certificate or diploma",
    "Apprenticeship or other trades certificate or diploma",
    "Apprenticeship, vocational education or training") ~ "College_and_certificates",
  og_dataset_copy$highest_education_level %in% c(
    "Completed mandatory school (15 years old)",
    "Completed high school or baccalaureate (18 years old)",
    "High school diploma or certificate") ~ "high_school",
  og_dataset_copy$highest_education_level %in% c(
    "Non-applicable (Child still in school)",
    "Other",
    "Some high school", 
    "Special education",
    "Elementary school or less") ~ "other",
  is.na(og_dataset_copy$highest_education_level) | og_dataset_copy$highest_education_level == "" ~ NA_character_,
  TRUE ~ as.character(og_dataset_copy$highest_education_level)
)
og_dataset_copy$highest_education_level_recoded <- factor(og_dataset_copy$highest_education_level_recoded)

# Check frequencies
sink(file="/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Outputs/Clustered/RAW_freq_output.txt")
lapply(og_dataset_copy, sjmisc::frq)
sink()
sink(NULL) # Disable active sink

### Recode family_ethnicity into new categories
og_dataset_copy$family_ethnicity_recoded <- dplyr::case_when(
  og_dataset_copy$family_ethnicity %in% c("White_Caucasian", "Caucasian", "Caucasien", "Caucasienne",
                                         "Caucasien/latin", "Caucasien/Latino",
                                         "CF", "French Canadian", "French Canadien", "France/CF",
                                         "CF/Espagne", "CF/portugal", "Portugal/France", 
                                         "Roumanie/Espagne", "Italie/CF", "Italie/Mexique", "Grec/CF",
                                         "French Canadian/European", "Portugais") ~ "White/European",
  og_dataset_copy$family_ethnicity %in% c("Africain", "Black", "Haiti", "Ile Marice") ~ "Black/African/Caribbean",
  og_dataset_copy$family_ethnicity %in% c("Amérindien/CF", "Indigenous, White_Caucasian") ~ "Indigenous / First Nations",
  og_dataset_copy$family_ethnicity %in% c("Brésil/CF", "Latin_American", "Latin_American, Southeast_Asian, White_Caucasian") ~ "Latin American",
  og_dataset_copy$family_ethnicity %in% c("Chine", "Chinese", "Southeast_Asian") ~ "Asian",
  og_dataset_copy$family_ethnicity %in% c("Algérie", "Arab", "Moyen-Orient", "West_Asian") ~ "Middle Eastern / West Asian",
  og_dataset_copy$family_ethnicity %in% c("Other_ethnicity", "White_Caucasian, Other_ethnicity") ~ "Other / Mixed",
  og_dataset_copy$family_ethnicity %in% c("Unknown", "") ~ "Unknown",
  is.na(og_dataset_copy$family_ethnicity) | og_dataset_copy$family_ethnicity == "" ~ NA_character_,
  TRUE ~ og_dataset_copy$family_ethnicity
)
og_dataset_copy$family_ethnicity_recoded <- factor(og_dataset_copy$family_ethnicity_recoded)

# Check frequencies
sink(file="RAW_freq_output.txt")
lapply(og_dataset_copy, sjmisc::frq)
sink()
sink(NULL) # Disable active sink

### Recode sex to factor
og_dataset_copy$sex <- factor(og_dataset_copy$sex)

# Save
write.csv(og_dataset_copy, file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Clustered/FINAL_DATABASE_USED_IN_R_SOM.csv", row.names = FALSE)

################### Basic Stats ########################

# Check frequencies of categorical variables
categorical_variables <- c("sex", "family_ethnicity_recoded", "highest_education_level_recoded")

sink(file="categorical_freq_output.txt")
lapply(og_dataset_copy[categorical_variables], sjmisc::frq)
sink()
sink(NULL) # Disable active sink

# Analysis - Sex
sjmisc::frq(og_dataset_copy$sex)

# Analysis - family_ethnicity_recoded
sjmisc::frq(og_dataset_copy$family_ethnicity_recoded)

# Analysis - highest_education_level_recoded
sjmisc::frq(og_dataset_copy$highest_education_level_recoded)

################# Variability/range in numerical data ######################

# Age
min_age <- min(og_dataset_copy$age_at_test, na.rm = TRUE) # 5
max_age <- max(og_dataset_copy$age_at_test, na.rm = TRUE) # 18

# IQ
min_iq <- min(og_dataset_copy$nonverbal_iq, na.rm = TRUE) # 30
max_iq <- max(og_dataset_copy$nonverbal_iq, na.rm = TRUE) # 132

# Behavioral Scores

# SRS Social Cognition T-score
min_srs_social_cognition <- min(og_dataset_copy$SRS_social_cognition_tscore, na.rm = TRUE) # 39
max_srs_social_cognition <- max(og_dataset_copy$SRS_social_cognition_tscore, na.rm = TRUE) # 90

# SRS Social Communication T-score
min_srs_social_communication <- min(og_dataset_copy$SRS_social_communication_tscore, na.rm = TRUE)
max_srs_social_communication <- max(og_dataset_copy$SRS_social_communication_tscore, na.rm = TRUE)

# SRS Restrictive Repetitive T-score
min_srs_restrictive_repetitive <- min(og_dataset_copy$SRS_restrictive_repetitive_tscore, na.rm = TRUE)
max_srs_restrictive_repetitive <- max(og_dataset_copy$SRS_restrictive_repetitive_tscore, na.rm = TRUE)

# ADHD T-score
min_adhd <- min(og_dataset_copy$attention_deficit_hyperactivity_tscore, na.rm = TRUE)
max_adhd <- max(og_dataset_copy$attention_deficit_hyperactivity_tscore, na.rm = TRUE)

### Normality

# Plot data to help visualize
columns_to_plot <- c("age_at_test", "nonverbal_iq", "SRS_restrictive_repetitive_tscore","SRS_social_communication_tscore", "SRS_social_cognition_tscore", "attention_deficit_hyperactivity_tscore")

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

pdf("scatter_plots.pdf")
for (col in columns_to_plot) {
  if (col %in% names(og_dataset_copy)) {
    get_plot_range(og_dataset_copy, col)
  }
}
dev.off()

# Normality - histograms
# Histograms

# Function to plot histograms for multiple variables
plot_histograms <- function(data, var_list, filename) {
  # Open the PDF file to save the plots
  pdf(filename)
  
  # Loop through each variable in the list
  for (var in var_list) {
    # Get the maximum count from the histogram data
    hist_data <- hist(data[[var]], plot = FALSE, breaks = 20)
    max_count <- max(hist_data$counts) * 1.1 # setting an automatic y limit with 10% buffer
    
    # Create the histogram
    hist(data[[var]], 
         main = paste("Distribution de ", var), 
         xlab = paste("Valeurs de ", var), 
         ylab = "Nombre de participants", 
         col = "beige", 
         border = "black", 
         xlim = c(min(data[[var]], na.rm = TRUE), max(data[[var]], na.rm = TRUE)), 
         ylim = c(0, max_count),
         breaks = 20)
  }
  
  # Close the PDF file
  dev.off()
}

# Call the function to save histograms for the variables
plot_histograms(og_dataset_copy, columns_to_plot, "histograms.pdf")


# Descriptives
sink(file = "descriptives.txt")
jmv::descriptives(og_dataset_copy,
                  vars = vars("age_at_test", "nonverbal_iq", "SRS_restrictive_repetitive_tscore","SRS_social_communication_tscore", "SRS_social_cognition_tscore", "attention_deficit_hyperactivity_tscore"),
                  sd = TRUE, range = TRUE, 
                  skew = TRUE, kurt = TRUE)
sink()
sink(NULL)

#############################
########## MANCOVA ##########
#############################

# --- Setup
df <- og_dataset_copy
df$cluster <- factor(df$cluster, levels = c(0, 1, 2, 3))

DVS <- c(
  "SRS_restrictive_repetitive_tscore",
  "SRS_social_communication_tscore",
  "SRS_social_cognition_tscore",
  "attention_deficit_hyperactivity_tscore",
  "nonverbal_iq"
)

# --- MANCOVA (multivariate test)
mancova_model <- manova(cbind(
  SRS_restrictive_repetitive_tscore,
  SRS_social_communication_tscore,
  SRS_social_cognition_tscore,
  attention_deficit_hyperactivity_tscore,
  nonverbal_iq
) ~ cluster + age_at_test + sex, data = df)

print(summary(mancova_model, test = "Pillai"))  # robust
print(summary(mancova_model, test = "Wilks"))   # traditional

# --- Univariate ANCOVAs per DV (safer & clearer for effect sizes)
# install.packages("effectsize"); install.packages("emmeans")  # if needed
library(effectsize)
library(emmeans)

fit_one <- function(y) {
  f <- as.formula(paste(y, "~ cluster + age_at_test + nonverbal_iq + sex"))
  lm(f, data = df)
}

fits <- lapply(DVS, fit_one)
names(fits) <- DVS

# Extract the cluster p-value, partial eta^2, omega^2, and N per DV
extract_stats <- function(fit) {
  aov_tab <- anova(fit)
  # row name may be "cluster" (Type I); if you prefer Type III, see note below
  cl_row <- which(rownames(aov_tab) == "cluster")
  p      <- if (length(cl_row)) aov_tab$`Pr(>F)`[cl_row] else NA_real_
  
  eta2   <- eta_squared(fit, partial = TRUE, ci = NULL)     # data frame
  eta2_c <- subset(eta2, grepl("^cluster$", Parameter))$Eta2_partial
  if (length(eta2_c) == 0) eta2_c <- NA_real_
  
  # omega^2 partial per effect (effectsize computes ω² from lm/aov)
  # Note: for small samples ω² can be slightly negative; often truncated at 0 in reporting.
  w2     <- omega_squared(fit, partial = TRUE, ci = NULL)
  w2_c   <- subset(w2, grepl("^cluster$", Parameter))$Omega2_partial
  if (length(w2_c) == 0) w2_c <- NA_real_
  
  c(n = stats::nobs(fit), p = p, eta2_partial = eta2_c, omega2_partial = w2_c)
}

tab <- as.data.frame(t(sapply(fits, extract_stats)))
tab$p_bonf <- p.adjust(tab$p, method = "bonferroni")
tab$p_fdr  <- p.adjust(tab$p, method = "BH")
tab


## Adjusted means by cluster
library(emmeans)

DVS <- c(
  "SRS_restrictive_repetitive_tscore",
  "SRS_social_communication_tscore",
  "SRS_social_cognition_tscore",
  "attention_deficit_hyperactivity_tscore"
)

get_emm <- function(y) {
  f <- as.formula(paste(y, "~ cluster + age + IQ + sex"))
  fit <- lm(f, data = df)
  summary(emmeans(fit, ~ cluster))   # adjusted means
}

emm_results <- lapply(DVS, get_emm)
names(emm_results) <- DVS

# See results
print(emm_results)
print(emm_results$SRS_restrictive_repetitive_tscore)
print(emm_results$SRS_social_communication_tscore)
print(emm_results$SRS_social_cognition_tscore)
print(emm_results$attention_deficit_hyperactivity_tscore)

4

# Grouped diagnostic distribution pie charts (0_3 vs 1_2)
og_dataset_copy$cluster_group <- factor(
  ifelse(as.character(og_dataset_copy$cluster) %in% c("0", "3"), "0_3", "1_2"),
  levels = c("0_3", "1_2")
)

grouped_clusters <- levels(og_dataset_copy$cluster_group)

for (grp in grouped_clusters) {
  cluster_data <- og_dataset_copy %>% filter(cluster_group == grp)
  total_n <- nrow(cluster_data)
  diag_counts <- cluster_data %>%
    group_by(diagnosis_group) %>%
    summarise(count = n(), .groups = "drop") %>%
    mutate(perc = count / sum(count) * 100)

  # Ensure all diagnosis groups are present (even if count = 0)
  diag_counts <- diag_counts %>%
    tidyr::complete(diagnosis_group = levels(og_dataset_copy$diagnosis_group), fill = list(count = 0, perc = 0)) %>%
    filter(count > 0)

  # Prepare geometry helpers for label positioning
  diag_counts <- diag_counts %>%
    arrange(desc(diagnosis_group)) %>%
    mutate(
      ymax = cumsum(count),
      ymin = c(0, head(ymax, n = -1)),
      mid = (ymax + ymin) / 2,
      angle = 90 - 360 * (mid / sum(count)),
      hjust = ifelse(angle < -90, 1, 0),
      angle = ifelse(angle < -90, angle + 180, angle)
    )

  grouped_pie <- ggplot(diag_counts, aes(x = "", y = count, fill = diagnosis_group)) +
    geom_col(width = 1, color = "white") +
    coord_polar(theta = "y", start = 0) +
    labs(
      title = paste0("Cluster ", grp, " (n = ", total_n, ") - Diagnostic Distribution"),
      fill = "Diagnosis"
    ) +
    theme_void(base_size = 16) +
    scale_fill_manual(values = diagnosis_colors, drop = FALSE) +
    guides(fill = guide_legend(override.aes = list(size = 6))) +
    theme(
      plot.title = element_text(hjust = 0, size = 12, face = "bold"),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 12)
    ) +
    geom_text(
      data = diag_counts %>% filter(count > 0),
      aes(
        y = mid,
        label = paste0(count, " (", sprintf("%.1f", perc), "%)"),
        angle = angle,
        hjust = hjust
      ),
      x = 1, # position labels outside the pie
      size = 5,
      color = "white",
      inherit.aes = FALSE
    )

  print(grouped_pie)
}

# Create a demographic summary table for each cluster
library(dplyr)

# Assuming your main data frame is called 'df' and has a column 'cluster'
# Adjust the data frame name if needed

demographic_summary <- df %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    male_n = sum(sex == "M", na.rm = TRUE),
    male_perc = round(100 * mean(sex == "M", na.rm = TRUE), 1),
    female_n = sum(sex == "F", na.rm = TRUE),
    female_perc = round(100 * mean(sex == "F", na.rm = TRUE), 1),
    age_mean = round(mean(age, na.rm = TRUE), 1),
    age_sd = round(sd(age, na.rm = TRUE), 1),
    iq_mean = round(mean(IQ, na.rm = TRUE), 1),
    iq_sd = round(sd(IQ, na.rm = TRUE), 1)
  )

# Frequency tables for categorical variables by cluster
sex_table <- df %>%
  group_by(cluster, sex) %>%
  summarise(n = n()) %>%
  mutate(perc = round(100 * n / sum(n), 1)) %>%
  arrange(cluster, desc(n))

ethnicity_table <- df %>%
  group_by(cluster, family_ethnicity_recoded) %>%
  summarise(n = n()) %>%
  mutate(perc = round(100 * n / sum(n), 1)) %>%
  arrange(cluster, desc(n))

education_table <- df %>%
  group_by(cluster, highest_education_level_recoded) %>%
  summarise(n = n()) %>%
  mutate(perc = round(100 * n / sum(n), 1)) %>%
  arrange(cluster, desc(n))

# Save the tables in pretty format for inclusion in a scientific article (e.g., as markdown or LaTeX)
library(knitr)

# Save as markdown tables (plain text, easy to copy into manuscripts)
writeLines(
  kable(demographic_summary, caption = "Demographic summary by cluster", format = "markdown"),
  "R_output/demographic_summary_by_cluster.txt"
)

writeLines(
  kable(sex_table, caption = "Sex distribution by cluster", format = "markdown"),
  "R_output/sex_distribution_by_cluster.txt"
)

writeLines(
  kable(ethnicity_table, caption = "Family ethnicity distribution by cluster", format = "markdown"),
  "R_output/family_ethnicity_distribution_by_cluster.txt"
)

writeLines(
  kable(education_table, caption = "Highest education level distribution by cluster", format = "markdown"),
  "R_output/highest_education_level_distribution_by_cluster.txt"
)

########################## Demographic Comparisons ##########################

# Test if clusters are comparable demographically
# This is important to ensure any differences in behavioral measures 
# are not due to demographic confounds

sink(file = "R_output/demographic_comparisons.txt")

cat("DEMOGRAPHIC COMPARISONS BETWEEN CLUSTERS\n")
cat("========================================\n\n")

# 1. AGE COMPARISON (Independent samples t-test)
cat("1. AGE COMPARISON\n")
cat("-----------------\n")
age_test <- t.test(age ~ cluster, data = df)
print(age_test)
cat("Effect size (Cohen's d):", abs(age_test$statistic) * sqrt(1/df$cluster[df$cluster == 0] %>% length() + 1/df$cluster[df$cluster == 1] %>% length()), "\n\n")

# 2. IQ COMPARISON (Independent samples t-test)
cat("2. IQ COMPARISON\n")
cat("----------------\n")
iq_test <- t.test(IQ ~ cluster, data = df)
print(iq_test)
cat("Effect size (Cohen's d):", abs(iq_test$statistic) * sqrt(1/df$cluster[df$cluster == 0] %>% length() + 1/df$cluster[df$cluster == 1] %>% length()), "\n\n")

# 3. SEX COMPARISON (Chi-square test)
cat("3. SEX COMPARISON\n")
cat("-----------------\n")
sex_table_test <- table(df$cluster, df$sex)
sex_chi2 <- chisq.test(sex_table_test)
print(sex_chi2)
cat("Cramér's V (effect size):", sqrt(sex_chi2$statistic / (nrow(df) * (min(nrow(sex_table_test), ncol(sex_table_test)) - 1))), "\n\n")

# 4. EDUCATION COMPARISON (Chi-square test)
cat("4. EDUCATION LEVEL COMPARISON\n")
cat("-----------------------------\n")
education_table_test <- table(df$cluster, df$highest_education_level_recoded)
education_chi2 <- chisq.test(education_table_test)
print(education_chi2)
cat("Cramér's V (effect size):", sqrt(education_chi2$statistic / (nrow(df) * (min(nrow(education_table_test), ncol(education_table_test)) - 1))), "\n\n")

# 5. ETHNICITY COMPARISON (Chi-square test)
cat("5. ETHNICITY COMPARISON\n")
cat("----------------------\n")
ethnicity_table_test <- table(df$cluster, df$family_ethnicity_recoded)
ethnicity_chi2 <- chisq.test(ethnicity_table_test)
print(ethnicity_chi2)
cat("Cramér's V (effect size):", sqrt(ethnicity_chi2$statistic / (nrow(df) * (min(nrow(ethnicity_table_test), ncol(ethnicity_table_test)) - 1))), "\n\n")

# Summary of results
cat("SUMMARY\n")
cat("=======\n")
cat("Significant differences (p < 0.05) indicate potential demographic confounds.\n")
cat("Non-significant differences suggest clusters are demographically comparable.\n")
cat("Effect sizes help interpret practical significance:\n")
cat("- Cohen's d: 0.2 = small, 0.5 = medium, 0.8 = large\n")
cat("- Cramér's V: 0.1 = small, 0.3 = medium, 0.5 = large\n\n")

sink()
sink(NULL)


# ================ EEG FEATURE ANALYSIS ==================
install.packages("car", "psych", "corrplot", "randomForest", "VIM", "GGally")


eeg_behavioral_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/merged_EEG_behavioral_data_V2.csv")

# Load required libraries
library(car)      # for MANOVA and Levene's test
library(psych)    # for descriptive statistics
library(ggplot2)  # for visualizations
library(dplyr)    # for data manipulation
library(corrplot) # for correlation plots
library(randomForest) # for feature importance
library(VIM)      # for missing data visualization
library(GGally)   # for scatter plot matrix

# Define EEG feature columns
eeg_features <- c("AlphaPower_MaxAcrossChannels", 
                  "Aperiodic_Offset", "Aperiodic_Exponent", 
                  "Average_Delta_Power",
                  "Average_Theta_Power",
                  "Average_Alpha_Power",
                  "Average_Beta_Power",
                  "Average_Gamma_Power",
                  "Average_PeriodicPSD_Delta",
                  "Average_PeriodicPSD_Theta",
                  "Average_PeriodicPSD_Alpha",
                  "Average_PeriodicPSD_Beta",
                  "Average_PeriodicPSD_Gamma",
                  "Average_RelDelta_Power",
                  "Average_RelTheta_Power",
                  "Average_RelAlpha_Power",
                  "Average_RelBeta_Power",
                  "Average_RelGamma_Power")

# Convert cluster to factor
eeg_behavioral_dataset$cluster <- as.factor(eeg_behavioral_dataset$cluster)

# Create output file for EEG analysis
sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/eeg_analysis_output.txt")

cat("========================================\n")
cat("SIMPLE EEG ANALYSIS BY CLUSTER\n")
cat("========================================\n\n")

# Simple summary statistics by cluster
cat("SAMPLE SIZES BY CLUSTER:\n")
print(table(eeg_behavioral_dataset$cluster))
cat("\n")

# Mean and SD for each EEG feature by cluster
cat("MEAN EEG VALUES BY CLUSTER:\n")
cat("============================\n\n")

for(cluster_id in levels(eeg_behavioral_dataset$cluster)) {
  cat("CLUSTER", cluster_id, ":\n")
  cat("-------------------\n")
  cluster_data <- eeg_behavioral_dataset[eeg_behavioral_dataset$cluster == cluster_id, eeg_features]
  
  # Calculate means and SDs
  means <- colMeans(cluster_data, na.rm = TRUE)
  sds <- apply(cluster_data, 2, sd, na.rm = TRUE)
  
  # Create summary table
  summary_table <- data.frame(
    Feature = names(means),
    Mean = round(means, 3),
    SD = round(sds, 3)
  )
  
  print(summary_table)
  cat("\n")
}

# Simple ANOVA results
cat("ANOVA RESULTS (F-statistics and p-values):\n")
cat("==========================================\n\n")

anova_summary <- data.frame()
for(feature in eeg_features) {
  formula_str <- paste(feature, "~ cluster")
  anova_model <- aov(as.formula(formula_str), data = eeg_behavioral_dataset)
  anova_result <- summary(anova_model)
  
  f_stat <- anova_result[[1]]$`F value`[1]
  p_val <- anova_result[[1]]$`Pr(>F)`[1]
  
  anova_summary <- rbind(anova_summary, 
                        data.frame(Feature = feature, 
                                 F_statistic = round(f_stat, 3), 
                                 p_value = round(p_val, 4),
                                 Significant = ifelse(p_val < 0.05, "YES", "NO")))
}

print(anova_summary)
cat("\n")

sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/grouped_clusters_anova_stats_output.txt")
# Grouped-cluster ANOVA: (0,3) vs (1,2)
cat("GROUPED-CLUSTER ANOVA: (0,3) vs (1,2)\n")
cat("====================================\n\n")

# Define grouped factor
eeg_behavioral_dataset$cluster_group <- factor(
  ifelse(as.character(eeg_behavioral_dataset$cluster) %in% c("0", "3"), "0_3", "1_2"),
  levels = c("0_3", "1_2")
)

# Show sample sizes per grouped cluster
cat("SAMPLE SIZES BY GROUPED CLUSTER:\n")
print(table(eeg_behavioral_dataset$cluster_group))
cat("\n")

# Means and SDs by grouped cluster for each EEG feature
cat("MEAN EEG VALUES BY GROUPED CLUSTER:\n")
cat("===================================\n\n")
for (grp in levels(eeg_behavioral_dataset$cluster_group)) {
  cat("GROUP", grp, ":\n")
  cat("-------------------\n")
  grp_data <- eeg_behavioral_dataset[eeg_behavioral_dataset$cluster_group == grp, eeg_features]
  g_means <- colMeans(grp_data, na.rm = TRUE)
  g_sds   <- apply(grp_data, 2, sd, na.rm = TRUE)
  grp_table <- data.frame(
    Feature = names(g_means),
    Mean = round(g_means, 3),
    SD = round(g_sds, 3)
  )
  print(grp_table)
  cat("\n")
}

# --- MANCOVA for Grouped Clusters (0_3 vs 1_2)
# Create grouped cluster variable for the main dataset
df$cluster_group <- factor(
  ifelse(as.character(df$cluster) %in% c("0", "3"), "0_3", "1_2"),
  levels = c("0_3", "1_2")
)

# =======================================================
# MANCOVA for grouped clusters
mancova_grouped_model <- manova(cbind(
  SRS_restrictive_repetitive_tscore,
  SRS_social_communication_tscore,
  SRS_social_cognition_tscore,
  attention_deficit_hyperactivity_tscore
) ~ cluster_group + age + IQ + sex, data = df)

cat("MANCOVA RESULTS FOR GROUPED CLUSTERS (0_3 vs 1_2):\n")
cat("================================================\n\n")

cat("Pillai's Trace (robust):\n")
summary(mancova_grouped_model, test = "Pillai")

cat("\nWilks' Lambda (traditional):\n")
summary(mancova_grouped_model, test = "Wilks")

# --- Univariate ANCOVAs per DV for grouped clusters
cat("\nUNIVARIATE ANCOVAs FOR GROUPED CLUSTERS:\n")
cat("========================================\n\n")

fit_grouped_one <- function(y) {
  f <- as.formula(paste(y, "~ cluster_group + age + IQ + sex"))
  lm(f, data = df)
}

fits_grouped <- lapply(DVS, fit_grouped_one)
names(fits_grouped) <- DVS

# Extract the cluster_group p-value, partial eta^2, omega^2, and N per DV
extract_grouped_stats <- function(fit) {
  aov_tab <- anova(fit)
  cl_row <- which(rownames(aov_tab) == "cluster_group")
  p      <- if (length(cl_row)) aov_tab$`Pr(>F)`[cl_row] else NA_real_
  
  eta2   <- eta_squared(fit, partial = TRUE, ci = NULL)
  eta2_c <- subset(eta2, grepl("^cluster_group$", Parameter))$Eta2_partial
  if (length(eta2_c) == 0) eta2_c <- NA_real_
  
  w2     <- omega_squared(fit, partial = TRUE, ci = NULL)
  w2_c   <- subset(w2, grepl("^cluster_group$", Parameter))$Omega2_partial
  if (length(w2_c) == 0) w2_c <- NA_real_
  
  c(n = stats::nobs(fit), p = p, eta2_partial = eta2_c, omega2_partial = w2_c)
}

tab_grouped <- as.data.frame(t(sapply(fits_grouped, extract_grouped_stats)))
tab_grouped$p_bonf <- p.adjust(tab_grouped$p, method = "bonferroni")
tab_grouped$p_fdr  <- p.adjust(tab_grouped$p, method = "BH")
tab_grouped

cat("Grouped Cluster ANCOVA Results:\n")
print(tab_grouped)
cat("\n")

## Adjusted means by grouped cluster
cat("ADJUSTED MEANS BY GROUPED CLUSTER:\n")
cat("=================================\n\n")

get_grouped_emm <- function(y) {
  f <- as.formula(paste(y, "~ cluster_group + age + IQ + sex"))
  fit <- lm(f, data = df)
  summary(emmeans(fit, ~ cluster_group))
}

emm_grouped_results <- lapply(DVS, get_grouped_emm)
names(emm_grouped_results) <- DVS

# See results
print(emm_grouped_results)
cat("\n")


# ANOVA per feature with grouped clusters
cat("ANOVA EEG RESULTS BY GROUPED CLUSTER (F and p-values):\n")
cat("==================================================\n\n")

anova_grouped_summary <- data.frame()
for (feature in eeg_features) {
  formula_str <- paste(feature, "~ cluster_group")
  aov_model <- aov(as.formula(formula_str), data = eeg_behavioral_dataset)
  aov_res <- summary(aov_model)
  f_val <- aov_res[[1]]$`F value`[1]
  p_val <- aov_res[[1]]$`Pr(>F)`[1]
  anova_grouped_summary <- rbind(
    anova_grouped_summary,
    data.frame(
      Feature = feature,
      F_statistic = round(f_val, 3),
      p_value = round(p_val, 4),
      Significant = ifelse(p_val < 0.05, "YES", "NO")
    )
  )
}

print(anova_grouped_summary)
cat("\n")
dev.off()

sink(file = 'demographic_table.txt')
# Scientific article-style table with global and individual cluster demographics
cat("SCIENTIFIC ARTICLE TABLE - GLOBAL AND CLUSTER DEMOGRAPHICS\n")
cat("==========================================================\n\n")

# Create comprehensive table for all clusters
all_cluster_summary <- df %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    # Demographics
    male_n = sum(sex == "M", na.rm = TRUE),
    male_perc = round(100 * mean(sex == "M", na.rm = TRUE), 1),
    female_n = sum(sex == "F", na.rm = TRUE),
    female_perc = round(100 * mean(sex == "F", na.rm = TRUE), 1),
    age_mean = round(mean(age, na.rm = TRUE), 1),
    age_sd = round(sd(age, na.rm = TRUE), 1),
    iq_mean = round(mean(IQ, na.rm = TRUE), 1),
    iq_sd = round(sd(IQ, na.rm = TRUE), 1),
    # Behavioral scores
    srs_social_cognition_mean = round(mean(SRS_social_cognition_tscore, na.rm = TRUE), 1),
    srs_social_cognition_sd = round(sd(SRS_social_cognition_tscore, na.rm = TRUE), 1),
    srs_social_communication_mean = round(mean(SRS_social_communication_tscore, na.rm = TRUE), 1),
    srs_social_communication_sd = round(sd(SRS_social_communication_tscore, na.rm = TRUE), 1),
    srs_restrictive_repetitive_mean = round(mean(SRS_restrictive_repetitive_tscore, na.rm = TRUE), 1),
    srs_restrictive_repetitive_sd = round(sd(SRS_restrictive_repetitive_tscore, na.rm = TRUE), 1),
    adhd_mean = round(mean(attention_deficit_hyperactivity_tscore, na.rm = TRUE), 1),
    adhd_sd = round(sd(attention_deficit_hyperactivity_tscore, na.rm = TRUE), 1)
  )

# Create global summary
global_summary <- df %>%
  summarise(
    n = n(),
    male_n = sum(sex == "M", na.rm = TRUE),
    male_perc = round(100 * mean(sex == "M", na.rm = TRUE), 1),
    female_n = sum(sex == "F", na.rm = TRUE),
    female_perc = round(100 * mean(sex == "F", na.rm = TRUE), 1),
    age_mean = round(mean(age, na.rm = TRUE), 1),
    age_sd = round(sd(age, na.rm = TRUE), 1),
    iq_mean = round(mean(IQ, na.rm = TRUE), 1),
    iq_sd = round(sd(IQ, na.rm = TRUE), 1),
    srs_social_cognition_mean = round(mean(SRS_social_cognition_tscore, na.rm = TRUE), 1),
    srs_social_cognition_sd = round(sd(SRS_social_cognition_tscore, na.rm = TRUE), 1),
    srs_social_communication_mean = round(mean(SRS_social_communication_tscore, na.rm = TRUE), 1),
    srs_social_communication_sd = round(sd(SRS_social_communication_tscore, na.rm = TRUE), 1),
    srs_restrictive_repetitive_mean = round(mean(SRS_restrictive_repetitive_tscore, na.rm = TRUE), 1),
    srs_restrictive_repetitive_sd = round(sd(SRS_restrictive_repetitive_tscore, na.rm = TRUE), 1),
    adhd_mean = round(mean(attention_deficit_hyperactivity_tscore, na.rm = TRUE), 1),
    adhd_sd = round(sd(attention_deficit_hyperactivity_tscore, na.rm = TRUE), 1)
  )

# Print the table in scientific format
cat("Table 1. Demographic and Behavioral Characteristics by Cluster\n")
cat("=============================================================\n\n")

# Helper function to get cluster data
get_cluster_data <- function(cluster_num) {
  if(cluster_num == "Global") {
    return(global_summary)
  } else {
    return(all_cluster_summary[all_cluster_summary$cluster == cluster_num, ])
  }
}

# Create grouped cluster summary for the table
grouped_cluster_summary <- df %>%
  group_by(cluster_group) %>%
  summarise(
    n = n(),
    # Demographics
    male_n = sum(sex == "M", na.rm = TRUE),
    male_perc = round(100 * mean(sex == "M", na.rm = TRUE), 1),
    female_n = sum(sex == "F", na.rm = TRUE),
    female_perc = round(100 * mean(sex == "F", na.rm = TRUE), 1),
    age_mean = round(mean(age, na.rm = TRUE), 1),
    age_sd = round(sd(age, na.rm = TRUE), 1),
    iq_mean = round(mean(IQ, na.rm = TRUE), 1),
    iq_sd = round(sd(IQ, na.rm = TRUE), 1),
    # Behavioral scores
    srs_social_cognition_mean = round(mean(SRS_social_cognition_tscore, na.rm = TRUE), 1),
    srs_social_cognition_sd = round(sd(SRS_social_cognition_tscore, na.rm = TRUE), 1),
    srs_social_communication_mean = round(mean(SRS_social_communication_tscore, na.rm = TRUE), 1),
    srs_social_communication_sd = round(sd(SRS_social_communication_tscore, na.rm = TRUE), 1),
    srs_restrictive_repetitive_mean = round(mean(SRS_restrictive_repetitive_tscore, na.rm = TRUE), 1),
    srs_restrictive_repetitive_sd = round(sd(SRS_restrictive_repetitive_tscore, na.rm = TRUE), 1),
    adhd_mean = round(mean(attention_deficit_hyperactivity_tscore, na.rm = TRUE), 1),
    adhd_sd = round(sd(attention_deficit_hyperactivity_tscore, na.rm = TRUE), 1)
  )

# Create formatted table with global and grouped clusters (3 columns)
scientific_table <- data.frame(
  Characteristic = c(
    "Sample size, n",
    "Sex, n (%)",
    "  Male",
    "  Female", 
    "Age, years, M (SD)",
    "IQ, M (SD)",
    "SRS Social Cognition T-score, M (SD)",
    "SRS Social Communication T-score, M (SD)", 
    "SRS Restrictive Repetitive T-score, M (SD)",
    "ADHD T-score, M (SD)"
  ),
  `Global (N=all)` = c(
    paste0(global_summary$n),
    "",
    paste0(global_summary$male_n, " (", global_summary$male_perc, "%)"),
    paste0(global_summary$female_n, " (", global_summary$female_perc, "%)"),
    paste0(global_summary$age_mean, " (", global_summary$age_sd, ")"),
    paste0(global_summary$iq_mean, " (", global_summary$iq_sd, ")"),
    paste0(global_summary$srs_social_cognition_mean, " (", global_summary$srs_social_cognition_sd, ")"),
    paste0(global_summary$srs_social_communication_mean, " (", global_summary$srs_social_communication_sd, ")"),
    paste0(global_summary$srs_restrictive_repetitive_mean, " (", global_summary$srs_restrictive_repetitive_sd, ")"),
    paste0(global_summary$adhd_mean, " (", global_summary$adhd_sd, ")")
  ),
  `Clusters 0+3 (Severe)` = c(
    paste0(grouped_cluster_summary$n[grouped_cluster_summary$cluster_group == "0_3"]),
    "",
    paste0(grouped_cluster_summary$male_n[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$male_perc[grouped_cluster_summary$cluster_group == "0_3"], "%)"),
    paste0(grouped_cluster_summary$female_n[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$female_perc[grouped_cluster_summary$cluster_group == "0_3"], "%)"),
    paste0(grouped_cluster_summary$age_mean[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$age_sd[grouped_cluster_summary$cluster_group == "0_3"], ")"),
    paste0(grouped_cluster_summary$iq_mean[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$iq_sd[grouped_cluster_summary$cluster_group == "0_3"], ")"),
    paste0(grouped_cluster_summary$srs_social_cognition_mean[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$srs_social_cognition_sd[grouped_cluster_summary$cluster_group == "0_3"], ")"),
    paste0(grouped_cluster_summary$srs_social_communication_mean[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$srs_social_communication_sd[grouped_cluster_summary$cluster_group == "0_3"], ")"),
    paste0(grouped_cluster_summary$srs_restrictive_repetitive_mean[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$srs_restrictive_repetitive_sd[grouped_cluster_summary$cluster_group == "0_3"], ")"),
    paste0(grouped_cluster_summary$adhd_mean[grouped_cluster_summary$cluster_group == "0_3"], 
           " (", grouped_cluster_summary$adhd_sd[grouped_cluster_summary$cluster_group == "0_3"], ")")
  ),
  `Clusters 1+2 (Mild)` = c(
    paste0(grouped_cluster_summary$n[grouped_cluster_summary$cluster_group == "1_2"]),
    "",
    paste0(grouped_cluster_summary$male_n[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$male_perc[grouped_cluster_summary$cluster_group == "1_2"], "%)"),
    paste0(grouped_cluster_summary$female_n[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$female_perc[grouped_cluster_summary$cluster_group == "1_2"], "%)"),
    paste0(grouped_cluster_summary$age_mean[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$age_sd[grouped_cluster_summary$cluster_group == "1_2"], ")"),
    paste0(grouped_cluster_summary$iq_mean[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$iq_sd[grouped_cluster_summary$cluster_group == "1_2"], ")"),
    paste0(grouped_cluster_summary$srs_social_cognition_mean[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$srs_social_cognition_sd[grouped_cluster_summary$cluster_group == "1_2"], ")"),
    paste0(grouped_cluster_summary$srs_social_communication_mean[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$srs_social_communication_sd[grouped_cluster_summary$cluster_group == "1_2"], ")"),
    paste0(grouped_cluster_summary$srs_restrictive_repetitive_mean[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$srs_restrictive_repetitive_sd[grouped_cluster_summary$cluster_group == "1_2"], ")"),
    paste0(grouped_cluster_summary$adhd_mean[grouped_cluster_summary$cluster_group == "1_2"], 
           " (", grouped_cluster_summary$adhd_sd[grouped_cluster_summary$cluster_group == "1_2"], ")")
  )
)

# Print the formatted table
print(scientific_table)
cat("\n")
dev.off()

# Add statistical comparisons
cat("STATISTICAL COMPARISONS BETWEEN GROUPED CLUSTERS:\n")
cat("===============================================\n\n")

# Age comparison
age_test <- t.test(age ~ cluster_group, data = eeg_behavioral_dataset)
cat("Age: t =", round(age_test$statistic, 2), ", p =", round(age_test$p.value, 3), "\n")

# IQ comparison  
iq_test <- t.test(IQ ~ cluster_group, data = eeg_behavioral_dataset)
cat("IQ: t =", round(iq_test$statistic, 2), ", p =", round(iq_test$p.value, 3), "\n")

# Sex comparison
sex_table <- table(eeg_behavioral_dataset$cluster_group, eeg_behavioral_dataset$sex)
sex_chi2 <- chisq.test(sex_table)
cat("Sex: χ² =", round(sex_chi2$statistic, 2), ", p =", round(sex_chi2$p.value, 3), "\n")

# Behavioral comparisons
srs_soc_cog_test <- t.test(SRS_social_cognition_tscore ~ cluster_group, data = eeg_behavioral_dataset)
cat("SRS Social Cognition: t =", round(srs_soc_cog_test$statistic, 2), ", p =", round(srs_soc_cog_test$p.value, 3), "\n")

srs_soc_comm_test <- t.test(SRS_social_communication_tscore ~ cluster_group, data = eeg_behavioral_dataset)
cat("SRS Social Communication: t =", round(srs_soc_comm_test$statistic, 2), ", p =", round(srs_soc_comm_test$p.value, 3), "\n")

srs_rr_test <- t.test(SRS_restrictive_repetitive_tscore ~ cluster_group, data = eeg_behavioral_dataset)
cat("SRS Restrictive Repetitive: t =", round(srs_rr_test$statistic, 2), ", p =", round(srs_rr_test$p.value, 3), "\n")

adhd_test <- t.test(attention_deficit_hyperactivity_tscore ~ cluster_group, data = eeg_behavioral_dataset)
cat("ADHD: t =", round(adhd_test$statistic, 2), ", p =", round(adhd_test$p.value, 3), "\n")

cat("\n")


# Box plots for grouped clusters (0_3 vs 1_2)
cat("Creating boxplots for grouped clusters (0_3 vs 1_2)...\n")

# Function to get significance for grouped clusters
get_grouped_significance <- function(feature, data) {
  # Perform t-test between the two groups
  group_0_3 <- data[data$cluster_group == "0_3", feature]
  group_1_2 <- data[data$cluster_group == "1_2", feature]
  
  # Remove NA values
  group_0_3 <- group_0_3[!is.na(group_0_3)]
  group_1_2 <- group_1_2[!is.na(group_1_2)]
  
  if(length(group_0_3) > 1 && length(group_1_2) > 1) {
    t_test <- t.test(group_0_3, group_1_2)
    p_val <- t_test$p.value
    significant <- p_val < 0.05
    highly_significant <- p_val < 0.001
  } else {
    p_val <- NA
    significant <- FALSE
    highly_significant <- FALSE
  }
  
  return(list(p_value = p_val, significant = significant, highly_significant = highly_significant))
}

# Function to add significance annotations to grouped cluster plots
add_grouped_significance_annotations <- function(plot, feature, data) {
  # Get significance
  sig_result <- get_grouped_significance(feature, data)
  
  # Get y-axis range for positioning
  y_range <- range(data[[feature]], na.rm = TRUE)
  y_max <- max(y_range)
  y_min <- min(y_range)
  y_range_span <- y_max - y_min
  
  # Add significance line and annotation if significant
  plot_with_annotations <- plot
  
  if(sig_result$significant) {
    # Calculate y position for the line
    y_pos <- y_max + (y_range_span * 0.1)
    
    # Add significance line connecting the two groups
    plot_with_annotations <- plot_with_annotations +
      annotate("segment", 
               x = 1, xend = 2, 
               y = y_pos, yend = y_pos,
               color = "black", size = 0.8)
    
    # Add significance marker
    if(sig_result$highly_significant) {
      plot_with_annotations <- plot_with_annotations +
        annotate("text", 
                 x = 1.5, 
                 y = y_pos + (y_range_span * 0.02),
                 label = "**", 
                 size = 5, 
                 fontface = "bold",
                 color = "black")
    } else {
      plot_with_annotations <- plot_with_annotations +
        annotate("text", 
                 x = 1.5, 
                 y = y_pos + (y_range_span * 0.02),
                 label = "*", 
                 size = 5, 
                 fontface = "bold",
                 color = "black")
    }
  }
  
  return(plot_with_annotations)
}

# Create box plots for ALL EEG features with grouped clusters
pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/grouped_cluster_eeg_features.pdf", 
    width = 12, height = 8)

for(feature in eeg_features) {
  # Create base box plot for grouped clusters
  p <- ggplot(eeg_behavioral_dataset, aes(x = cluster_group, y = .data[[feature]], fill = cluster_group)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.6, size = 1) +
    labs(title = paste("Distribution of", feature, "by Grouped Severity Clusters"),
         x = "Grouped Cluster  - Clusters 0 + 3 (Severe) vs Clusters 1 + 2 (Mild)", 
         y = feature) +
    theme_minimal() +
    theme(legend.position = "none",
          plot.title = element_text(size = 12, face = "bold"),
          axis.text = element_text(size = 10),
          axis.title = element_text(size = 11)) +
    scale_fill_manual(values = c("0_3" = "#E91E63", "1_2" = "#2196F3"))
  
  # Add significance annotations
  p_with_annotations <- add_grouped_significance_annotations(p, feature, eeg_behavioral_dataset)
  
  print(p_with_annotations)
}

dev.off()
cat("Grouped cluster EEG feature boxplots saved to: /Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/grouped_cluster_eeg_features.pdf\n")

# Top 5 most different features
significant_features <- anova_summary[anova_summary$Significant == "YES", ]
if(nrow(significant_features) > 0) {
  cat("TOP 5 MOST DISCRIMINATIVE EEG FEATURES:\n")
  cat("=======================================\n")
  top_features <- head(significant_features[order(significant_features$F_statistic, decreasing = TRUE), ], 5)
  print(top_features)
  cat("\n")
} else {
  cat("No significant differences found between clusters.\n\n")
}

# Cluster comparison table for top features
if(nrow(significant_features) > 0) {
  cat("CLUSTER COMPARISON FOR TOP FEATURES:\n")
  cat("===================================\n\n")
  
  top_5_features <- head(significant_features[order(significant_features$F_statistic, decreasing = TRUE), "Feature"], 5)
  
  for(feature in top_5_features) {
    cat(feature, ":\n")
    cat("--------\n")
    
    # Get means by cluster for this feature
    cluster_means <- aggregate(eeg_behavioral_dataset[[feature]] ~ cluster, 
                              data = eeg_behavioral_dataset, 
                              FUN = function(x) round(mean(x, na.rm = TRUE), 3))
    names(cluster_means) <- c("Cluster", "Mean")
    print(cluster_means)
    cat("\n")
  }
}

# Box plots for ALL EEG features with pairwise comparisons
cat("Creating boxplots for all EEG features with pairwise comparisons...\n")

# Function to get pairwise comparisons and significance levels
get_pairwise_comparisons <- function(feature, data) {
  # Perform Tukey HSD test
  formula_str <- paste(feature, "~ cluster")
  anova_model <- aov(as.formula(formula_str), data = data)
  tukey_results <- TukeyHSD(anova_model)
  
  # Extract comparison results
  comparisons <- tukey_results$cluster
  comparison_df <- data.frame(
    comparison = rownames(comparisons),
    p_value = comparisons[, "p adj"],
    significant = comparisons[, "p adj"] < 0.05,
    highly_significant = comparisons[, "p adj"] < 0.001
  )
  
  return(comparison_df)
}

# Function to add significance annotations to plots
add_significance_annotations <- function(plot, feature, data) {
  # Get pairwise comparisons
  comparisons <- get_pairwise_comparisons(feature, data)
  
  # Get y-axis range for positioning
  y_range <- range(data[[feature]], na.rm = TRUE)
  y_max <- max(y_range)
  y_min <- min(y_range)
  y_range_span <- y_max - y_min
  
  # Add significance lines and annotations
  plot_with_annotations <- plot
  
  # Add comparison lines and significance markers at different heights
  if(nrow(comparisons) > 0) {
    # Filter to only significant comparisons
    significant_comparisons <- comparisons[comparisons$significant == TRUE, ]
    
    if(nrow(significant_comparisons) > 0) {
      # Calculate different y positions for each comparison
      base_y <- y_max + (y_range_span * 0.1)
      y_step <- y_range_span * 0.08  # Space between lines
      
      for(i in 1:nrow(significant_comparisons)) {
        comp <- significant_comparisons[i, ]
        
        # Parse comparison (e.g., "1-0" means cluster 1 vs cluster 0)
        clusters <- strsplit(comp$comparison, "-")[[1]]
        cluster1 <- as.numeric(clusters[1])
        cluster2 <- as.numeric(clusters[2])
        
        # Calculate x positions for the line (centered on clusters)
        x1 <- cluster1 + 1  # +1 because clusters are 0,1,2,3 but positions are 1,2,3,4
        x2 <- cluster2 + 1
        
        # Calculate y position for this specific comparison
        y_pos <- base_y + (i - 1) * y_step
        
        # Add significance line
        plot_with_annotations <- plot_with_annotations +
          annotate("segment", 
                   x = x1, xend = x2, 
                   y = y_pos, yend = y_pos,
                   color = "black", size = 0.8)
        
        # Add significance marker directly above the line
        if(comp$highly_significant) {
          plot_with_annotations <- plot_with_annotations +
            annotate("text", 
                     x = (x1 + x2) / 2, 
                     y = y_pos + (y_range_span * 0.02),
                     label = "**", 
                     size = 5, 
                     fontface = "bold",
                     color = "black")
        } else if(comp$significant) {
          plot_with_annotations <- plot_with_annotations +
            annotate("text", 
                     x = (x1 + x2) / 2, 
                     y = y_pos + (y_range_span * 0.02),
                     label = "*", 
                     size = 5, 
                     fontface = "bold",
                     color = "black")
        }
      }
    }
  }
  
  return(plot_with_annotations)
}

# Create box plots for ALL EEG features
pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/all_eeg_features_with_comparisons.pdf", 
    width = 12, height = 8)

for(feature in eeg_features) {
  # Create base box plot
  p <- ggplot(eeg_behavioral_dataset, aes(x = cluster, y = .data[[feature]], fill = cluster)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.6, size = 1) +
    labs(title = paste("Distribution of", feature, "by Cluster"),
         x = "Cluster", 
         y = feature) +
    theme_minimal() +
    theme(legend.position = "none",
          plot.title = element_text(size = 12, face = "bold"),
          axis.text = element_text(size = 10),
          axis.title = element_text(size = 11))
  
  # Add significance annotations
  p_with_annotations <- add_significance_annotations(p, feature, eeg_behavioral_dataset)
  
  print(p_with_annotations)
}

dev.off()
cat("All EEG feature boxplots with pairwise comparisons saved to: /Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/all_eeg_features_with_comparisons.pdf\n")

cat("\nSIMPLE ANALYSIS COMPLETE!\n")
cat("==========================\n")
cat("Results saved to: /Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/eeg_analysis_output.txt\n")


# ================= REGRESSION: SRS ~ Periodic Theta ==================
# Linear regressions of SRS measures on Average_PeriodicPSD_Theta

# Ensure grouped factor exists in the EEG-behavioral dataset
if (!"cluster_group" %in% names(eeg_behavioral_dataset)) {
  eeg_behavioral_dataset$cluster_group <- factor(
    ifelse(as.character(eeg_behavioral_dataset$cluster) %in% c("0", "3"), "0_3", "1_2"),
    levels = c("0_3", "1_2")
  )
}

srs_outcomes <- c(
  "SRS_social_cognition_tscore",
  "SRS_social_communication_tscore",
  "SRS_restrictive_repetitive_tscore"
)

theta_feature <- "Average_PeriodicPSD_Theta"

sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/eeg_theta_srs_regression.txt")
cat("==============================================================\n")
cat("REGRESSION: SRS MEASURES ~ Average_PeriodicPSD_Theta (adjusted)\n")
cat("==============================================================\n\n")

if (!(theta_feature %in% names(eeg_behavioral_dataset))) {
  cat("Feature not found:", theta_feature, "\n")
} else {
  reg_results <- data.frame()
  for (y in srs_outcomes) {
    if (!(y %in% names(eeg_behavioral_dataset))) {
      next
    }
    f <- as.formula(paste(
      y,
      "~",
      paste(c(theta_feature, "age_at_test", "nonverbal_iq", "sex", "cluster_group"), collapse = " + ")
    ))
    fit <- lm(f, data = eeg_behavioral_dataset)
    smry <- summary(fit)
    coefs <- smry$coefficients
    if (theta_feature %in% rownames(coefs)) {
      beta <- coefs[theta_feature, "Estimate"]
      se   <- coefs[theta_feature, "Std. Error"]
      tval <- coefs[theta_feature, "t value"]
      pval <- coefs[theta_feature, "Pr(>|t|)"]
    } else {
      beta <- se <- tval <- pval <- NA_real_
    }
    r2   <- smry$r.squared
    r2a  <- smry$adj.r.squared
    nobs <- stats::nobs(fit)
    reg_results <- rbind(
      reg_results,
      data.frame(
        Outcome = y,
        Theta_Beta = round(beta, 4),
        Theta_SE = round(se, 4),
        Theta_t = round(tval, 3),
        Theta_p = signif(pval, 3),
        R2 = round(r2, 3),
        R2_adj = round(r2a, 3),
        N = nobs
      )
    )
  }
  print(reg_results, row.names = FALSE)
}
sink()

# Scatter plots with regression lines for each SRS vs Periodic Theta
if (theta_feature %in% names(eeg_behavioral_dataset)) {
  pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/srs_vs_periodic_theta.pdf", width = 10, height = 8)
  for (y in srs_outcomes) {
    if (!(y %in% names(eeg_behavioral_dataset))) next
    p <- ggplot(eeg_behavioral_dataset, aes(x = .data[[theta_feature]], y = .data[[y]], color = cluster_group)) +
      geom_point(alpha = 0.7) +
      geom_smooth(method = "lm", se = TRUE, color = "black") +
      labs(
        title = paste("Regression:", y, "~", theta_feature),
        x = theta_feature,
        y = y,
        color = "Group"
      ) +
      theme_minimal()
    print(p)
  }
  dev.off()
}

# ================= REGRESSION: ADHD ~ Periodic Theta ==================
# Linear regression of ADHD t-score on Average_PeriodicPSD_Theta

adhd_outcome <- "attention_deficit_hyperactivity_tscore"

sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/eeg_theta_adhd_regression.txt")
cat("========================================================\n")
cat("REGRESSION: ADHD ~ Average_PeriodicPSD_Theta (adjusted)\n")
cat("========================================================\n\n")

if (!(theta_feature %in% names(eeg_behavioral_dataset))) {
  cat("Feature not found:", theta_feature, "\n")
} else if (!(adhd_outcome %in% names(eeg_behavioral_dataset))) {
  cat("Outcome not found:", adhd_outcome, "\n")
} else {
  f <- as.formula(paste(
    adhd_outcome,
    "~",
    paste(c(theta_feature, "age_at_test", "nonverbal_iq", "sex", "cluster_group"), collapse = " + ")
  ))
  fit <- lm(f, data = eeg_behavioral_dataset)
  smry <- summary(fit)
  coefs <- smry$coefficients
  if (theta_feature %in% rownames(coefs)) {
    beta <- coefs[theta_feature, "Estimate"]
    se   <- coefs[theta_feature, "Std. Error"]
    tval <- coefs[theta_feature, "t value"]
    pval <- coefs[theta_feature, "Pr(>|t|)"]
  } else {
    beta <- se <- tval <- pval <- NA_real_
  }
  r2   <- smry$r.squared
  r2a  <- smry$adj.r.squared
  nobs <- stats::nobs(fit)
  res_line <- data.frame(
    Outcome = adhd_outcome,
    Theta_Beta = round(beta, 4),
    Theta_SE = round(se, 4),
    Theta_t = round(tval, 3),
    Theta_p = signif(pval, 3),
    R2 = round(r2, 3),
    R2_adj = round(r2a, 3),
    N = nobs
  )
  print(res_line, row.names = FALSE)
}
sink()

# Scatter plot with regression line for ADHD vs Periodic Theta
if (theta_feature %in% names(eeg_behavioral_dataset) && adhd_outcome %in% names(eeg_behavioral_dataset)) {
  pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/adhd_vs_periodic_theta.pdf", width = 8, height = 6)
  p <- ggplot(eeg_behavioral_dataset, aes(x = .data[[theta_feature]], y = .data[[adhd_outcome]], color = cluster_group)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE, color = "black") +
    labs(
      title = paste("Regression:", adhd_outcome, "~", theta_feature),
      x = theta_feature,
      y = adhd_outcome,
      color = "Group"
    ) +
    theme_minimal()
  print(p)
  dev.off()
}

# ================= CORRELATIONS: SRS ↔ Periodic Theta =================
# Pearson and Spearman correlations overall and by grouped clusters

sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/eeg_theta_srs_correlations.txt")
cat("==============================================================\n")
cat("CORRELATIONS: SRS MEASURES ↔ Average_PeriodicPSD_Theta\n")
cat("(Pearson and Spearman; overall and by grouped clusters)\n")
cat("==============================================================\n\n")

if (!(theta_feature %in% names(eeg_behavioral_dataset))) {
  cat("Feature not found:", theta_feature, "\n")
} else {
  # Overall correlations
  cat("OVERALL CORRELATIONS\n")
  cat("--------------------\n")
  corr_rows <- list()
  for (y in srs_outcomes) {
    if (!(y %in% names(eeg_behavioral_dataset))) next
    x <- eeg_behavioral_dataset[[theta_feature]]
    yy <- eeg_behavioral_dataset[[y]]
    ok <- is.finite(x) & is.finite(yy)
    if (sum(ok) >= 3) {
      ct_p <- suppressWarnings(cor.test(x[ok], yy[ok], method = "pearson"))
      ct_s <- suppressWarnings(cor.test(x[ok], yy[ok], method = "spearman", exact = FALSE))
      corr_rows[[length(corr_rows) + 1]] <- data.frame(
        Outcome = y,
        Pearson_r = round(unname(ct_p$estimate), 3),
        Pearson_p = signif(ct_p$p.value, 3),
        Spearman_rho = round(unname(ct_s$estimate), 3),
        Spearman_p = signif(ct_s$p.value, 3),
        N = sum(ok)
      )
    }
  }
  if (length(corr_rows)) print(do.call(rbind, corr_rows), row.names = FALSE)
  cat("\n")

  # By grouped clusters
  if (!"cluster_group" %in% names(eeg_behavioral_dataset)) {
    eeg_behavioral_dataset$cluster_group <- factor(
      ifelse(as.character(eeg_behavioral_dataset$cluster) %in% c("0", "3"), "0_3", "1_2"),
      levels = c("0_3", "1_2")
    )
  }
  cat("CORRELATIONS BY GROUPED CLUSTERS (0_3 vs 1_2)\n")
  cat("---------------------------------------------\n")
  for (grp in levels(eeg_behavioral_dataset$cluster_group)) {
    cat("Group:", grp, "\n")
    sub <- eeg_behavioral_dataset[eeg_behavioral_dataset$cluster_group == grp, ]
    corr_rows <- list()
    for (y in srs_outcomes) {
      if (!(y %in% names(sub))) next
      x <- sub[[theta_feature]]
      yy <- sub[[y]]
      ok <- is.finite(x) & is.finite(yy)
      if (sum(ok) >= 3) {
        ct_p <- suppressWarnings(cor.test(x[ok], yy[ok], method = "pearson"))
        ct_s <- suppressWarnings(cor.test(x[ok], yy[ok], method = "spearman", exact = FALSE))
        corr_rows[[length(corr_rows) + 1]] <- data.frame(
          Outcome = y,
          Pearson_r = round(unname(ct_p$estimate), 3),
          Pearson_p = signif(ct_p$p.value, 3),
          Spearman_rho = round(unname(ct_s$estimate), 3),
          Spearman_p = signif(ct_s$p.value, 3),
          N = sum(ok)
        )
      }
    }
    if (length(corr_rows)) print(do.call(rbind, corr_rows), row.names = FALSE)
    cat("\n")
  }
}
sink()

# Scatter plots with correlation annotations (overall)
if (theta_feature %in% names(eeg_behavioral_dataset)) {
  pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/srs_theta_correlations.pdf", width = 10, height = 8)
  for (y in srs_outcomes) {
    if (!(y %in% names(eeg_behavioral_dataset))) next
    # Compute overall Pearson r for annotation
    x <- eeg_behavioral_dataset[[theta_feature]]
    yy <- eeg_behavioral_dataset[[y]]
    ok <- is.finite(x) & is.finite(yy)
    ann <- ""
    if (sum(ok) >= 3) {
      ct_p <- suppressWarnings(cor.test(x[ok], yy[ok], method = "pearson"))
      ann <- paste0("r = ", round(unname(ct_p$estimate), 2), ", p = ", signif(ct_p$p.value, 2))
    }
    p <- ggplot(eeg_behavioral_dataset, aes(x = .data[[theta_feature]], y = .data[[y]], color = cluster_group)) +
      geom_point(alpha = 0.7) +
      geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
      labs(
        title = paste("Correlation:", y, "↔", theta_feature),
        subtitle = ann,
        x = theta_feature,
        y = y,
        color = "Group"
      ) +
      theme_minimal()
    print(p)
  }
  dev.off()
}

# ==============================

# ============ ANALYSIS FOR 3-GROUP CLUSTERS: 0, 1_2, 3 ============

# Create 3-level grouped cluster on main df
df$cluster_group_3 <- factor(
  dplyr::case_when(
    as.character(df$cluster) == "0" ~ "0",
    as.character(df$cluster) %in% c("1", "2") ~ "1_2",
    as.character(df$cluster) == "3" ~ "3",
    TRUE ~ NA_character_
  ),
  levels = c("0", "1_2", "3")
)

# MANCOVA for 3-level grouped clusters
mancova_group3 <- manova(cbind(
  SRS_restrictive_repetitive_tscore,
  SRS_social_communication_tscore,
  SRS_social_cognition_tscore,
  attention_deficit_hyperactivity_tscore
) ~ cluster_group_3 + age + IQ + sex, data = df)

sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/mancova_cluster_group_3.txt")
cat("MANCOVA RESULTS FOR CLUSTER GROUPS (0, 1_2, 3)\n")
cat("============================================\n\n")
cat("Pillai's Trace:\n"); print(summary(mancova_group3, test = "Pillai")); cat("\n")
cat("Wilks' Lambda:\n"); print(summary(mancova_group3, test = "Wilks")); cat("\n")

# Univariate ANCOVAs per DV with effect sizes
cat("UNIVARIATE ANCOVAs (with covariates)\n")
cat("------------------------------------\n\n")
DVS <- c(
  "SRS_restrictive_repetitive_tscore",
  "SRS_social_communication_tscore",
  "SRS_social_cognition_tscore",
  "attention_deficit_hyperactivity_tscore"
)

fit_g3 <- function(y) lm(as.formula(paste(y, "~ cluster_group_3 + age + IQ + sex")), data = df)
fits_g3 <- lapply(DVS, fit_g3); names(fits_g3) <- DVS

g3_rows <- lapply(fits_g3, function(fit) {
  aov_tab <- anova(fit)
  p <- aov_tab$`Pr(>F)`[which(rownames(aov_tab) == "cluster_group_3")]
  e <- effectsize::eta_squared(fit, partial = TRUE, ci = NULL)
  eta <- subset(e, grepl("^cluster_group_3$", Parameter))$Eta2_partial
  if (length(eta) == 0) eta <- NA_real_
  w <- effectsize::omega_squared(fit, partial = TRUE, ci = NULL)
  omg <- subset(w, grepl("^cluster_group_3$", Parameter))$Omega2_partial
  if (length(omg) == 0) omg <- NA_real_
  data.frame(p = p, eta2_partial = eta, omega2_partial = omg, n = stats::nobs(fit))
})

tab_g3 <- as.data.frame(do.call(rbind, g3_rows))
tab_g3$p_bonf <- p.adjust(tab_g3$p, method = "bonferroni")
tab_g3$p_fdr  <- p.adjust(tab_g3$p, method = "BH")
tab_g3$DV <- DVS
tab_g3 <- tab_g3[, c("DV", setdiff(names(tab_g3), "DV"))]
print(tab_g3); cat("\n")

# Adjusted means by cluster_group_3
cat("ADJUSTED MEANS BY CLUSTER GROUP (0, 1_2, 3)\n")
cat("------------------------------------------\n\n")
emm_g3 <- lapply(DVS, function(y) {
  fit <- lm(as.formula(paste(y, "~ cluster_group_3 + age + IQ + sex")), data = df)
  summary(emmeans::emmeans(fit, ~ cluster_group_3))
})
names(emm_g3) <- DVS
print(emm_g3)
sink()

# Diagnostic distribution pie charts for 3-level groups
og_dataset_copy$cluster_group_3 <- factor(
  dplyr::case_when(
    as.character(og_dataset_copy$cluster) == "0" ~ "0",
    as.character(og_dataset_copy$cluster) %in% c("1", "2") ~ "1_2",
    as.character(og_dataset_copy$cluster) == "3" ~ "3",
    TRUE ~ NA_character_
  ),
  levels = c("0", "1_2", "3")
)

pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/diagnosis_pies_cluster_group_3.pdf", width = 8, height = 6)
for (grp in levels(og_dataset_copy$cluster_group_3)) {
  cluster_data <- dplyr::filter(og_dataset_copy, cluster_group_3 == grp)
  total_n <- nrow(cluster_data)
  diag_counts <- cluster_data %>%
    dplyr::group_by(diagnosis_group) %>%
    dplyr::summarise(count = dplyr::n(), .groups = "drop") %>%
    dplyr::mutate(perc = count / sum(count) * 100) %>%
    tidyr::complete(diagnosis_group = levels(og_dataset_copy$diagnosis_group), fill = list(count = 0, perc = 0)) %>%
    dplyr::filter(count > 0) %>%
    dplyr::arrange(dplyr::desc(diagnosis_group)) %>%
    dplyr::mutate(
      ymax = cumsum(count), ymin = dplyr::lag(ymax, default = 0), mid = (ymax + ymin) / 2,
      angle = 90 - 360 * (mid / sum(count)), hjust = ifelse(angle < -90, 1, 0), angle = ifelse(angle < -90, angle + 180, angle)
    )
  pie_chart <- ggplot2::ggplot(diag_counts, ggplot2::aes(x = "", y = count, fill = diagnosis_group)) +
    ggplot2::geom_col(width = 1, color = "white") +
    ggplot2::coord_polar(theta = "y", start = 0) +
    ggplot2::labs(title = paste0("Group ", grp, " (n = ", total_n, ") - Diagnostic Distribution"), fill = "Diagnosis") +
    ggplot2::theme_void(base_size = 16) +
    ggplot2::scale_fill_manual(values = diagnosis_colors, drop = FALSE) +
    ggplot2::guides(fill = ggplot2::guide_legend(override.aes = list(size = 6))) +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0, size = 12, face = "bold"), legend.title = ggplot2::element_text(size = 12), legend.text = ggplot2::element_text(size = 12)) +
    ggplot2::geom_text(data = dplyr::filter(diag_counts, count > 0), ggplot2::aes(y = mid, label = paste0(count, " (", sprintf("%.1f", perc), "%)"), angle = angle, hjust = hjust), x = 1.1, size = 3, inherit.aes = FALSE)
  print(pie_chart)
}
dev.off()

# EEG ANOVAs and boxplots by cluster_group_3
if (!"cluster_group_3" %in% names(eeg_behavioral_dataset)) {
  eeg_behavioral_dataset$cluster_group_3 <- factor(
    dplyr::case_when(
      as.character(eeg_behavioral_dataset$cluster) == "0" ~ "0",
      as.character(eeg_behavioral_dataset$cluster) %in% c("1", "2") ~ "1_2",
      as.character(eeg_behavioral_dataset$cluster) == "3" ~ "3",
      TRUE ~ NA_character_
    ),
    levels = c("0", "1_2", "3")
  )
}

sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/eeg_anova_cluster_group_3.txt")
cat("EEG ANOVA BY CLUSTER GROUP (0, 1_2, 3)\n")
cat("====================================\n\n")
anova_g3 <- data.frame()
for (feature in eeg_features) {
  model <- aov(as.formula(paste(feature, "~ cluster_group_3")), data = eeg_behavioral_dataset)
  sr <- summary(model)
  f <- sr[[1]]$`F value`[1]; p <- sr[[1]]$`Pr(>F)`[1]
  anova_g3 <- rbind(anova_g3, data.frame(Feature = feature, F_statistic = round(f, 3), p_value = round(p, 4), Significant = ifelse(p < 0.05, "YES", "NO")))
}
print(anova_g3); cat("\n")
sink()

pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/eeg_boxplots_cluster_group_3.pdf", width = 12, height = 8)
get_pairwise_comparisons_group3 <- function(feature, data) {
  formula_str <- paste(feature, "~ cluster_group_3")
  anova_model <- aov(stats::as.formula(formula_str), data = data)
  tukey_results <- TukeyHSD(anova_model)
  comps <- tukey_results$cluster_group_3
  data.frame(
    comparison = rownames(comps),
    p_value = comps[, "p adj"],
    significant = comps[, "p adj"] < 0.05,
    highly_significant = comps[, "p adj"] < 0.001,
    stringsAsFactors = FALSE
  )
}

add_significance_annotations_group3 <- function(plot, feature, data) {
  comparisons <- get_pairwise_comparisons_group3(feature, data)
  if (nrow(comparisons) == 0) return(plot)
  y_range <- range(data[[feature]], na.rm = TRUE)
  y_max <- max(y_range); y_min <- min(y_range); y_span <- y_max - y_min
  if (!is.finite(y_span) || y_span == 0) y_span <- 1
  sig <- comparisons[comparisons$significant, , drop = FALSE]
  if (nrow(sig) == 0) return(plot)
  base_y <- y_max + (y_span * 0.1)
  y_step <- y_span * 0.08
  lvl <- levels(data$cluster_group_3)
  p_out <- plot
  for (i in seq_len(nrow(sig))) {
    comp <- sig$comparison[i]
    parts <- strsplit(comp, "-")[[1]]
    g1 <- parts[1]; g2 <- parts[2]
    x1 <- match(g1, lvl); x2 <- match(g2, lvl)
    if (is.na(x1) || is.na(x2)) next
    y_pos <- base_y + (i - 1) * y_step
    p_out <- p_out + ggplot2::annotate("segment", x = x1, xend = x2, y = y_pos, yend = y_pos, color = "black", size = 0.8)
    label <- if (sig$highly_significant[i]) "**" else "*"
    p_out <- p_out + ggplot2::annotate("text", x = (x1 + x2) / 2, y = y_pos + (y_span * 0.02), label = label, size = 5, fontface = "bold")
  }
  p_out
}

for (feature in eeg_features) {
  p <- ggplot2::ggplot(eeg_behavioral_dataset, ggplot2::aes(x = cluster_group_3, y = .data[[feature]], fill = cluster_group_3)) +
    ggplot2::geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.2, alpha = 0.6, size = 1) +
    ggplot2::labs(title = paste("Distribution of", feature, "by Group (0, 1_2, 3)"), x = "Group", y = feature) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none", plot.title = ggplot2::element_text(size = 12, face = "bold"), axis.text = ggplot2::element_text(size = 10), axis.title = ggplot2::element_text(size = 11))
  p_annot <- add_significance_annotations_group3(p, feature, eeg_behavioral_dataset)
  print(p_annot)
}
dev.off()








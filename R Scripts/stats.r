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
# CLUSTERS WITH KMEANS (+ CONTROLS)
# original_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_cluster_db.csv")

# CLUSTERS WITH KMEANS (ONLY ADHD/ASD)
# original_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_ASD_ADHD_cluster_db.csv")

# CLUSTER WITH GMM (ONLY ASD/ADHD)
original_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_ASD_ADHD_gfmm_cluster_db.csv")

og_dataset_copy <- original_dataset # Make a copy for preprocessing

########################### Preprocessing ############################

#### Plausibilité des scores ###
# Check frequencies
sink(file="RAW_freq_output.txt")
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
  og_dataset_copy$highest_education_level == "" ~ NA_character_,
  TRUE ~ og_dataset_copy$highest_education_level
)
og_dataset_copy$highest_education_level_recoded <- factor(og_dataset_copy$highest_education_level_recoded)

# Check frequencies
sink(file="RAW_freq_output.txt")
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
  og_dataset_copy$family_ethnicity == "" ~ NA_character_,
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
write.csv(og_dataset_copy, file = "FINAL_DATABASE_USED_IN_R_GMM.csv", row.names = FALSE)

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

################# Variability in numerical data ######################

# Age
min_age <- min(og_dataset_copy$age, na.rm = TRUE) # 5
mean_age_cluster0 <- mean(
  og_dataset_copy$age[og_dataset_copy$cluster == 0],
  na.rm = TRUE
) # 10.75
mean_age_cluster1 <- mean(
  og_dataset_copy$age[og_dataset_copy$cluster == 1],
  na.rm = TRUE
) # 10.31
max_age <- max(og_dataset_copy$age, na.rm = TRUE) # 18

# IQ
min_iq <- min(og_dataset_copy$IQ, na.rm = TRUE) # 30
mean_iq_cluster0 <- mean(
  og_dataset_copy$IQ[og_dataset_copy$cluster == 0],
  na.rm = TRUE
) # 96
mean_iq_cluster1 <- mean(
  og_dataset_copy$IQ[og_dataset_copy$cluster == 1],
  na.rm = TRUE
) # 82
max_iq <- max(og_dataset_copy$IQ, na.rm = TRUE) # 132

# Behavioral Scores

# SRS Social Cognition T-score
min_srs_social_cognition <- min(og_dataset_copy$SRS_social_cognition_tscore, na.rm = TRUE) # 39
mean_srs_social_cognition_cluster0 <- mean(
  og_dataset_copy$SRS_social_cognition_tscore[og_dataset_copy$cluster == 0],
  na.rm = TRUE
)
mean_srs_social_cognition_cluster1 <- mean(
  og_dataset_copy$SRS_social_cognition_tscore[og_dataset_copy$cluster == 1],
  na.rm = TRUE
)
max_srs_social_cognition <- max(og_dataset_copy$SRS_social_cognition_tscore, na.rm = TRUE) # 90

# SRS Social Communication T-score
min_srs_social_communication <- min(og_dataset_copy$SRS_social_communication_tscore, na.rm = TRUE)
mean_srs_social_communication_cluster0 <- mean(
  og_dataset_copy$SRS_social_communication_tscore[og_dataset_copy$cluster == 0],
  na.rm = TRUE
)
mean_srs_social_communication_cluster1 <- mean(
  og_dataset_copy$SRS_social_communication_tscore[og_dataset_copy$cluster == 1],
  na.rm = TRUE
)
max_srs_social_communication <- max(og_dataset_copy$SRS_social_communication_tscore, na.rm = TRUE)

# SRS Restrictive Repetitive T-score
min_srs_restrictive_repetitive <- min(og_dataset_copy$SRS_restrictive_repetitive_tscore, na.rm = TRUE)
mean_srs_restrictive_repetitive_cluster0 <- mean(
  og_dataset_copy$SRS_restrictive_repetitive_tscore[og_dataset_copy$cluster == 0],
  na.rm = TRUE
)
mean_srs_restrictive_repetitive_cluster1 <- mean(
  og_dataset_copy$SRS_restrictive_repetitive_tscore[og_dataset_copy$cluster == 1],
  na.rm = TRUE
)
max_srs_restrictive_repetitive <- max(og_dataset_copy$SRS_restrictive_repetitive_tscore, na.rm = TRUE)

# ADHD T-score
min_adhd <- min(og_dataset_copy$attention_deficit_hyperactivity_tscore, na.rm = TRUE)
mean_adhd_cluster0 <- mean(
  og_dataset_copy$attention_deficit_hyperactivity_tscore[og_dataset_copy$cluster == 0],
  na.rm = TRUE
)
mean_adhd_cluster1 <- mean(
  og_dataset_copy$attention_deficit_hyperactivity_tscore[og_dataset_copy$cluster == 1],
  na.rm = TRUE
)
max_adhd <- max(og_dataset_copy$attention_deficit_hyperactivity_tscore, na.rm = TRUE)

### Normality

# Plot data to help visualize
columns_to_plot <- c("age", "IQ", "SRS_restrictive_repetitive_tscore","SRS_social_communication_tscore", "SRS_social_cognition_tscore", "attention_deficit_hyperactivity_tscore")

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
                  vars = vars("age", "IQ", "SRS_restrictive_repetitive_tscore","SRS_social_communication_tscore", "SRS_social_cognition_tscore", "attention_deficit_hyperactivity_tscore"),
                  sd = TRUE, range = TRUE, 
                  skew = TRUE, kurt = TRUE)
sink()
sink(NULL)

# SRS_restrictive_repetitive_tscore    SRS_social_communication_tscore    SRS_social_cognition_tscore    attention_deficit_hyperactivity_tscore
# have Kurtosis index in between -1.3 and -1
# With sample size N = 130, Central Limit Theorem kicks in - no need to transform for mild deviations

########## MANCOVA 
# --- Setup
df <- og_dataset_copy
df$cluster <- factor(df$cluster, levels = c(0, 1, 2, 3))
df$sex     <- factor(df$sex)

DVS <- c(
  "SRS_restrictive_repetitive_tscore",
  "SRS_social_communication_tscore",
  "SRS_social_cognition_tscore",
  "attention_deficit_hyperactivity_tscore"
)

# --- MANCOVA (multivariate test)
mancova_model <- manova(cbind(
  SRS_restrictive_repetitive_tscore,
  SRS_social_communication_tscore,
  SRS_social_cognition_tscore,
  attention_deficit_hyperactivity_tscore
) ~ cluster + age + IQ + sex, data = df)

summary(mancova_model, test = "Pillai")  # robust
summary(mancova_model, test = "Wilks")   # traditional

# --- Univariate ANCOVAs per DV (safer & clearer for effect sizes)
# install.packages("effectsize"); install.packages("emmeans")  # if needed
library(effectsize)
library(emmeans)

fit_one <- function(y) {
  f <- as.formula(paste(y, "~ cluster + age + IQ + sex"))
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

# Plot a pie chart for each cluster showing the diagnostic group distribution
# (no labels on the pie sections, only the pie and legend)

# Ensure the variables are numeric (if not already)
og_dataset_copy$ADHD <- as.numeric(og_dataset_copy$ADHD)
og_dataset_copy$ASD <- as.numeric(og_dataset_copy$ASD)
og_dataset_copy$ASD_behavior <- as.numeric(og_dataset_copy$ASD_behavior)
og_dataset_copy$`No.ASD.ADHD` <- as.numeric(og_dataset_copy$`No.ASD.ADHD`)
og_dataset_copy$cluster <- as.factor(og_dataset_copy$cluster)

library(ggplot2)
library(dplyr)
library(tidyr)

# For each cluster, determine the diagnostic group for each participant
diagnosis_labels <- function(row) {
  asd <- !is.na(row["ASD"]) && row["ASD"] == 1
  asd_behavior <- !is.na(row["ASD_behavior"]) && row["ASD_behavior"] == 1
  adhd <- !is.na(row["ADHD"]) && row["ADHD"] == 1

  if (asd && adhd) {
    return("ASD + ADHD")
  } else if (asd) {
    return("ASD")
  } else if (asd_behavior && adhd) {
    return("ADHD + ASD behavior")
  } else if (adhd) {
    return("ADHD")
  } else {
    return("No ASD/ADHD")
  }
}

og_dataset_copy$diagnosis_group <- apply(og_dataset_copy[, c("ASD", "ASD_behavior", "ADHD")], 1, diagnosis_labels)

# Do not enforce a fixed order for diagnosis groups
og_dataset_copy$diagnosis_group <- factor(og_dataset_copy$diagnosis_group)

# Custom color palette for diagnosis groups (more distinct colors)
diagnosis_colors <- c(
  "ASD" = "#1976D2",              # deep blue
  "ADHD" = "#E91E63",             # pink
  "ASD + ADHD" = "#8E24AA",            # purple
  "ADHD + ASD behavior" = "#BA68C8"   # light purple
)

clusters <- levels(og_dataset_copy$cluster)

for (cl in clusters) {
  cluster_data <- og_dataset_copy %>% filter(cluster == cl)
  total_n <- nrow(cluster_data)
  diag_counts <- cluster_data %>%
    group_by(diagnosis_group) %>%
    summarise(count = n(), .groups = "drop") %>%
    mutate(perc = count / sum(count) * 100)

 # Ensure all diagnosis groups are present (even if count = 0)
  diag_counts <- diag_counts %>%
    complete(diagnosis_group = levels(og_dataset_copy$diagnosis_group), fill = list(count = 0, perc = 0))
  # Remove groups with count == 0 so they don't get plotted as tiny slivers
  diag_counts <- diag_counts %>% filter(count > 0)

  # Plot pie chart with labels positioned outside the chart area
  # Add labels with n (%) outside the pie slices

  diag_counts <- diag_counts %>%
    mutate(
      label = ifelse(
        count > 0,
        paste0(count, " (", sprintf("%.1f", perc), "%)"),
        ""
      )
    )

  # Calculate cumulative proportions for label positioning
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
  pie_chart <- ggplot(diag_counts, aes(x = "", y = count, fill = diagnosis_group)) +
    geom_col(width = 1, color = "white") +
    coord_polar(theta = "y", start = 0) +
    labs(
      title = paste0("Cluster ", cl, " (n = ", total_n, ") - Diagnostic Distribution"),
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
        label = label,
        angle = angle,
        hjust = hjust
      ),
      x = 1.1, # position labels outside the pie
      size = 3,
      inherit.aes = FALSE
    )

  print(pie_chart)
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


eeg_behavioral_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/merged_EEG_behavioral_data.csv")

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
eeg_features <- c("Aperiodic_Offset", "Aperiodic_Exponent", 
                  "Average_Delta_Power", "Average_Theta_Power", "Average_Alpha_Power", 
                  "Average_Beta_Power", "Average_Gamma_Power",
                  "Average_PeriodicPSD_Delta", "Average_PeriodicPSD_Theta", 
                  "Average_PeriodicPSD_Alpha", "Average_PeriodicPSD_Beta", 
                  "Average_PeriodicPSD_Gamma",
                  "Average_RelDelta_Power", "Average_RelTheta_Power", 
                  "Average_RelAlpha_Power", "Average_RelBeta_Power", "Average_RelGamma_Power")

# Convert cluster to factor
eeg_behavioral_dataset$cluster <- as.factor(eeg_behavioral_dataset$cluster)

# Create output file for EEG analysis
sink("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/eeg_analysis_output.txt")

cat("========================================\n")
cat("EEG FEATURE ANALYSIS\n")
cat("========================================\n\n")

# 1. DESCRIPTIVE STATISTICS BY CLUSTER
cat("1. DESCRIPTIVE STATISTICS BY CLUSTER\n")
cat("=====================================\n\n")

# Overall descriptives
cat("Overall Sample Size:", nrow(eeg_behavioral_dataset), "\n")
cat("Cluster Distribution:\n")
print(table(eeg_behavioral_dataset$cluster))
cat("\n")

# Descriptive statistics by cluster
for(cluster_id in levels(eeg_behavioral_dataset$cluster)) {
  cat("CLUSTER", cluster_id, "DESCRIPTIVES:\n")
  cat("----------------------------------------\n")
  cluster_data <- eeg_behavioral_dataset[eeg_behavioral_dataset$cluster == cluster_id, eeg_features]
  desc_stats <- describe(cluster_data)
  print(desc_stats)
  cat("\n")
}

# 2. ASSUMPTION TESTING FOR PARAMETRIC TESTS
cat("2. ASSUMPTION TESTING\n")
cat("=====================\n\n")

# Test normality for each EEG feature by cluster
cat("Shapiro-Wilk Tests for Normality (by cluster):\n")
cat("(Note: p < 0.05 indicates non-normality)\n\n")

normality_results <- data.frame()
for(feature in eeg_features) {
  for(cluster_id in levels(eeg_behavioral_dataset$cluster)) {
    cluster_data <- eeg_behavioral_dataset[eeg_behavioral_dataset$cluster == cluster_id, feature]
    if(length(cluster_data) >= 3) {  # Minimum sample size for Shapiro-Wilk
      shapiro_test <- shapiro.test(cluster_data)
      normality_results <- rbind(normality_results, 
                                data.frame(Feature = feature, 
                                         Cluster = cluster_id, 
                                         W = shapiro_test$statistic, 
                                         p_value = shapiro_test$p.value))
    }
  }
}
print(normality_results)
cat("\n")

# Levene's test for homogeneity of variance
cat("Levene's Test for Homogeneity of Variance:\n")
cat("(p < 0.05 indicates unequal variances)\n\n")

levene_results <- data.frame()
for(feature in eeg_features) {
  levene_test <- leveneTest(eeg_behavioral_dataset[[feature]] ~ eeg_behavioral_dataset$cluster)
  levene_results <- rbind(levene_results, 
                         data.frame(Feature = feature, 
                                  F = levene_test$`F value`[1], 
                                  p_value = levene_test$`Pr(>F)`[1]))
}
print(levene_results)
cat("\n")

# 3. CLUSTER DIFFERENCES IN EEG FEATURES
cat("3. CLUSTER DIFFERENCES IN EEG FEATURES\n")
cat("======================================\n\n")

# One-way ANOVA for each EEG feature
cat("One-way ANOVA Results for Each EEG Feature:\n")
cat("(p < 0.05 indicates significant differences between clusters)\n\n")

anova_results <- data.frame()
for(feature in eeg_features) {
  formula_str <- paste(feature, "~ cluster")
  anova_model <- aov(as.formula(formula_str), data = eeg_behavioral_dataset)
  anova_summary <- summary(anova_model)
  
  anova_results <- rbind(anova_results, 
                        data.frame(Feature = feature, 
                                 F = anova_summary[[1]]$`F value`[1], 
                                 p_value = anova_summary[[1]]$`Pr(>F)`[1]))
}
print(anova_results)
cat("\n")

# Post-hoc pairwise comparisons (Tukey HSD)
cat("Post-hoc Pairwise Comparisons (Tukey HSD):\n")
cat("==========================================\n\n")

for(feature in eeg_features) {
  if(anova_results[anova_results$Feature == feature, "p_value"] < 0.05) {
    cat("Significant differences found for", feature, ":\n")
    formula_str <- paste(feature, "~ cluster")
    anova_model <- aov(as.formula(formula_str), data = eeg_behavioral_dataset)
    tukey_results <- TukeyHSD(anova_model)
    print(tukey_results)
    cat("\n")
  }
}

# MANOVA for all EEG features together
cat("4. MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)\n")
cat("=============================================\n\n")

# Create formula for MANOVA
manova_formula <- as.formula(paste("cbind(", paste(eeg_features, collapse = ", "), ") ~ cluster"))

# Perform MANOVA
manova_model <- manova(manova_formula, data = eeg_behavioral_dataset)
manova_summary <- summary(manova_model)

cat("MANOVA Results:\n")
print(manova_summary)

# Pillai's trace test
cat("\nPillai's Trace Test:\n")
print(summary(manova_model, test = "Pillai"))

# Wilks' Lambda test
cat("\nWilks' Lambda Test:\n")
print(summary(manova_model, test = "Wilks"))

# Hotelling-Lawley trace test
cat("\nHotelling-Lawley Trace Test:\n")
print(summary(manova_model, test = "Hotelling-Lawley"))

# Roy's largest root test
cat("\nRoy's Largest Root Test:\n")
print(summary(manova_model, test = "Roy"))

cat("\n")

# 5. FEATURE IMPORTANCE ANALYSIS
cat("5. FEATURE IMPORTANCE ANALYSIS\n")
cat("==============================\n\n")

# Random Forest for feature importance
cat("Random Forest Feature Importance:\n")
cat("(Higher values indicate more important features for cluster prediction)\n\n")

# Prepare data for random forest
rf_data <- eeg_behavioral_dataset[, c("cluster", eeg_features)]
rf_data <- rf_data[complete.cases(rf_data), ]  # Remove rows with missing values

# Train random forest
rf_model <- randomForest(cluster ~ ., data = rf_data, importance = TRUE, ntree = 1000)

# Extract feature importance
importance_scores <- importance(rf_model)
importance_df <- data.frame(
  Feature = rownames(importance_scores),
  MeanDecreaseAccuracy = importance_scores[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = importance_scores[, "MeanDecreaseGini"]
)

# Sort by importance
importance_df <- importance_df[order(importance_df$MeanDecreaseAccuracy, decreasing = TRUE), ]
print(importance_df)
cat("\n")

# 6. LOW PROBABILITY CLUSTER ANALYSIS
cat("6. LOW PROBABILITY CLUSTER ANALYSIS\n")
cat("===================================\n\n")

# Define low probability threshold (e.g., max probability < 0.7)
eeg_behavioral_dataset$max_cluster_prob <- pmax(
  eeg_behavioral_dataset$cluster_0_prob,
  eeg_behavioral_dataset$cluster_1_prob,
  eeg_behavioral_dataset$cluster_2_prob,
  eeg_behavioral_dataset$cluster_3_prob
)

eeg_behavioral_dataset$low_confidence <- eeg_behavioral_dataset$max_cluster_prob < 0.7

cat("Low Confidence Classifications (max probability < 0.7):\n")
print(table(eeg_behavioral_dataset$low_confidence))
cat("\n")

# Compare EEG features between high and low confidence classifications
cat("EEG Feature Differences: High vs Low Confidence Classifications\n")
cat("(t-tests for each feature)\n\n")

confidence_comparison <- data.frame()
for(feature in eeg_features) {
  high_conf <- eeg_behavioral_dataset[!eeg_behavioral_dataset$low_confidence, feature]
  low_conf <- eeg_behavioral_dataset[eeg_behavioral_dataset$low_confidence, feature]
  
  if(length(high_conf) > 1 && length(low_conf) > 1) {
    t_test <- t.test(high_conf, low_conf)
    confidence_comparison <- rbind(confidence_comparison, 
                                 data.frame(Feature = feature, 
                                          High_Conf_Mean = mean(high_conf, na.rm = TRUE),
                                          Low_Conf_Mean = mean(low_conf, na.rm = TRUE),
                                          t_statistic = t_test$statistic,
                                          p_value = t_test$p.value))
  }
}
print(confidence_comparison)
cat("\n")

# 7. CORRELATION ANALYSIS
cat("7. CORRELATION ANALYSIS\n")
cat("=======================\n\n")

# Correlation matrix of EEG features
eeg_correlations <- cor(eeg_behavioral_dataset[, eeg_features], use = "complete.obs")
cat("Correlation Matrix of EEG Features:\n")
print(round(eeg_correlations, 3))
cat("\n")

# Find highly correlated features (|r| > 0.7)
high_correlations <- which(abs(eeg_correlations) > 0.7 & eeg_correlations != 1, arr.ind = TRUE)
if(nrow(high_correlations) > 0) {
  cat("Highly Correlated Features (|r| > 0.7):\n")
  for(i in 1:nrow(high_correlations)) {
    row_idx <- high_correlations[i, 1]
    col_idx <- high_correlations[i, 2]
    cat(rownames(eeg_correlations)[row_idx], "&", 
        colnames(eeg_correlations)[col_idx], 
        "r =", round(eeg_correlations[row_idx, col_idx], 3), "\n")
  }
} else {
  cat("No highly correlated features found (|r| > 0.7)\n")
}
cat("\n")

sink()
sink(NULL)

# 8. VISUALIZATIONS
cat("Creating visualizations...\n")

# Create output directory for plots if it doesn't exist
if(!dir.exists("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots")) {
  dir.create("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots", recursive = TRUE)
}

# 8.1 Box plots for each EEG feature by cluster
pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/eeg_features_by_cluster_boxplots.pdf", 
    width = 12, height = 8)

# Create box plots for each EEG feature
for(feature in eeg_features) {
  p <- ggplot(eeg_behavioral_dataset, aes(x = cluster, y = .data[[feature]], fill = cluster)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.5) +
    labs(title = paste("Distribution of", feature, "by Cluster"),
         x = "Cluster", 
         y = feature) +
    theme_minimal() +
    theme(legend.position = "none")
  print(p)
}

dev.off()

# 8.2 Correlation heatmap
pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/eeg_correlation_heatmap.pdf", 
    width = 12, height = 10)

corrplot(eeg_correlations, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black",
         title = "EEG Features Correlation Matrix")

dev.off()

# 8.3 Feature importance plot
pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/feature_importance.pdf", 
    width = 10, height = 8)

# Create feature importance plot
importance_plot <- ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseAccuracy), 
                                            y = MeanDecreaseAccuracy)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(title = "EEG Feature Importance for Cluster Prediction",
       x = "EEG Features",
       y = "Mean Decrease in Accuracy") +
  theme_minimal()

print(importance_plot)
dev.off()

# 8.4 Cluster probability distributions
pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/cluster_probability_distributions.pdf", 
    width = 12, height = 8)

# Plot distribution of maximum cluster probabilities
prob_plot <- ggplot(eeg_behavioral_dataset, aes(x = max_cluster_prob)) +
  geom_histogram(bins = 20, fill = "lightblue", alpha = 0.7, color = "black") +
  geom_vline(xintercept = 0.7, color = "red", linetype = "dashed", size = 1) +
  labs(title = "Distribution of Maximum Cluster Probabilities",
       x = "Maximum Cluster Probability",
       y = "Frequency") +
  theme_minimal()

print(prob_plot)

# Plot cluster probabilities by actual cluster assignment
cluster_prob_data <- eeg_behavioral_dataset %>%
  select(cluster, cluster_0_prob, cluster_1_prob, cluster_2_prob, cluster_3_prob) %>%
  pivot_longer(cols = starts_with("cluster_"), 
               names_to = "prob_cluster", 
               values_to = "probability") %>%
  mutate(prob_cluster = gsub("cluster_", "", prob_cluster),
         prob_cluster = gsub("_prob", "", prob_cluster))

prob_by_cluster_plot <- ggplot(cluster_prob_data, aes(x = prob_cluster, y = probability, fill = cluster)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Cluster Probabilities by Actual Cluster Assignment",
       x = "Probability Cluster",
       y = "Probability",
       fill = "Actual Cluster") +
  theme_minimal()

print(prob_by_cluster_plot)
dev.off()

# 8.5 Scatter plots for top important features
pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/top_features_scatter.pdf", 
    width = 12, height = 10)

# Get top 6 most important features
top_features <- head(importance_df$Feature, 6)

# Create scatter plot matrix for top features
if(length(top_features) >= 2) {
  pairs_plot <- ggpairs(eeg_behavioral_dataset[, c("cluster", top_features)], 
                        aes(color = cluster, alpha = 0.6),
                        lower = list(continuous = "points"),
                        upper = list(continuous = "cor"),
                        diag = list(continuous = "densityDiag"))
  print(pairs_plot)
}

dev.off()

cat("Visualizations saved to /Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/R_output/plots/\n")
cat("Analysis complete!\n")














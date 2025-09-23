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

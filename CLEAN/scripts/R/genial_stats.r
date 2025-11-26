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
TIMESTAMP = 'NOV26'
database_filepath = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/merged_clustered_EEG_features_RSRio_NOV_24_2025.csv"
isROIAnalysis = FALSE
analysis_label = if(isROIAnalysis) "ROI" else "GLOBAL"

base_eeg_features <- c('hurst_2s', 
                       'pow_delta_2s', 'pow_theta_2s', 'pow_alpha_2s', 'pow_beta_2s', 'pow_gamma_2s', 'pow_low_gamma_2s', 'pow_high_gamma_2s',
                       'pow_per_delta_2s', 'pow_per_theta_2s', 'pow_per_alpha_2s', 'pow_per_gamma_2s', 'pow_per_low_gamma_2s', 'pow_per_high_gamma_2s',
                       'higuchi_fd_5s', 'katz_fd_5s',
                       'samp_entropy_5s', 'CI_5s', 'CI_lowscale_5s', 'CI_highscale_5s')
base_eeg_2s_features <- c('hurst_2s', 
                          'pow_delta_2s', 'pow_theta_2s', 'pow_alpha_2s', 'pow_beta_2s', 'pow_gamma_2s', 'pow_low_gamma_2s', 'pow_high_gamma_2s',
                          'pow_per_delta_2s', 'pow_per_theta_2s', 'pow_per_alpha_2s', 'pow_per_gamma_2s', 'pow_per_low_gamma_2s', 'pow_per_high_gamma_2s')

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

pdf(file.path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/", paste0("scatter_plots_", analysis_label, "_", TIMESTAMP, ".pdf")))
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
pdf(file.path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/", paste0("histograms_", analysis_label, "_", TIMESTAMP, ".pdf")), width = 10, height = 7)
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
          file.path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/", paste0("normality_tests_", analysis_label, "_", TIMESTAMP, ".csv")),
          row.names = FALSE)

####################################################################
############################ ANOVA #################################
####################################################################
df <- db_copy
df$cluster <- factor(df$cluster, levels = c(0, 1, 2, 3))
df$sex <- factor(df$sex)
covariates <- c("age_at_test", "sex")

# ---- Feature-wise ANCOVA for each EEG feature ----#
sink(file.path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/", paste0("EEG_features_statsmodel_summaries_", analysis_label, "_", TIMESTAMP, ".txt")))
results <- lapply(eeg_features, function(feat) {
  cat("\n======================\n")
  cat("Feature:", feat, "\n")
  cat("======================\n\n")
  
  dat_feat <- df[, c("cluster", "age_at_test", "sex", feat)]
  dat_feat <- na.omit(dat_feat) # Remove rows with NA
  
  dat_feat$cluster <- droplevels(dat_feat$cluster)
  dat_feat$sex     <- droplevels(dat_feat$sex)
  
  # If cluster has fewer than 2 levels, we cannot fit the full model
  if (nlevels(dat_feat$cluster) < 2) {
    cat("Skipped: cluster has fewer than 2 levels after NA removal\n")
    return(data.frame(feature = feat, p_value = NA_real_))
  }
  
  # If sex has only 1 level, drop sex from the model for this feature
  if (nlevels(dat_feat$sex) < 2) {
    cat("Warning: sex has only 1 level after NA removal, fitting model without sex\n")
    form <- as.formula(paste(feat, "~ cluster + age_at_test"))
  } else {
    form <- as.formula(paste(feat, "~ cluster + age_at_test + sex"))
  }
  
  # Fit ANOVA
  fit <- lm(form, data = dat_feat)
  an <- anova(fit)
  
  print(summary(fit))
  
  # Extract the row of the cluster effect
  p_val <- an["cluster", "Pr(>F)"]
  
  data.frame(
    feature = feat,
    p_value = p_val
  )
})
sink()
results <- bind_rows(results)

# Table of significant features
signif_feats <- results |>
  dplyr::filter(!is.na(p_value) & p_value < 0.05) |>
  dplyr::pull(feature)

sink(file.path(
  "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/",
  paste0("EEG_features_statsmodel_summaries_SIGNIF_", analysis_label, "_", TIMESTAMP, ".txt")
))

for (feat in signif_feats) {
  cat("\n======================\n")
  cat("Feature:", feat, "\n")
  cat("======================\n\n")
  
  dat_feat <- df[, c("cluster", "age_at_test", "sex", feat)]
  dat_feat <- na.omit(dat_feat)
  dat_feat$cluster <- droplevels(dat_feat$cluster)
  dat_feat$sex     <- droplevels(dat_feat$sex)
  
  if (nlevels(dat_feat$sex) < 2) {
    form <- as.formula(paste(feat, "~ cluster + age_at_test"))
  } else {
    form <- as.formula(paste(feat, "~ cluster + age_at_test + sex"))
  }
  
  fit <- lm(form, data = dat_feat)
  an  <- anova(fit)
  
  print(summary(fit))
  cat("\nANOVA table:\n")
  print(an)
}

sink()


# ---- BOX PLOTS ---- #
out_dir <- "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/"

pdf(file.path(out_dir,
              paste0("boxplots_sig_features_with_stars_", analysis_label, "_", TIMESTAMP, ".pdf")),
    width = 7, height = 5)

for (feat in signif_feats) {
  
  dat_feat <- df[, c("cluster", "age_at_test", "sex", feat)]
  dat_feat <- na.omit(dat_feat)
  dat_feat$cluster <- droplevels(factor(dat_feat$cluster))
  dat_feat$sex     <- droplevels(factor(dat_feat$sex))
  
  if (nrow(dat_feat) == 0 || nlevels(dat_feat$cluster) < 2) next
  
  if (nlevels(dat_feat$sex) < 2) {
    form <- as.formula(paste(feat, "~ cluster + age_at_test"))
  } else {
    form <- as.formula(paste(feat, "~ cluster + age_at_test + sex"))
  }
  
  fit <- lm(form, data = dat_feat)
  
  emm   <- emmeans::emmeans(fit, "cluster")
  pw    <- pairs(emm, adjust = "none")
  pw_df <- as.data.frame(pw)
  
  sig_pw <- pw_df %>% dplyr::filter(p.value < 0.05)
  
  n_total <- nrow(dat_feat)
  
  y_max   <- max(dat_feat[[feat]], na.rm = TRUE)
  y_min   <- min(dat_feat[[feat]], na.rm = TRUE)
  y_range <- y_max - y_min
  if (y_range == 0) y_range <- 1
  
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
    theme_bw()
  
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
# 1. Load necessary libraries
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("tidyr")) install.packages("tidyr")
if (!require("dplyr")) install.packages("dplyr")

library(ggplot2)
library(tidyr)
library(dplyr)

# =================
# 1. Load database
# =================
# (Keep your file path exactly as is)
database_filepath = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/OUTPUTS/Final/merged_clustered_EEG_features_global_RSRio_DEC_16_2025_logtransformed.csv"
db <- read.csv(database_filepath)
db_copy <- db

# ==============================================================================
# 2. Dynamic Data Generation (Percentile Conversion)
# ==============================================================================

# FIRST: Convert scores to clinical percentiles for each participant
# T-scores: Mean=50, SD=10
# IQ: Mean=100, SD=15
plot_db_prep <- db_copy %>%
  mutate(
    # Convert T-scores to Percentiles
    RRB_pct    = pnorm(SRS_restrictive_repetitive_tscore, mean = 50, sd = 10) * 100,
    S_Com_pct  = pnorm(SRS_social_communication_tscore, mean = 50, sd = 10) * 100,
    S_Cogn_pct = pnorm(SRS_social_cognition_tscore, mean = 50, sd = 10) * 100,
    ADHD_pct   = pnorm(attention_deficit_hyperactivity_tscore, mean = 50, sd = 10) * 100,
    
    # Convert IQ to Percentiles
    NVIQ_pct   = pnorm(nonverbal_iq, mean = 100, sd = 15) * 100
  )

# SECOND: Aggregate by Cluster using the new Percentile columns
plot_db <- plot_db_prep %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    
    # --- Calculate Means of Percentiles ---
    RRB    = mean(RRB_pct, na.rm = TRUE),
    S_Com  = mean(S_Com_pct, na.rm = TRUE),
    S_Cogn = mean(S_Cogn_pct, na.rm = TRUE),
    ADHD   = mean(ADHD_pct, na.rm = TRUE),
    NVIQ   = mean(NVIQ_pct, na.rm = TRUE),
    
    # --- Calculate Standard Deviations of Percentiles ---
    RRB_SD    = sd(RRB_pct, na.rm = TRUE),
    S_Com_SD  = sd(S_Com_pct, na.rm = TRUE),
    S_Cogn_SD = sd(S_Cogn_pct, na.rm = TRUE),
    ADHD_SD   = sd(ADHD_pct, na.rm = TRUE),
    NVIQ_SD   = sd(NVIQ_pct, na.rm = TRUE),
    
    .groups = 'drop'
  ) %>%
  mutate(Cluster = paste("Cluster", cluster))

# ==============================================================================
# 3. Data Wrangling
# ==============================================================================

# Pivot Means
df_means <- plot_db %>%
  select(Cluster, n, RRB, S_Com, S_Cogn, ADHD, NVIQ) %>%
  pivot_longer(cols = c(RRB, S_Com, S_Cogn, ADHD, NVIQ), 
               names_to = "Measure", values_to = "Mean")

# Pivot SDs
df_sds <- plot_db %>%
  select(Cluster, ends_with("_SD")) %>%
  pivot_longer(cols = -Cluster, 
               names_to = "Measure", values_to = "SD") %>%
  mutate(Measure = gsub("_SD", "", Measure))

# Join them together
plot_data <- left_join(df_means, df_sds, by = c("Cluster", "Measure"))

# --- Rename for Legend ---
plot_data <- plot_data %>%
  mutate(Measure = recode(Measure,
                          "RRB" = "Restrictive/Repetitive Behavior",
                          "S_Com" = "Social Communication",
                          "S_Cogn" = "Social Cognition",
                          "ADHD" = "ADHD",
                          "NVIQ" = "NVIQ"))

# Set Factor Levels
plot_data$Measure <- factor(plot_data$Measure, 
                            levels = c("Restrictive/Repetitive Behavior", 
                                       "Social Communication", 
                                       "Social Cognition", 
                                       "ADHD", 
                                       "NVIQ"))

# Create custom X-axis labels
plot_data$ClusterLabel <- paste0(plot_data$Cluster, "\n(n=", plot_data$n, ")")

# ==============================================================================
# 4. Generate the Plot
# ==============================================================================
p <- ggplot(plot_data, aes(x = ClusterLabel, y = Mean, fill = Measure)) +
  
  # Bars
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), 
           width = 0.7, color = "black", linewidth = 0.3) +
  
  # Error Bars
  geom_errorbar(aes(ymin = pmax(0, Mean - SD), ymax = pmin(100, Mean + SD)),
                position = position_dodge(width = 0.8), 
                width = 0.25, linewidth = 0.4) +
  
  # SD Labels (using round instead of sprintf for cleaner percentile look)
  geom_text(aes(label = round(SD, 1), y = pmin(100, Mean + SD)), 
            position = position_dodge(width = 0.8), 
            vjust = -0.5, 
            size = 3) +
  
  # Styling
  theme_classic(base_size = 14) +
  scale_fill_brewer(palette = "Set2", name = "Measures:") +
  
  # Y-axis limited to 0-100 (Percentiles)
  coord_cartesian(ylim = c(0, 110)) + 
  scale_y_continuous(breaks = seq(0, 100, 20)) +
  
  labs(title = "Behavioral Profiles by Cluster (Percentiles)",
       x = NULL,
       y = "Percentile") +
  
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(color = "black"),
    axis.text.y = element_text(color = "black")
  )

# 5. Display
print(p)
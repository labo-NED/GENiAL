########################## Install packages ##########################
# install.packages('readxl')
# install.packages("sjmisc")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("tidyr")
# install.packages("stringr")
# install.packages("pheatmap")
# install.packages("tibble")

########################## Activate packages #########################
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(tibble)

########################## Import dataset ############################
# CLUSTERS WITH SOM (+ CONTROLS)
original_dataset <- read.csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/merged_clustered_behavioral_EEG_features_global_RSRio_DEC_16_2025.csv")

og_dataset_copy <- original_dataset # Make a copy for preprocessing

########################### Preprocessing ############################

# Convert cluster to factor
og_dataset_copy$cluster <- as.factor(og_dataset_copy$cluster)

# Function to parse diagnosis column and determine diagnostic group
diagnosis_labels <- function(diagnosis_string) {
  # Handle missing or empty values
  if (is.na(diagnosis_string) || diagnosis_string == "" || diagnosis_string == "None") {
    return("No diagnosis")
  }
  
  # Split by comma or semicolon and clean whitespace
  # Handle both "," and "; " as separators
  diagnosis_string <- gsub(";", ",", diagnosis_string)  # Replace semicolons with commas
  diagnoses <- strsplit(diagnosis_string, ",")[[1]]
  diagnoses <- trimws(diagnoses)
  diagnoses <- tolower(diagnoses)
  
  # Check for ASD, autism_behavior, and ADHD
  # Handle various possible labels
  has_asd <- any(diagnoses %in% c("autism", "asd"))
  has_asd_behavior <- any(diagnoses %in% c("autism_behavior", "autistic_behav", "autistic behavior"))
  has_adhd <- any(diagnoses %in% c("adhd", "attention_deficit_hyperactivity", "attention deficit hyperactivity"))
  
  # Determine diagnostic group
  if (has_asd && has_adhd) {
    return("ASD + ADHD")
  } else if (has_asd) {
    return("ASD")
  } else if (has_asd_behavior && has_adhd) {
    return("ADHD + ASD behavior")
  } else if (has_adhd) {
    return("ADHD")
  } else {
    return("Other diagnosis")
  }
}

# Apply the function to create diagnosis_group
og_dataset_copy$diagnosis_group <- sapply(og_dataset_copy$diagnosis, diagnosis_labels)

# Do not enforce a fixed order for diagnosis groups
og_dataset_copy$diagnosis_group <- factor(og_dataset_copy$diagnosis_group)

# Custom color palette for diagnosis groups (more distinct colors)
diagnosis_colors <- c(
  "ASD" = "#1976D2",              # deep blue
  "ADHD" = "#E91E63",             # pink
  "ASD + ADHD" = "#8E24AA",            # purple
  "ADHD + ASD behavior" = "#BA68C8",   # light purple
  "No diagnosis" = "#BDBDBD",          # light gray
  "Other diagnosis" = "#757575"        # medium gray
)

########################## Diagnoses Pie Charts -- FULL sample ##########################
cluster_data <- og_dataset_copy
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
pie_chart <- ggplot(diag_counts, aes(x = 1, y = count, fill = diagnosis_group)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y", start = 0) +
  xlim(c(0.5, 1.5)) +  # Fixed x limits to ensure consistent pie chart radius across all clusters
  labs(
    title = paste0("All Participants ", " (n = ", total_n, ") - Diagnostic Distribution"),
    fill = "Diagnosis"
  ) +
  theme_void(base_size = 16) +
  scale_fill_manual(values = diagnosis_colors, drop = FALSE) +
  guides(fill = guide_legend(override.aes = list(size = 4))) +
  theme(
    plot.title = element_text(hjust = 0, size = 12, face = "bold"),
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 8)
  ) +
  geom_text(
    data = diag_counts %>% filter(count > 0),
    aes(
      y = mid,
      label = label,
      angle = angle,
      hjust = hjust
    ),
    x = 1, # position labels outside the pie
    size = 3,
    color = "white",
    inherit.aes = FALSE
  )

# Save pie chart as PNG with fixed aspect ratio for the chart portion
ggsave(
  filename = paste0("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/diagnosis_pie_chart_full_sample.png"),
  plot = pie_chart,
  width = 8,
  height = 6,
  dpi = 300,
  limitsize = FALSE # ensure the size is always used, disables small chart clipping
)
# Also display in RStudio/interactive session
print(pie_chart)

########################## Pie Charts for Individual Clusters ##########################

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
  pie_chart <- ggplot(diag_counts, aes(x = 1, y = count, fill = diagnosis_group)) +
    geom_col(width = 1, color = "white") +
    coord_polar(theta = "y", start = 0) +
    xlim(c(0.5, 1.5)) +  # Fixed x limits to ensure consistent pie chart radius across all clusters
    labs(
      title = paste0("Cluster ", cl, " (n = ", total_n, ") - Diagnostic Distribution"),
      fill = "Diagnosis"
    ) +
    theme_void(base_size = 16) +
    scale_fill_manual(values = diagnosis_colors, drop = FALSE) +
    guides(fill = guide_legend(override.aes = list(size = 6))) +
    theme(
      plot.title = element_text(hjust = 0, size = 12, face = "bold"),
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 8)
    ) +
    geom_text(
      data = diag_counts %>% filter(count > 0),
      aes(
        y = mid,
        label = label,
        angle = angle,
        hjust = hjust
      ),
      x = 1, # position labels outside the pie
      size = 5,
      color = "white",
      inherit.aes = FALSE
    )

  # Save pie chart as PNG with fixed aspect ratio for the chart portion
  ggsave(
    filename = paste0("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Clustered/diagnosis_pie_chart_cluster_", cl, ".png"),
    plot = pie_chart,
    width = 8,
    height = 6,
    dpi = 300,
    limitsize = FALSE # ensure the size is always used, disables small chart clipping
  )
  # Also display in RStudio/interactive session
  print(pie_chart)
}

########################## Diagnosis Frequency Analysis by Cluster ##########################

# Function to parse all individual diagnoses from a diagnosis string
parse_individual_diagnoses <- function(diagnosis_string) {
  # Handle missing or empty values
  if (is.na(diagnosis_string) || diagnosis_string == "" || diagnosis_string == "None") {
    return(character(0))
  }
  
  # Split by comma or semicolon and clean whitespace
  diagnosis_string <- gsub(";", ",", diagnosis_string)  # Replace semicolons with commas
  diagnoses <- strsplit(diagnosis_string, ",")[[1]]
  diagnoses <- trimws(diagnoses)
  diagnoses <- tolower(diagnoses)
  
  # Return non-empty diagnoses
  return(diagnoses[diagnoses != ""])
}

# Create a long-format dataset with one row per diagnosis per participant
# First, identify participants with no diagnosis
participants_with_diagnosis <- og_dataset_copy %>%
  rowwise() %>%
  mutate(
    individual_diagnoses = list(parse_individual_diagnoses(diagnosis)),
    has_diagnosis = length(individual_diagnoses) > 0
  ) %>%
  ungroup()

# Count participants with no diagnosis per cluster
no_diagnosis_counts <- participants_with_diagnosis %>%
  filter(!has_diagnosis) %>%
  group_by(cluster) %>%
  summarise(count = n(), .groups = "drop") %>%
  mutate(
    individual_diagnoses = "no_diagnosis",
    diagnosis_label = "No Diagnosis"
  )

# Create long format for participants with diagnoses
diagnosis_long <- participants_with_diagnosis %>%
  filter(has_diagnosis) %>%
  unnest(individual_diagnoses) %>%
  select(cluster, individual_diagnoses) %>%
  filter(!is.na(individual_diagnoses) & individual_diagnoses != "")

# Count frequencies of each diagnosis per cluster (including "no_diagnosis")
diagnosis_freq_by_cluster <- diagnosis_long %>%
  group_by(cluster, individual_diagnoses) %>%
  summarise(count = n(), .groups = "drop") %>%
  bind_rows(
    no_diagnosis_counts %>% 
      select(cluster, individual_diagnoses, count) %>%
      mutate(individual_diagnoses = as.character(individual_diagnoses))
  ) %>%
  arrange(cluster, desc(count))

# Calculate percentages per cluster
cluster_totals <- og_dataset_copy %>%
  group_by(cluster) %>%
  summarise(total_participants = n(), .groups = "drop")

diagnosis_freq_by_cluster <- diagnosis_freq_by_cluster %>%
  left_join(cluster_totals, by = "cluster") %>%
  mutate(
    percentage = round((count / total_participants) * 100, 1),
    diagnosis_label = ifelse(
      individual_diagnoses == "no_diagnosis",
      "No Diagnosis",
      str_to_title(gsub("_", " ", individual_diagnoses))
    )
  ) %>%
  select(cluster, diagnosis_label, count, total_participants, percentage) %>%
  arrange(cluster, desc(count))

# Print summary table
cat("\n========================================\n")
cat("DIAGNOSIS FREQUENCY ANALYSIS BY CLUSTER\n")
cat("========================================\n\n")

for (cl in levels(og_dataset_copy$cluster)) {
  cluster_data <- diagnosis_freq_by_cluster %>% filter(cluster == cl)
  total_n <- unique(cluster_data$total_participants)
  sum_counts <- sum(cluster_data$count)
  
  cat(sprintf("CLUSTER %s (n = %d participants)\n", cl, total_n))
  cat("----------------------------------------\n")
  print(cluster_data %>% select(diagnosis_label, count, percentage), row.names = FALSE)
  cat(sprintf("\nSum of diagnosis counts: %d (should equal %d)\n", sum_counts, total_n))
  if (sum_counts == total_n) {
    cat("✓ Counts match total participants\n")
  } else {
    cat("⚠ WARNING: Counts do not match total participants!\n")
  }
  cat("\n")
}

# Create a wide-format table for easier comparison
diagnosis_freq_wide <- diagnosis_freq_by_cluster %>%
  select(cluster, diagnosis_label, count) %>%
  pivot_wider(
    names_from = cluster,
    values_from = count,
    values_fill = 0
  ) %>%
  arrange(desc(rowSums(select(., -diagnosis_label))))

# Save frequency tables
write.csv(
  diagnosis_freq_by_cluster,
  file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Clustered/diagnosis_frequency_by_cluster_long.csv",
  row.names = FALSE
)

write.csv(
  diagnosis_freq_wide,
  file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Clustered/diagnosis_frequency_by_cluster_wide.csv",
  row.names = FALSE
)

cat("Frequency tables saved to:\n")
cat("  - diagnosis_frequency_by_cluster_long.csv (long format)\n")
cat("  - diagnosis_frequency_by_cluster_wide.csv (wide format)\n\n")

# Create bar plot showing top diagnoses per cluster
top_n_diagnoses <- 10  # Number of top diagnoses to show per cluster

for (cl in levels(og_dataset_copy$cluster)) {
  cluster_all <- diagnosis_freq_by_cluster %>% filter(cluster == cl)
  
  # Separate "No Diagnosis" from other diagnoses
  no_diag <- cluster_all %>% filter(diagnosis_label == "No Diagnosis")
  others <- cluster_all %>% 
    filter(diagnosis_label != "No Diagnosis") %>%
    slice_max(count, n = top_n_diagnoses, with_ties = FALSE)
  
  # Combine: top others + "No Diagnosis" if present
  cluster_diag <- bind_rows(others, no_diag) %>%
    distinct(diagnosis_label, .keep_all = TRUE)
  
  if (nrow(cluster_diag) > 0) {
    p <- ggplot(cluster_diag, aes(x = reorder(diagnosis_label, count), y = count)) +
      geom_col(fill = "#1976D2", alpha = 0.8) +
      geom_text(aes(label = paste0(count, " (", percentage, "%)")), 
                hjust = -0.1, size = 3) +
      coord_flip() +
      labs(
        title = paste0("Top ", min(top_n_diagnoses, nrow(cluster_diag)), 
                      " Diagnoses in Cluster ", cl, " (n = ", unique(cluster_diag$total_participants), ")"),
        x = "Diagnosis",
        y = "Count (Percentage)"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 9)
      ) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
    
    ggsave(
      filename = paste0("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Clustered/diagnosis_frequency_cluster_", cl, ".png"),
      plot = p,
      width = 10,
      height = 6,
      dpi = 300
    )
    
    print(p)
  }
}

# Create a heatmap showing all diagnoses across all clusters
# Select top diagnoses by overall frequency for heatmap first
# Exclude "No Diagnosis" from heatmap as it's not a real diagnosis
top_diagnoses_overall <- diagnosis_freq_by_cluster %>%
  filter(diagnosis_label != "No Diagnosis") %>%
  group_by(diagnosis_label) %>%
  summarise(total_count = sum(count), .groups = "drop") %>%
  slice_max(total_count, n = 20) %>%
  pull(diagnosis_label)

diagnosis_freq_matrix <- diagnosis_freq_by_cluster %>%
  filter(diagnosis_label %in% top_diagnoses_overall) %>%
  select(cluster, diagnosis_label, percentage) %>%
  pivot_wider(
    names_from = cluster,
    values_from = percentage,
    values_fill = 0
  ) %>%
  column_to_rownames("diagnosis_label") %>%
  as.matrix()

# Create heatmap
library(pheatmap)

pdf("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Clustered/diagnosis_frequency_heatmap.pdf", 
    width = 8, height = 10)
pheatmap(
  diagnosis_freq_matrix,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  display_numbers = TRUE,
  number_format = "%.1f",
  main = "Diagnosis Frequency Heatmap by Cluster (%)",
  color = colorRampPalette(c("white", "#1976D2"))(100),
  fontsize = 8,
  fontsize_row = 7,
  fontsize_col = 10
)
dev.off()

cat("Heatmap saved to: diagnosis_frequency_heatmap.pdf\n\n")

# Summary statistics
cat("========================================\n")
cat("SUMMARY STATISTICS\n")
cat("========================================\n\n")

total_unique_diagnoses <- length(unique(diagnosis_long$individual_diagnoses))
cat(sprintf("Total unique diagnoses across all clusters: %d\n\n", total_unique_diagnoses))

diagnosis_summary <- diagnosis_freq_by_cluster %>%
  group_by(diagnosis_label) %>%
  summarise(
    total_count = sum(count),
    present_in_clusters = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_count))

cat("Top 20 most common diagnoses overall (excluding 'No Diagnosis'):\n")
diagnosis_summary_filtered <- diagnosis_summary %>%
  filter(diagnosis_label != "No Diagnosis")
print(head(diagnosis_summary_filtered, 20), row.names = FALSE)

# Show "No Diagnosis" separately
no_diag_summary <- diagnosis_summary %>%
  filter(diagnosis_label == "No Diagnosis")
if (nrow(no_diag_summary) > 0) {
  cat("\nParticipants with No Diagnosis:\n")
  print(no_diag_summary, row.names = FALSE)
}
cat("\n")

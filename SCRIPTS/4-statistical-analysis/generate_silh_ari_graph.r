# Silhouette and stability ARI vs K (SOM / K-means model selection)

library(ggplot2)

TIMESTAMP <- "APR_17_2026"
ROOT_DIR <- "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL"
STATS_PATH <- file.path(ROOT_DIR, "DATA/OUTPUTS/Stats")

k_metrics <- data.frame(
  K = 2:8,
  Silhouette = c(0.4083, 0.3117, 0.3027, 0.2767, 0.2579, 0.2433, 0.2312),
  Stability_ARI = c(1.0000, 0.9630, 0.9813, 0.9489, 0.9216, 0.7520, 0.8053)
)

dir.create(STATS_PATH, recursive = TRUE, showWarnings = FALSE)

grey_theme <- theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(color = "black", face = "plain"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "grey20"),
    panel.grid.major = element_line(color = "grey88", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "grey40", linewidth = 0.4),
    legend.position = "none"
  )

p_sil <- ggplot(k_metrics, aes(x = K, y = Silhouette)) +
  geom_line(color = "grey35", linewidth = 0.7) +
  geom_point(color = "black", fill = "grey75", shape = 21, size = 3, stroke = 0.4) +
  scale_x_continuous(breaks = k_metrics$K) +
  labs(
    x = "K (number of clusters)",
    y = "Silhouette score",
    title = "Silhouette score by K"
  ) +
  grey_theme

p_ari <- ggplot(k_metrics, aes(x = K, y = Stability_ARI)) +
  geom_line(color = "grey35", linewidth = 0.7) +
  geom_point(color = "black", fill = "grey75", shape = 21, size = 3, stroke = 0.4) +
  scale_x_continuous(breaks = k_metrics$K) +
  labs(
    x = "K (number of clusters)",
    y = "Stability (ARI)",
    title = "Stability (mean pairwise ARI) by K"
  ) +
  grey_theme

out_sil <- file.path(STATS_PATH, paste0("som_k_selection_silhouette_", TIMESTAMP, ".pdf"))
out_ari <- file.path(STATS_PATH, paste0("som_k_selection_stability_ari_", TIMESTAMP, ".pdf"))

ggsave(out_sil, plot = p_sil, width = 5, height = 3.8)
ggsave(out_ari, plot = p_ari, width = 5, height = 3.8)

message("Saved: ", out_sil)
message("Saved: ", out_ari)

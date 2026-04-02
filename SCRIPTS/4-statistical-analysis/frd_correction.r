# FDR (Benjamini–Hochberg) for manually entered ANOVA p-values — 12 EEG features

test_names <- c(
  "hurst",
  "delta",
  "theta",
  "alpha",
  "beta",
  "low_gamma",
  "high_gamma",
  "fooof_eponent",
  "fooof_offset",
  "higuchi",
  "CI_loscale",
  "CI_highcale"
)

p_raw <- c(
  0.03597, # hurst
  0.01757, # delta
  0.0001511, # theta
  0.196, # alpha
  1.21e-10, # beta
  0.1573, # low_gamma
  0.02748, # high gamma
  9.142e-12, # fooof_eponent
  2.2e-16, # fooof_offset
  3.049e-06, # higuchi
  4.845e-09, # CI_loscale
  0.2965 # CI_highcale
)

# cluster0 / cluster1 / cluster2 vs ref (same order as test_names)
p_c0_3 <- c(0.02183, 0.04805, 0.028067, 0.7639, 0.3119, 0.1075, 0.03564, 0.54783, 0.573, 0.17066, 0.4765, 0.2148)
p_c1_3 <- c(0.00649, 0.00985, 0.482236, 0.6322, 0.1055, 0.0147, 0.00175, 0.11141, 0.874, 0.00238, 0.7378, 0.0454)
p_c2_3 <- c(0.05747, 0.43115, 0.313896, 0.8473, 0.6572, 0.0396, 0.02219, 0.24940, 0.399, 0.07936, 0.9984, 0.2878)


p_fdr <- p.adjust(p_raw, method = "BH")
results <- data.frame(
  test = test_names,
  p_ANOVA = p_raw,
  p_FDR_BH = p_fdr,
  significant_0.05 = p_fdr < 0.05
)

pm <- cbind(p_c0_3, p_c1_3, p_c2_3)
v <- as.vector(t(pm))
v_adj <- rep(NA_real_, length(v))
ok <- !is.na(v)
v_adj[ok] <- p.adjust(v[ok], method = "BH")
adj <- matrix(v_adj, nrow = nrow(pm), ncol = 3, byrow = TRUE)

fdr_with_star <- function(p) {
  out <- rep(NA_character_, length(p))
  okp <- !is.na(p)
  out[okp] <- ifelse(
    p[okp] < 0.05,
    paste0(sprintf("%.4g", p[okp]), " *"),
    sprintf("%.4g", p[okp])
  )
  out
}

cluster_results <- data.frame(
  test = test_names,
  p_c0_3 = p_c0_3,
  p_c1_3 = p_c1_3,
  p_c2_3 = p_c2_3,
  p_FDR_c0_3 = fdr_with_star(adj[, 1]),
  p_FDR_c1_3 = fdr_with_star(adj[, 2]),
  p_FDR_c2_3 = fdr_with_star(adj[, 3]),
  stringsAsFactors = FALSE
)

print(results, digits = 5, na.print = ".")
cat("\n")
print(cluster_results, digits = 5, na.print = ".")

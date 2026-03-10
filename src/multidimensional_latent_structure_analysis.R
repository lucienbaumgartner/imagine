library(readxl)
library(dplyr)
library(mclust)
library(ggplot2)
library(reshape2)
library(scatterplot3d)
library(tidyr)
library(glmnet)
library(nnet)
library(purrr)
library(scales)

rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load data
df <- read_xlsx("../input/corpus_annotated.xlsx", sheet = 1)
names(df) <- gsub("\\smeans", "", tolower(names(df)))

# Drop excluded snippets and those with too little information
df <- df %>% mutate(across(ends_with("ity"), ~ as.numeric(.))) # will introduce NAs for X and E
df <- na.omit(df)

# Select only the annotation dimensions
annot <- df %>%
  select(intentionality, factivity, pictoriality)

# Check ranges (should be within 1–7 or slightly outside if averaged)
apply(annot, 2, range)

# Pairwise correlations
cor(annot, use = "pairwise.complete.obs")

# Distributions
annot_long <- melt(annot) # Long form
ggplot(annot_long, aes(x = value, fill = variable)) + 
  geom_histogram(binwidth=0.5) + 
  facet_wrap(~ variable) +
  theme_light()

# Standardize annotation space
annot_z <- as.data.frame(scale(annot))

scatterplot3d(annot_z$intentionality, 
              annot_z$factivity,
              annot_z$pictoriality, 
              pch=19, color="blue")

# Distributions scaled
annot_z_long <- melt(annot_z) # Long form
ggplot(annot_z_long, aes(x = value, fill = variable)) + 
  geom_histogram(binwidth=0.5) + 
  facet_wrap(~ variable) +
  theme_light()

## Clustering
# 1- vs 2- vs 3-component solution
gmm_res <- Mclust(annot_z, G = 1:3)
summary(gmm_res) 
# BIC values
plot(gmm_res, what = "BIC",
     main = "BIC for Gaussian mixture models")

# option with three clusters is preferred, but we use a two-cluster solution for the mock up

# 2-component solution
gmm_res <- Mclust(annot_z, G = 2)

# Cluster assignments
annot_z$sense <- gmm_res$classification

# Posterior probabilities (membership confidence)
annot_z$posterior_max <- apply(gmm_res$z, 1, max)

# Check cluster means
aggregate(cbind(intentionality, factivity, pictoriality) ~ sense, data = annot_z, mean)

# PCA to check for 2D cluster separation
pca_res <- prcomp(annot_z, center = FALSE, scale. = FALSE)

# Variance explained
summary(pca_res)

# Combine clustering with PCA
pca_res <- bind_cols(pca_res$x, annot_z)

ggplot(pca_res, aes(PC1, PC2, color = as.factor(sense))) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA of annotation space with latent sense clusters",
       color = "Sense") +
  theme_minimal()

# 3D cluster separation
scatterplot3d(annot_z$intentionality, 
              annot_z$factivity,
              annot_z$pictoriality, 
              pch=19, color=annot_z$sense)

### GLM with ridge regularization
# Fit multinomial regression with 2-wy interactions as IV L2 (ridge) regularization for stability
y <- as.factor(paste0("sense ", annot_z$sense))
x_int <- model.matrix(~ (intentionality + factivity + pictoriality)^2, data = annot_z)[,-1]
cv_fit_int <- cv.glmnet(x_int, y, family = "multinomial", alpha = 0, type.measure = "class")
best_lambda <- cv_fit_int$lambda.min

coefs <- coef(cv_fit_int, s = best_lambda)

# Convert to data frame for inspection
coefs_df <- do.call(rbind, lapply(names(coefs), function(cluster){
  data.frame(cluster = cluster,
             term = rownames(coefs[[cluster]]),
             coef = as.numeric(coefs[[cluster]]))
}))

# Plot interactions
vars <- c("intentionality", "factivity", "pictoriality")
pairs <- combn(vars, 2, simplify = FALSE)

plots <- list()

for (vp in pairs) {
  x_var <- vp[1]
  y_var <- vp[2]
  fix_var <- setdiff(vars, vp)
  
  # Build grid manually
  grid <- expand.grid(
    seq(min(annot_z[[x_var]]), max(annot_z[[x_var]]), length.out = 50),
    seq(min(annot_z[[y_var]]), max(annot_z[[y_var]]), length.out = 50)
  )
  colnames(grid) <- c(x_var, y_var)
  
  # Add fixed variable at its median
  grid[[fix_var]] <- median(annot_z[[fix_var]])
  
  # Model matrix
  grid_mm <- model.matrix(~ (intentionality + factivity + pictoriality)^2, data = grid)[,-1]
  
  # Predicted probabilities
  probs <- predict(cv_fit_int, newx = grid_mm, type = "response", s = best_lambda)
  probs <- as.data.frame(probs[,,1])
  colnames(probs) <- levels(y)
  grid <- cbind(grid, probs)
  
  # Long format
  grid_long <- pivot_longer(grid, cols = colnames(probs), names_to = "cluster", values_to = "prob") %>%
    group_by(cluster) %>%
    mutate(prob_scaled = rescale(prob)) %>%
    ungroup()
  
  # Heatmap plot
  p <- ggplot(grid_long, aes_string(x = x_var, y = y_var, fill = "prob_scaled")) +
    geom_tile() +
    facet_wrap(~ cluster, ncol = 1) +
    scale_fill_viridis_c(option = "magma") +
    labs(x = paste0(x_var, " (z-score)"),
         y = paste0(y_var, " (z-score)"),
         fill = "Predicted probability",
         title = paste("Cluster probabilities:", x_var, "vs", y_var)) +
    scale_y_continuous(expand = c(0,0)) +
    scale_x_continuous(expand = c(0,0)) +
    theme_light() +
    theme(
      panel.grid = element_blank()
    )

  plots[[paste(x_var, y_var, sep = "_")]] <- p
}

plots$intentionality_factivity
plots$intentionality_pictoriality
plots$factivity_pictoriality

# Robustness check via bootstrapping
n_boot <- 500
n_vars <- ncol(x_int)
n_clusters <- length(levels(y))

# Arrays to store results
coefs_boot <- array(NA, dim = c(n_boot, n_vars, n_clusters))
selected_boot <- array(0, dim = c(n_boot, n_vars, n_clusters))  # 1 if nonzero

set.seed(1847)
for(i in 1:n_boot){
  # resample rows with replacement
  rows <- sample(1:nrow(x_int), replace = TRUE)
  
  # fit multinomial glmnet
  fit <- glmnet(x_int[rows,], y[rows], family="multinomial", alpha=0)
  
  # extract coefficients at best lambda
  coefs_list <- coef(fit, s = best_lambda)
  
  for(k in 1:n_clusters){
    # coefs_list[[k]] includes intercept as row 1, remove intercept if desired
    coefs_k <- as.numeric(coefs_list[[k]][-1, ]) # drop intercept
    coefs_boot[i, , k] <- coefs_k
    
    # mark which coefficients are non-zero
    selected_boot[i, , k] <- as.numeric(coefs_k != 0)
  }
}

# Compute bootstrap CI
ci_lower <- apply(coefs_boot, c(2,3), function(x) quantile(x, 0.025, na.rm=TRUE))
ci_upper <- apply(coefs_boot, c(2,3), function(x) quantile(x, 0.975, na.rm=TRUE))
coef_median <- apply(coefs_boot, c(2,3), median, na.rm=TRUE)

# Compute stability (selection frequency)
stability <- apply(selected_boot, c(2,3), mean, na.rm=TRUE)
stability # all predictors are selected in 100% of iterations

# Convert to data.frame for reporting
rename_terms <- function(terms) {
  parts <- strsplit(terms, ":", fixed = TRUE)
  
  renamed <- vapply(parts, function(p) {
    paste(toupper(substr(p, 1, 1)), collapse = ":")
  }, character(1))
  
  renamed
}

coef_df <- expand.grid(
  term = rename_terms(colnames(x_int)),
  cluster = as.factor(paste0("sense ", 1:n_clusters))
) %>%
  mutate(
    median = as.vector(coef_median),
    ci_lower = as.vector(ci_lower),
    ci_upper = as.vector(ci_upper),
    stability = as.vector(stability)
  )

# Quick view
head(coef_df)

ggplot(coef_df, aes(x = term, y = median, color = cluster)) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                position = position_dodge(width = 0.6),
                width = 0.2) +
  geom_point(position = position_dodge(width = 0.6), size = 1) +
  labs(
    color = "Cluster",
    x = "Term",
    y = "Median Coefficients"
  ) +
  theme_light()

# Write out clustering
annot_z <- annot_z %>% rename_with(~ paste0(.x, "_z"), .cols = 1:3)
corpus <- bind_cols(df, annot_z)
write.csv(corpus, file = "../output/data/corpus.csv", quote = T, row.names = F)

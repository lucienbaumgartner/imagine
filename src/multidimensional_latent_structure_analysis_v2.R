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
library(caret)

rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load feature annotation
df_features <- read_xlsx("../input/corpus_features.xlsx", sheet = 1)
names(df_features) <- gsub("\\smeans", "", tolower(names(df_features)))
df_features <- df_features %>% select(-number)

# Load sense annotation
df_senses <- read_xlsx("../input/corpus_senses.xlsx", sheet = 1)
names(df_senses) <- tolower(names(df_senses))
df_senses <- df_senses %>% select(-sense_arb)
str(df_senses)

# Select variables
df_senses <- df_senses %>% select(id, no, exclusions, starts_with("sense"))

# Merge datasets
df <- left_join(df_features, df_senses, by = c("id", "no"))

# Drop excluded snippets and those with sense disagreements
df <- df %>% mutate(across(ends_with("ity"), ~ as.numeric(.))) # will introduce NAs for X and E
table((is.na(df$factivity) | is.na(df$intentionality) | is.na(df$pictoriality)) == !is.na(df$exclusions)) # 56 exclusion discrepancies between data sets
df <- df %>% filter(is.na(exclusions)) %>% select(-exclusions) # drop sense annotation exclusions
df <- na.omit(df) # drop feature annotation exclusions
df <- df %>% filter(sense_nch == sense_ah) # drop sense disagreements

# Factor for sense
df <- df %>% 
  rename(sense = sense_nch) %>% 
  select(-sense_ah) %>% 
  mutate(
    sense = factor(case_when(
      sense == "1" ~ "1-VIZUALIZE/PICTURE",
      sense == "2" ~ "2-THINK/BELIVE",
      sense == "3" ~ "3-SUPPOSE/ASSUME",
      sense == "4" ~ "4-FALSE_PERCEPTION",
      sense == "5" ~ "5-EXCLAMATIVE",
      sense == "O" ~ "O-OTHER",
      sense == "?" ~ "?-UNCLASSIFIABLE"
  )))

table(df$sense)

# Drop edge cases
df <- df %>% 
  filter(!sense %in% c("O-OTHER", "?-UNCLASSIFIABLE", "5-EXCLAMATIVE")) %>% 
  mutate(sense = factor(sense))
levels(df$sense)

# Scale features
df <- df %>% mutate(across(ends_with("ity"), ~ scale(.)))

### Observed distributional overlap
df %>%
  select(sense, intentionality, factivity, pictoriality) %>%
  pivot_longer(-sense, names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = value, fill = sense, color = sense)) +
  geom_density(alpha = 0.3, linewidth = 0.6) +
  facet_wrap(~ feature, ncol = 1) +
  scale_fill_brewer(palette = "Set1") +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  labs(
    x = "z-score",
    y = "Density",
    fill = "Sense",
    color = "Sense",
    title = "Feature distributions by sense"
  ) +
  theme(legend.position = "bottom")

### GLM with ridge regularization
# Fit multinomial regression with 2-wy interactions as IV L2 (ridge) regularization for stability
y <- df$sense
x_int <- model.matrix(~ (intentionality + factivity + pictoriality)^2, data = df)[,-1]
cv_fit_int <- cv.glmnet(x_int, y, family = "multinomial", alpha = 0, type.measure = "class")
best_lambda <- cv_fit_int$lambda.min

# Predicted classes on training data
pred_class <- predict(cv_fit_int, newx = x_int, s = best_lambda, type = "class")
pred_class <- factor(pred_class, levels = levels(y))

# Confusion matrix
conf_mat <- table(Predicted = pred_class, Actual = y)
print(conf_mat)

# Overall accuracy
accuracy <- mean(pred_class == y)
cat("Training accuracy:", round(accuracy, 3), "\n")

# Per-class precision, recall, F1
conf_detail <- confusionMatrix(pred_class, y)
print(conf_detail$byClass[, c("Precision", "Recall", "F1")])

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
    seq(min(df[[x_var]]), max(df[[x_var]]), length.out = 50),
    seq(min(df[[y_var]]), max(df[[y_var]]), length.out = 50)
  )
  colnames(grid) <- c(x_var, y_var)
  
  # Add fixed variable at its median
  grid[[fix_var]] <- median(df[[fix_var]])
  
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
  sense = as.factor(paste0("sense ", 1:n_clusters))
) %>%
  mutate(
    median = as.vector(coef_median),
    ci_lower = as.vector(ci_lower),
    ci_upper = as.vector(ci_upper)
  )

# Quick view
head(coef_df)

coef_df <- coef_df %>% 
  mutate(sense = case_when(
    sense == "sense 1" ~ "1-VIZUALIZE/PICTURE",
    sense == "sense 2" ~ "2-THINK/BELIVE",
    sense == "sense 3" ~ "3-SUPPOSE/ASSUME",
    sense == "sense 4" ~ "4-FALSE_PERCEPTION"
  ))

ggplot(coef_df, aes(x = term, y = median, color = term)) +
  geom_hline(yintercept = 0) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                position = position_dodge(width = 0.6),
                width = 0.2) +
  geom_point(position = position_dodge(width = 0.6), size = 1) +
  facet_grid(~ sense) +
  labs(
    color = "Term",
    x = "Sense",
    y = "Median Coefficients"
  ) +
  theme_light()

# Write out clustering
write.csv(df, file = "../output/data/corpus_v2.csv", quote = T, row.names = F)

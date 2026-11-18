################################################################################
### FINAL ANALYSIS PIPELINE: OFT-ANCHORED + FEATURE LOADING ANALYSIS         ###
################################################################################

# --- SETUP: LOAD ALL NECESSARY LIBRARIES ---
library(lme4)
library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(stringr)
library(knitr)
library(pheatmap) 
library(scales)
library(ggrepel)

set.seed(123)

OUT_DIR <- "outputs/05_stage5/pca_analysis_R"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

################################################################################
### PART 1: DATA PREPARATION & RESIDUALIZATION (No Changes)                        ###
################################################################################

cat("\n\n--- PART 1: DATA PREPARATION & RESIDUALIZATION ---\n")

file_path <- "outputs/05_stage5/study_group_feature_master_presence_gt10_full10meta_zscored.csv"
full_data <- read_csv(file_path)

all_colnames <- colnames(full_data)
core_features <- grep("^(freq50_s|transsummary_self_trans_s|transsummary_row_entropy_s)", all_colnames, value = TRUE)

analysis_data <- full_data %>%
    filter(!is.na(label_binary)) %>%
    mutate(loco_context = factor(case_when(
        str_detect(tolower(env_context), "food intake|ast") ~ "Food_AST",
        TRUE ~ as.character(env_context)
    )))

residualized_data <- analysis_data
for (feature in core_features) {
    lm_model <- lm(as.formula(paste(feature, "~ study_cohort + env_context")), data = analysis_data, na.action = na.exclude)
    residuals_vec <- residuals(lm_model)
    non_na_indices <- which(!is.na(analysis_data[[feature]]))
    residualized_data[[feature]][non_na_indices] <- residuals_vec
}

final_clean_data <- residualized_data %>%
    filter(complete.cases(select(., all_of(core_features))))

cat("Data preparation and residualization complete.\n")


################################################################################
### PART 2: FUNCTION FOR ROBUST, STRATIFIED PCA (CORRECTED SYNTAX)           ###
################################################################################

get_stable_pc_loadings <- function(data, features, n_iterations = 1000) {
    cat(paste("    - Running robust PCA with", n_iterations, "iterations...\n"))
    pca_input_matrix <- data %>% select(all_of(features))
    
    reference_pca <- prcomp(pca_input_matrix, center = TRUE, scale. = TRUE)
    # FIX: Changed $$ to $
    reference_loadings <- reference_pca$rotation 
    
    aligned_loadings_list <- list()
    for (i in 1:n_iterations) {
        balanced_subsample <- data %>% group_by(env_context, study_cohort) %>% slice_sample(n = 1, replace = TRUE) %>% ungroup()
        subsample_matrix <- balanced_subsample %>% select(all_of(features))
        variances <- apply(subsample_matrix, 2, var)
        features_to_use <- names(variances[variances > 1e-10])
        if (length(features_to_use) < 2) next
        
        iter_pca <- prcomp(subsample_matrix[, features_to_use], center = TRUE, scale. = TRUE)
        # FIX: Changed $$ to $
        iter_loadings <- matrix(0, nrow = length(features), ncol = ncol(iter_pca$rotation), 
                                dimnames = list(features, colnames(iter_pca$rotation)))
        iter_loadings[rownames(iter_pca$rotation), ] <- iter_pca$rotation
        
        for (pc_index in 1:min(ncol(iter_loadings), ncol(reference_loadings))) {
            dot_product <- sum(reference_loadings[, pc_index] * iter_loadings[, pc_index])
            if (dot_product < 0) { iter_loadings[, pc_index] <- iter_loadings[, pc_index] * -1 }
        }
        aligned_loadings_list[[i]] <- iter_loadings
    }
    stable_unbiased_loadings <- Reduce('+', aligned_loadings_list) / length(aligned_loadings_list)
    return(stable_unbiased_loadings)
}


################################################################################
### PART 3: MODIFIED CV LOOP - FLEXIBLE PC HANDLING (No Changes)             ###
################################################################################
cat("\n\n--- PART 3: BEGINNING OFT-ANCHORED ROBUSTNESS TESTING ---\n")
all_contexts <- unique(final_clean_data$loco_context)
other_contexts <- setdiff(all_contexts, "OFT")
summary_table <- data.frame()
all_fold_loadings <- list() 
all_fold_variances <- list()
n_permutations <- 999 

for (partner_context in other_contexts) {
    training_pair_label <- paste("OFT +", partner_context)
    cat(paste("\n--- FOLD: Training on '", training_pair_label, "' ---\n", sep=""))
    train_data <- final_clean_data %>% filter(loco_context %in% c("OFT", partner_context))
    cat(sprintf("  - Using %d observations for this fold.\n", nrow(train_data)))
    stable_loadings <- get_stable_pc_loadings(train_data, core_features, n_iterations = 1000)
    all_fold_loadings[[training_pair_label]] <- stable_loadings
    train_matrix_scaled <- scale(train_data %>% select(all_of(core_features)), center = TRUE, scale = TRUE)
    train_pc_scores <- as.data.frame(train_matrix_scaled %*% stable_loadings)
    max_pcs_to_test <- 4
    all_pcs_generated <- colnames(train_pc_scores)
    pcs_to_test <- intersect(paste0("PC", 1:max_pcs_to_test), all_pcs_generated)
    if(length(pcs_to_test) < 1) {cat("  - WARNING: PCA failed to generate any PCs for this fold. Skipping.\n"); next}
    cat(sprintf("  - PCA generated %d PCs. Using: %s\n", length(pcs_to_test), paste(pcs_to_test, collapse=", ")))
    pc_vars <- apply(train_pc_scores, 2, var); total_var <- sum(pc_vars)
    all_fold_variances[[training_pair_label]] <- data.frame(PC = names(pc_vars), fold = training_pair_label, variance = pc_vars, prop_variance = pc_vars / total_var)
    modeling_data <- bind_cols(train_data, train_pc_scores) %>% select(label_binary, study_cohort, env_context, all_of(pcs_to_test))
    model_formula <- as.formula(paste("label_binary ~", paste(pcs_to_test, collapse = " + "), "+ env_context + (1 | study_cohort)"))
    cat("    - Fitting real model and getting observed z-scores...\n")
    real_model <- glmer(model_formula, data = modeling_data, family = binomial(link = "logit"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    real_summary <- summary(real_model)$coefficients
    observed_stats <- real_summary[pcs_to_test[pcs_to_test %in% rownames(real_summary)], "z value"]
    pcs_to_test <- names(observed_stats) 
    if(length(pcs_to_test) < 1) {cat("  - WARNING: No PC coefficients were estimated. Skipping fold.\n"); next}
    cat(paste("    - Running", n_permutations, "permutations...\n"))
    null_stats_list <- sapply(pcs_to_test, function(x) numeric(n_permutations), simplify = FALSE)
    for (i in 1:n_permutations) {
        permuted_data <- modeling_data %>% group_by(study_cohort) %>% mutate(shuffled_label = sample(label_binary)) %>% ungroup()
        null_summary <- tryCatch({
            perm_formula <- as.formula(paste("shuffled_label ~", paste(pcs_to_test, collapse = " + "), "+ env_context + (1 | study_cohort)"))
            summary(glmer(perm_formula, data = permuted_data, family = binomial(link = "logit"), control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))))$coefficients
        }, error = function(e) { return(NULL) })
        for(pc in pcs_to_test){
            if (!is.null(null_summary) && pc %in% rownames(null_summary)) null_stats_list[[pc]][i] <- null_summary[pc, "z value"] else null_stats_list[[pc]][i] <- NA
        }
    }
    p_values <- sapply(pcs_to_test, function(pc){
        valid_perms <- sum(!is.na(null_stats_list[[pc]])); (sum(abs(null_stats_list[[pc]]) >= abs(observed_stats[pc]), na.rm = TRUE) + 1) / (valid_perms + 1)
    })
    current_results <- data.frame(training_pair = training_pair_label, term = pcs_to_test, observed_z_score = observed_stats, permutation_p_value = p_values)
    summary_table <- rbind(summary_table, current_results)
    cat("    - Fold complete.\n")
}
cat("\n\n--- ANALYSIS COMPLETE ---\n")

################################################################################
### PARTS 4, 5, 6: SUMMARIZING RESULTS (No Changes)                          ###
################################################################################
cat("\n\n--- PART 4: FINAL SIGNIFICANCE ROBUSTNESS SUMMARY ---\n"); if(nrow(summary_table) > 0) {final_summary <- summary_table %>% mutate(observed_z_score = round(observed_z_score, 2), permutation_p_value = round(permutation_p_value, 4), significance = case_when(permutation_p_value < 0.001 ~ "***", permutation_p_value < 0.01  ~ "**", permutation_p_value < 0.05  ~ "*", TRUE ~ "")) %>% arrange(term, training_pair); cat("\n--- Summary of PC Significance Across OFT-Anchored Folds ---\n"); print(kable(final_summary, caption = "Robustness of PC effects in OFT-Anchored training sets", row.names = FALSE)); z_score_plot <- ggplot(final_summary, aes(x = training_pair, y = observed_z_score, fill = term)) + geom_col(position = "dodge") + geom_hline(yintercept = c(1.96, -1.96), linetype = "dashed", color = "red") + facet_wrap(~term, scales = "free_x", ncol = 2) + labs(title = "Stability of PC Z-Scores in OFT-Anchored Folds", subtitle = "PCs are derived from 'OFT + partner context'", x = "Training Pair", y = "Observed Z-Score from GLMM") + theme_minimal(base_size = 14) + theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1)); ggsave(file.path(OUT_DIR, "oft_anchored_zscores.png"), z_score_plot, width=10, height=8, dpi=150); print(z_score_plot)} else {cat("No results to summarize.\n")}
cat("\n\n--- PART 5: PC LOADING STABILITY ANALYSIS ---\n"); for(pc_to_check in c("PC1", "PC2", "PC3", "PC4")){valid_folds <- names(all_fold_loadings)[sapply(all_fold_loadings, function(l) pc_to_check %in% colnames(l))]; if(length(valid_folds) < 2) {cat(paste0("\nSkipping stability analysis for ", pc_to_check, ": Not present in enough folds.\n")); next}; loadings_df <- do.call(cbind, lapply(all_fold_loadings[valid_folds], function(loadings) {loadings[, pc_to_check]})); loading_correlation_matrix <- cor(loadings_df); cat(paste0("\n\n--- Correlation of ", pc_to_check, " Loadings Across OFT-Anchored Folds ---\n")); print(round(loading_correlation_matrix, 3)); png(file.path(OUT_DIR, paste0("loading_stability_", pc_to_check, ".png")), width=600, height=500); pheatmap(loading_correlation_matrix, display_numbers = TRUE, cluster_rows = TRUE, cluster_cols = TRUE, main = paste("Stability of", pc_to_check, "Definition Across OFT-Anchored Folds"), fontsize_number = 10, color = colorRampPalette(c("firebrick", "white", "steelblue"))(100), breaks = seq(-1, 1, length.out = 101)); dev.off()}
cat("\n\n--- PART 6: AGGREGATED SCREE PLOT ---\n"); if (length(all_fold_variances) > 0) {variance_df <- do.call(rbind, all_fold_variances); aggregated_variance <- variance_df %>% group_by(PC) %>% summarise(avg_prop_variance = mean(prop_variance), sd_prop_variance = sd(prop_variance), .groups = 'drop') %>% mutate(PC_num = as.numeric(gsub("PC", "", PC))) %>% arrange(PC_num) %>% mutate(cumulative_variance = cumsum(avg_prop_variance)); aggregated_scree_plot <- ggplot(head(aggregated_variance, 10), aes(x = factor(PC, levels = PC))) + geom_col(aes(y = avg_prop_variance), fill = "skyblue3", alpha = 0.8) + geom_errorbar(aes(ymin = pmax(0, avg_prop_variance - sd_prop_variance), ymax = avg_prop_variance + sd_prop_variance), width = 0.2, color = "gray40") + geom_line(aes(y = cumulative_variance, group = 1), color = "firebrick", linewidth = 1) + geom_point(aes(y = cumulative_variance), color = "firebrick", size = 2.5) + scale_y_continuous(labels = scales::percent_format(), name = "Avg. Proportion of Variance Explained") + labs(title = "Aggregated Scree Plot Across OFT-Anchored Folds", subtitle = "Bars show average variance (±SD); red line shows cumulative average.", x = "Principal Component") + theme_minimal(base_size = 14); ggsave(file.path(OUT_DIR, "aggregated_scree_plot.png"), aggregated_scree_plot, width=8, height=5, dpi=150); print(aggregated_scree_plot); cat("\n--- Aggregated Variance Explained Table ---\n"); print(kable(head(aggregated_variance, 10), digits = 3))} else {cat("No variance information to plot.\n")}

################################################################################
### PART 7: ANALYZING FEATURE LOADINGS FOR KEY BEHAVIORAL SIGNATURES         ###
################################################################################

cat("\n\n--- PART 7: INTERPRETING THE KEY BEHAVIORAL SIGNATURES ---\n")

# Define the folds that represent our two key signatures based on the results
signature_A_folds <- c("OFT + Food_AST", "OFT + OFT_DUO")
signature_B_fold <- "OFT + Y maze"

# --- Analysis of Signature A: Stable PC1 ---
cat("\n\n--- Analysis of Signature A: Stable PC1 (from OFT+Food_AST & OFT+OFT_DUO) ---\n")

# Check which of the desired folds are actually available
valid_A_folds <- intersect(signature_A_folds, names(all_fold_loadings))
if (length(valid_A_folds) > 0) {
  # Extract the PC1 loading vectors from the two stable folds
  pc1_loadings_list <- lapply(all_fold_loadings[valid_A_folds], function(l) l[, "PC1"])
  pc1_loadings_df <- as.data.frame(do.call(cbind, pc1_loadings_list))
  pc1_loadings_df$feature <- rownames(pc1_loadings_df)

  # Calculate the mean and standard deviation for each feature's loading
  loadings_summary_A <- pc1_loadings_df %>%
    rowwise() %>%
    mutate(
      mean_loading = mean(c_across(-feature)),
      sd_loading = sd(c_across(-c(feature, mean_loading)))
    ) %>%
    select(feature, mean_loading, sd_loading) %>%
    ungroup() %>%
    arrange(desc(abs(mean_loading)))
  
  cat("\nTop 10 most important AND stable features for Signature A (PC1):\n")
  print(kable(head(loadings_summary_A, 10), digits=3))

  # Visualize the averaged loadings with stability info
  plot_sig_A <- ggplot(head(loadings_summary_A, 20), aes(x = reorder(feature, mean_loading), y = mean_loading)) +
    geom_col(aes(fill = mean_loading > 0), show.legend = FALSE) +
    geom_errorbar(aes(ymin = mean_loading - sd_loading, ymax = mean_loading + sd_loading), width = 0.2) +
    coord_flip() +
    labs(
      title = "Signature A: Feature Loadings for Stable PC1",
      subtitle = "Average of 'OFT+Food_AST' & 'OFT+OFT_DUO' folds. Error bars show SD.",
      x = "Behavioral Feature", y = "Average Loading"
    ) +
    theme_minimal()
  ggsave(file.path(OUT_DIR, "signature_A_pc1_loadings.png"), plot_sig_A, width=7, height=8, dpi=150)
  print(plot_sig_A)
} else {
  cat("Could not find loading data for Signature A folds.\n")
}


# --- Analysis of Signature B: Predictive PC2 ---
cat("\n\n--- Analysis of Signature B: Predictive PC2 (from OFT + Y maze) ---\n")

if (signature_B_fold %in% names(all_fold_loadings) && "PC2" %in% colnames(all_fold_loadings[[signature_B_fold]])) {
  pc2_loadings_B <- all_fold_loadings[[signature_B_fold]][, "PC2", drop = FALSE]
  
  loadings_summary_B <- as.data.frame(pc2_loadings_B) %>%
    tibble::rownames_to_column("feature") %>%
    rename(loading = PC2) %>%
    arrange(desc(abs(loading)))
  
  cat("\nTop 10 most important features for Signature B (PC2):\n")
  print(kable(head(loadings_summary_B, 10), digits=3))
  
  # Visualize the loadings
  plot_sig_B <- ggplot(head(loadings_summary_B, 20), aes(x = reorder(feature, loading), y = loading)) +
    geom_col(aes(fill = loading > 0), show.legend = FALSE) +
    coord_flip() +
    labs(
      title = "Signature B: Feature Loadings for Predictive PC2",
      subtitle = "Derived from 'OFT + Y maze' fold only.",
      x = "Behavioral Feature", y = "Loading"
    ) +
    theme_minimal()
  ggsave(file.path(OUT_DIR, "signature_B_pc2_loadings.png"), plot_sig_B, width=7, height=8, dpi=150)
  print(plot_sig_B)
} else {
  cat("Could not find PC2 loading data for the 'OFT + Y maze' fold.\n")
}

################################################################################
### PART 8: SYLLABLE SIGNATURE SCATTER PLOTS                                 ###
################################################################################
cat("\n\n--- PART 8: SYLLABLE SIGNATURE SCATTER PLOTS ---\n")

make_syllable_scatter <- function(summary_df, loading_col, title, out_path) {
  df <- summary_df %>%
    rename(loading_val = !!sym(loading_col)) %>%
    mutate(
      node = str_extract(feature, "s\\d+$"),
      type = case_when(
        str_detect(feature, "^freq50")     ~ "freq",
        str_detect(feature, "self_trans")  ~ "self_trans",
        str_detect(feature, "row_entropy") ~ "entropy",
        TRUE                               ~ NA_character_
      )
    ) %>%
    filter(!is.na(type)) %>%
    select(node, type, loading_val) %>%
    pivot_wider(names_from = type, values_from = loading_val) %>%
    mutate(
      usage      = (abs(freq) + abs(self_trans) + abs(entropy)) / 3,
      node_label = ifelse(grepl("^s0+$", node), "0", gsub("^s0*", "", node))
    )

  p <- ggplot(df, aes(x = freq, y = usage)) +
    geom_point(aes(color = self_trans, size = abs(entropy)), alpha = 0.85) +
    geom_text_repel(aes(label = node_label), size = 3.2,
                    min.segment.length = 0, segment.size = 0.3,
                    box.padding = 0.3, point.padding = 0.2,
                    max.overlaps = Inf, seed = 42) +
    scale_color_gradient2(low = "royalblue", mid = "white", high = "firebrick",
                          midpoint = 0, name = "Self-transition\nloading") +
    scale_size_continuous(name = "|Row-entropy\nloading|", range = c(2, 10)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.5) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.5) +
    labs(title = title,
         x = "Occurrence loading (freq50)",
         y = "Mean absolute loading (usage)") +
    scale_x_continuous(expand = expansion(mult = 0.15)) +
    scale_y_continuous(expand = expansion(mult = 0.15)) +
    theme_minimal(base_size = 13)

  ggsave(out_path, p, width = 8, height = 6, dpi = 150)
  cat(paste0("  -> ", out_path, "\n"))
  invisible(p)
}

if (exists("loadings_summary_A") && nrow(loadings_summary_A) > 0) {
  make_syllable_scatter(loadings_summary_A, "mean_loading",
    "Signature A: Syllable Profile (Stable PC1)",
    file.path(OUT_DIR, "signature_A_syllable_scatter.png"))
}

if (exists("loadings_summary_B") && nrow(loadings_summary_B) > 0) {
  make_syllable_scatter(loadings_summary_B, "loading",
    "Signature B: Syllable Profile (Predictive PC2)",
    file.path(OUT_DIR, "signature_B_syllable_scatter.png"))
}

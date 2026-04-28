# --- AUC-ROC curve
rm(list=ls())

library(jsonlite)
library(ggplot2)
library(dplyr)
library(purrr)
library(here)

# import model auc data
txt_irt <- readLines(here("data","MLdata","irt_roc_curve.json"), warn = FALSE, encoding = "UTF-8")
txt_pro <- readLines(here("data","MLdata","pro_roc_curve.json"), warn = FALSE, encoding = "UTF-8")

# repliace unreadable character to null
txt_irt <- gsub("\\bInfinity\\b", "null", txt_irt)
txt_irt <- gsub("\\b-Infinity\\b", "null", txt_irt)
txt_irt <- gsub("\\bNaN\\b", "null", txt_irt)

dat_irt <- fromJSON(paste(txt_irt, collapse = "\n"))

txt_pro <- gsub("\\bInfinity\\b", "null", txt_pro)
txt_pro <- gsub("\\b-Infinity\\b", "null", txt_pro)
txt_pro <- gsub("\\bNaN\\b", "null", txt_pro)

dat_pro <- fromJSON(paste(txt_pro, collapse = "\n"))


str(dat_irt)

#--------------------------------------
# discontinuation and OS
#--------------------------------------
time_points_group1 <- c("30", "60", "90", "180", "full") #

# --- OS - Total PRO model
logistic_OS_pro <- map_dfr(time_points_group1, function(tp) {
  ml_dat <- as.data.frame(dat_pro$pro_os_v4$logistic_regression[[paste0("vali_", tp)]])
  auc_value <- unique(ml_dat$auc)[1]
  ml_dat %>%
      mutate(label = paste0(ifelse(tp == "full", "Full", paste(tp, "days")),
              " (AUC = ", round(auc_value, 3), ")"))
})

# set factor sequence
logistic_OS_pro <- logistic_OS_pro %>%
  mutate(label = factor(label, levels = unique(label)))

#gradient_OS_pro2 <- gradient_OS_pro %>%
#              filter(label %in% c("Validation 90 days (AUC = 0.626)","Validation 180 days (AUC = 0.7)",
#                                  "Validation Full (AUC = 0.748)"))

pro_os <- ggplot(logistic_OS_pro, aes(x = fpr, y = tpr, color = label)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       color = "") +
  theme_bw() +
  theme(legend.position = c(0.65, 0.18),
        legend.background = element_rect(fill = "white", color = NA),
        legend.text = element_text(size = 15),
        axis.text = element_text(size = 16, color = "black", face = "bold"),
        axis.title = element_text(size = 16, face = "bold")) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_fixed()

print(pro_os)

ggsave("plot/figure2f.tiff",
       plot = pro_os,
       width = 6,
       height = 6,
       units = "in",
       dpi = 300)


# --- OS - IRT
logistic_OS <- map_dfr(time_points_group1, function(tp) {
  ml_dat <- as.data.frame(dat_irt$irt_overall_survival_death$logistic_regression[[paste0("vali_", tp)]])
  auc_value <- unique(ml_dat$auc)[1]
  ml_dat %>%
    mutate(label = paste0(ifelse(tp == "full", "Full", paste(tp, "days")),
                          " (AUC = ", round(auc_value, 3), ")"))
})

# set factor sequence
logistic_OS <- logistic_OS %>%
  mutate(label = factor(label, levels = unique(label)))

unique(logistic_OS$label)

irt_os <- ggplot(logistic_OS, aes(x = fpr, y = tpr, color = label)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       color = "") +
  theme_bw() +
  theme(legend.position = c(0.65, 0.18),
        legend.background = element_rect(fill = "white", color = NA),
        legend.text = element_text(size = 15),
        axis.text = element_text(size = 16, color = "black", face = "bold"),
        axis.title = element_text(size = 16, face = "bold")) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_fixed()

print(irt_os)

ggsave("plot/figure4b.tiff",
       plot = irt_os,
       width = 6,
       height = 6,
       units = "in",
       dpi = 300)

# --- Dose discontinuation
svm_discontinuation <- map_dfr(time_points_group1, function(tp) {
  ml_dat <- as.data.frame(dat_irt$irt_dose_discontinuation$svm_rbf[[paste0("vali_", tp)]])
  auc_value <- unique(ml_dat$auc)[1]
  ml_dat %>%
    mutate(label = paste0(ifelse(tp == "full", "Full", paste(tp, "days")),
                          " (AUC = ", round(auc_value, 3), ")"))
})

# set factor sequence
svm_discontinuation <- svm_discontinuation %>%
  mutate(label = factor(label, levels = unique(label)))

unique(svm_discontinuation$label)

svm_discontinuation2 <- svm_discontinuation %>%
              filter(label %in% c("180 days (AUC = 0.673)","Full (AUC = 0.707)"))

irt_dis <- ggplot(svm_discontinuation2, aes(x = fpr, y = tpr, color = label)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(
       x = "False Positive Rate",
       y = "True Positive Rate",
       color = "") +
  theme_bw() +
  theme(
    plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
    legend.position = c(0.65, 0.16),
    legend.background = element_rect(fill = "white", color = NA),
    legend.text = element_text(size = 15),
    axis.text = element_text(size = 16, color = "black", face = "bold"),
    axis.title = element_text(size = 16, face = "bold")) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_fixed()

print(irt_dis)

ggsave("plot/figure5c.tiff",
       plot = irt_dis,
       width = 6,
       height = 6,
       units = "in",
       dpi = 300)

#--------------------------------------
# Reduction, hospitalization, ED Visit
#--------------------------------------
time_points_group2 <- c("30", "60", "90", "event") 

# --- Dose reduction
rf_reduction <- map_dfr(time_points_group2, function(tp) {
  ml_dat <- as.data.frame(dat_irt$irt_dose_reduction$random_forest[[paste0("vali_",tp)]])
  auc_value <- unique(ml_dat$auc)[1]
  ml_dat %>%
    mutate(label = paste0(ifelse(tp == "event", "Before Event", paste(tp, "days")),
                          " (AUC = ", round(auc_value, 3), ")"))
})

# set factor sequence
rf_reduction <- rf_reduction %>%
  mutate(label = factor(label, levels = unique(label)))

unique(rf_reduction$label)

rf_reduction2 <- rf_reduction %>%
  filter(label %in% c("Before Event (AUC = 0.671)"))

irt_red <- ggplot(rf_reduction2, aes(x = fpr, y = tpr, color = label)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       color = "") +
  theme_bw() +
  theme(
    legend.position = c(0.62, 0.16),
    legend.background = element_rect(fill = "white", color = NA),
    legend.text = element_text(size = 15),
    axis.text = element_text(size = 16, color = "black", face = "bold"),
    axis.title = element_text(size = 16, face = "bold")) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_fixed()

print(irt_red)

ggsave("plot/figure5d.tiff",
       plot = irt_red,
       width = 6,
       height = 6,
       units = "in",
       dpi = 300)

# --- Hospitalization
rf_hopspitalizaiton <- map_dfr(time_points_group2, function(tp) {
  ml_dat <- as.data.frame(dat_irt$irt_hospitalization$random_forest[[paste0("vali_",tp)]])
  auc_value <- unique(ml_dat$auc)[1]
  ml_dat %>%
    mutate(label = paste0(ifelse(tp == "event", "Before Event", paste(tp, "days")),
                          " (AUC = ", round(auc_value, 3), ")"))
})

# set factor sequence
rf_hopspitalizaiton <- rf_hopspitalizaiton %>%
  mutate(label = factor(label, levels = unique(label)))

unique(rf_hopspitalizaiton$label)

random_hopspitalizaiton2 <- rf_hopspitalizaiton %>%
  filter(label %in% c("Before Event (AUC = 0.751)"))

irt_hos <- ggplot(random_hopspitalizaiton2, aes(x = fpr, y = tpr, color = label)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       color = "") +
  theme_bw() +
  theme(
    legend.position = c(0.65, 0.16),
    legend.background = element_rect(fill = "white", color = NA),
    legend.text = element_text(size = 15),
    axis.text = element_text(size = 16, color = "black", face = "bold"),
    axis.title = element_text(size = 16, face = "bold")) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_fixed()

print(irt_hos)

ggsave("plot/figure5a.tiff",
       plot = irt_hos,
       width = 6,
       height = 6,
       units = "in",
       dpi = 300)


# --- ED Visit
gradient_ER <- map_dfr(time_points_group2, function(tp) {
  ml_dat <- as.data.frame(dat_irt$irt_er$gradient_boosting[[paste0("vali_",tp)]])
  auc_value <- unique(ml_dat$auc)[1]
  ml_dat %>%
    mutate(label = paste0(ifelse(tp == "event", "Before Event", paste(tp, "days")),
                          " (AUC = ", round(auc_value, 3), ")"))
})

# set factor sequence
gradient_ER <- gradient_ER %>%
  mutate(label = factor(label, levels = unique(label)))

unique(gradient_ER$label)

gradient_ER2 <- gradient_ER %>%
  filter(label %in% c("Before Event (AUC = 0.652)"))

irt_ed <- ggplot(gradient_ER2, aes(x = fpr, y = tpr, color = label)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "False Positive Rate",
       y = "True Positive Rate",
       color = "") +
  theme_bw() +
  theme(
    legend.position = c(0.64, 0.16),
    legend.background = element_rect(fill = "white", color = NA),
    legend.text = element_text(size = 15),
    axis.text = element_text(size = 16, color = "black", face = "bold"),
    axis.title = element_text(size = 16, face = "bold")) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_fixed()

print(irt_ed)

ggsave("plot/figure5b.tiff",
       plot = irt_ed,
       width = 6,
       height = 6,
       units = "in",
       dpi = 300)


# --- SHAP Plot (bar plot)
# - irt model OS
shap_irt_bar <- read.csv(here("data","MLdata","shap_importance_irt_os.csv")) %>%
                mutate(feature = case_when(
                  feature == "IV" ~ "IV TREATMENT",
                  TRUE ~ feature
                  )) %>%
                slice_max(order_by = importance, n = 10)

shap_irt_bar <- shap_irt_bar %>%
  mutate(feature = factor(feature, levels = feature[order(importance)]))

p1 <- ggplot(shap_irt_bar, aes(x = importance, y = feature)) +
  geom_bar(
    stat  = "identity",
    fill  = "#1f8dd6",
    width = 0.7
  ) +
  scale_x_continuous(
    limits = c(0, max(shap_irt_bar$importance) * 1.08),
    expand = c(0, 0),
    breaks = c(0, 0.1, 0.2, 0.3, 0.4)
  ) +
  labs(
    #title    = "SHAP Feature Importance",
    #subtitle = "IRT - Overall Survival (Gradient Boosting)",
    x        = "mean(|SHAP value|) (average impact on model output magnitude)",
    y        = NULL
  ) +
  theme_bw()+
  theme(
    #plot.title    = element_text(hjust = 0.5, size = 13),
    #plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 14, color = "black"),
    axis.text.x = element_text(size = 15.5),
    axis.title.x  = element_text(size = 15, margin = margin(t = 8), face = "bold"),
    axis.line.x   = element_line(color = "black", linewidth = 0.5),
    panel.grid    = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin   = margin(t = 8, r = 20, b = 10, l = 10)
  )

print(p1)

ggsave("plot/figure4d.tiff",
       plot = p1,
       width = 8,
       height = 6,
       units = "in",
       dpi = 300)

# - prototal model OS
shap_prototal_bar <- read.csv(here("data","MLdata","shap_importance_pro_os.csv")) %>%
  filter(importance != "0") %>%
  mutate(feature = case_when(
    feature == "IV" ~ "IV TREATMENT",
    feature == "PO" ~ "PO TREATMENT",
    feature == "Emax" ~ "EMAX",
    feature == "Kd" ~ "KD",
    TRUE ~ feature
  )) %>%
  slice_max(order_by = importance, n = 10)

shap_prototal_bar$feature

shap_prototal_bar <- shap_prototal_bar %>%
  mutate(feature = factor(feature, levels = feature[order(importance)]))

p2 <- ggplot(shap_prototal_bar, aes(x = importance, y = feature)) +
  geom_bar(
    stat  = "identity",
    fill  = "#1f8dd6",
    width = 0.7
  ) +
  scale_x_continuous(
    limits = c(0, max(shap_prototal_bar$importance) * 1.08),
    expand = c(0, 0)
    #breaks = c(0, 0.1, 0.2, 0.3, 0.4)
  ) +
  labs(
    #title    = "SHAP Feature Importance",
    #subtitle = "IRT - Overall Survival (Gradient Boosting)",
    x        = "mean(|SHAP value|) (average impact on model output magnitude)",
    y        = NULL
  ) +
  theme_bw()+
  theme(
    #plot.title    = element_text(hjust = 0.5, size = 13),
    #plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 14, color = "black"),
    axis.text.x = element_text(size = 15.5),
    axis.title.x  = element_text(size = 15, margin = margin(t = 8), face = "bold"),
    axis.line.x   = element_line(color = "black", linewidth = 0.5),
    panel.grid    = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin   = margin(t = 8, r = 20, b = 10, l = 10)
  )

print(p2)

ggsave("plot/supplement4a.tiff",
       plot = p2,
       width = 8,
       height = 6,
       units = "in",
       dpi = 300)


# --- SHAP Plot (scatter plot)
# - irt model OS
shap_value_irt_bar <- read.csv(here("data","MLdata","shap_value_irt.csv")) %>%
                    select(-PROGGI,-ETHNICITY,-RACE,-CANCERTYPE,-PO, -y_true) %>%
                    rename("IV TREATMENT" = IV, EMAXPS = PROGPS, EMAXPD = PROGPD)

irt_order <- c("SLPPD", "SLPGI", "IV TREATMENT", "SLPPS", "BASEPD",
                "AGE", "KDPD", "BASEPS", "KDPS", "BASEGI",
                "EMAXPS", "KDGI", "HEIGHT", "EMAXPD")

irt_feature_order <- rev(irt_order)

irt_feature_value <- read.csv(here("data","MLdata","train_OS.csv")) %>%
                select(-DEATH,-SURVIVAL_TIME)

median(irt_feature_value$ETHNICITY)


library(tidyr)

shap_long <- shap_value_iRACEshap_long <- shap_value_irt_bar %>%
  mutate(sample_id = row_number()) %>%
  pivot_longer(-sample_id, names_to = "feature", values_to = "shap_value") %>%
  mutate(feature = factor(feature, levels = irt_feature_order))

library(ggbeeswarm)
p <- ggplot(shap_long, aes(x = shap_value, y = feature, color = shap_value)) +
  geom_vline(xintercept = 0, color = "gray50", linewidth = 1) +
  geom_quasirandom(
    groupOnX  = FALSE,  
    size      = 1.5,
    alpha     = 0.8,
    bandwidth = 0.2     #point cluster level
  ) +
  scale_color_gradientn(
    colors = c("#3B82C4", "#6A5ACD", "#C71585", "#D42B2B"),
    values = scales::rescale(c(-1, -0.2, 0.2, 1)),
    name   = "Feature value",
    guide  = guide_colorbar(
      barheight      = unit(8, "cm"),
      barwidth       = unit(0.4, "cm"),
      title.position = "right",
      title.hjust    = 0.5,
      title.vjust    = 0.5,
      label.theme    = element_text(color = NA),
      ticks          = FALSE
    )
  ) +
  scale_x_continuous(breaks = seq(-1.25, 0.75, by = 0.25)) +
  labs(
    x = "SHAP value (impact on model output)",
    y = NULL
  ) +
  theme_bw() +
  theme(
    axis.text    = element_text(size = 14, color = "black"),
    axis.text.x  = element_text(size = 15, color = "black"),
    axis.title.x       = element_text(size = 14, margin = margin(t = 8)),
    axis.line.x        = element_line(color = "black"),
    panel.grid.major.x = element_line(color = "gray92", linewidth = 0.3),
    panel.grid.major.y = element_blank(),
    legend.position    = "right",
    legend.title       = element_text(size = 15, hjust = 0.5, angle =90, margin = margin(l = -11)),
    plot.background    = element_rect(fill = "white", color = NA),
    plot.margin        = margin(t = 15, r = 20, b = 10, l = 10)
  )

print(p)















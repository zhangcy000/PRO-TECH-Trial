#1. Load packages and set up environment
rm(list=ls())

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(here)
library(ggh4x)

#=====================================================================================
# Function to generate the data for VPC
#=====================================================================================
draw_vpc_categorical <- function(project.name, item.number, simdata.list) {
  
  # Read empirical distribution data
  empirical <- read.csv(here("data","vpc_train_data",
                        paste0(project.name, "/ChartsData/VisualPredictiveCheck/y",item.number, "_distribution.txt")), header = TRUE)
  
  # Get simulated data for this item
  simdata <- simdata.list[[item.number]]
  
  # Extract bin edges from empirical data
  bins <- unique(c(empirical$binsTimeBefore, empirical$binsTimeAfter))
  bins <- sort(bins)
  binsmiddle <- (bins[2:length(bins)] + bins[1:(length(bins)-1)]) / 2
  
  # Assign simulated data to bins
  simdata$binindex <- findInterval(simdata$time, bins, all.inside = TRUE)
  
  # Get the column name for this item's simulated values
  y_col <- paste0("sim_y", item.number)
  
  # Calculate proportions for each level (0,1,2,3) in each bin for each replicate
  prop_list <- list()
  
  for(cat in 0:3) {
    prop_cat <- aggregate(
      simdata[[y_col]] == cat,
      by = list(bin = simdata$binindex, rep = simdata$rep),
      FUN = mean
    )
    names(prop_cat)[3] <- "proportion"
    prop_cat$category <- cat
    prop_list[[cat + 1]] <- prop_cat
  }
  
  # Combine all categories
  all_props <- do.call(rbind, prop_list)
  
  # Calculate median, 2.5% and 97.5% percentiles across replicates for each bin and category
  pred_lower <- aggregate(
    all_props$proportion,
    by = list(bin = all_props$bin, category = all_props$category),
    FUN = function(x) quantile(x, 0.025)
  )
  names(pred_lower)[3] <- "lower"
  
  pred_median <- aggregate(
    all_props$proportion,
    by = list(bin = all_props$bin, category = all_props$category),
    FUN = function(x) quantile(x, 0.50)
  )
  names(pred_median)[3] <- "median"
  
  pred_upper <- aggregate(
    all_props$proportion,
    by = list(bin = all_props$bin, category = all_props$category),
    FUN = function(x) quantile(x, 0.975)
  )
  names(pred_upper)[3] <- "upper"
  
  # Combine all percentiles
  pred_intervals <- merge(pred_lower, pred_median, by = c("bin", "category"))
  pred_intervals <- merge(pred_intervals, pred_upper, by = c("bin", "category"))
  pred_intervals$bin_middle <- binsmiddle[pred_intervals$bin]
  
  # Calculate total scores based on the percentiles of simulation data
  total_score_intervals <- pred_intervals %>%
    group_by(bin) %>%
    summarise(
      lower = sum(lower * category),
      median = sum(median * category),
      upper = sum(upper * category)
    ) %>%
    mutate(bin_middle = binsmiddle[bin])
  
  # Calculate total score for empirical data
  empirical_total <- empirical %>%
    mutate(
      category_value = case_when(
        category == "[1-1]" ~ 3,
        category == "[2-2]" ~ 2,
        category == "[3-3]" ~ 1,
        category == "[4-4]" ~ 0
      ),
      bin_middle = (binsTimeBefore + binsTimeAfter) / 2
    ) %>%
    group_by(bin_middle) %>%
    summarise(
      empirical_score = sum(propCategory_empirical * category_value)
    )
  
  # Return data frame for plotting 
  vpc_data <- data.frame(
    TIME = total_score_intervals$bin_middle,
    exp_l = total_score_intervals$lower,
    exp_h = total_score_intervals$upper,
    exp_obs = empirical_total$empirical_score[match(total_score_intervals$bin_middle, empirical_total$bin_middle)],
    PROCTCAE = paste0("PROCTCAE_", sprintf("%03d", item.number))
  )
  
  return(vpc_data)
}

# Function to create combined data for all 14 items
vpc_data_all <- function(project.name, simdata.list) {
  
  # Create list to store data
  vpc_data_list <- list()
  
  # Generate VPC data for each of the 14 items
  for(i in 1:14) {
    vpc_data_list[[i]] <- draw_vpc_categorical(project.name, i, simdata.list)
  }
  
  # Combine all data
  vpc_exp <- do.call(rbind, vpc_data_list)
  
  return(vpc_exp)
}

#===============================================================================
# VPC as in Monolix: the same observation times as in the data set are used
# whatever the new simulated dropout (death) time is
#===============================================================================
project.name <- "test7"

# Import all simulation data for all items
simdata.list <- list()

for(i in 1:14) {
  sim_yi <- read.csv(here("data","vpc_train_data",paste0(project.name, "/ChartsData/VisualPredictiveCheck/y", 
                     i, "_simulations.txt")), header = TRUE) %>%
            select(-split,-color,-filter)
  
  #change the reversed score back to raw score and save into the big dataset
  y_col <- paste0("sim_y", i)
  sim_yi[[y_col]] <- 4 - sim_yi[[y_col]]
  simdata.list[[i]] <- sim_yi
}

# generate the dataset for vpc 
vpc_basic <- vpc_data_all(project.name, simdata.list)

PRO_map <- c(PROCTCAE_001 = "Activity",
             PROCTCAE_002 = "Depression",
             PROCTCAE_003 = "Appetite",
             PROCTCAE_004 = "Pain_frequency",
             PROCTCAE_005 = "Pain_severity", 
             PROCTCAE_006 = "Pain_interference",
             PROCTCAE_007 = "Nausea_frequency",
             PROCTCAE_008 = "Nausea_severity",
             PROCTCAE_009 = "Vomit_frequency",
             PROCTCAE_010 = "Diarrhea_frequency",
             PROCTCAE_011 = "Constipation_severity",
             PROCTCAE_012 = "Dyspnea_severity",
             PROCTCAE_013 = "Dyspnea_interference", 
             PROCTCAE_014 = "Insomnia_severity")

vpc_basic <- vpc_basic %>%
  mutate(PROCTCAE = case_when(
    grepl(names(PRO_map)[1], PROCTCAE, ignore.case = TRUE) ~ PRO_map[1],
    grepl(names(PRO_map)[2], PROCTCAE, ignore.case = TRUE) ~ PRO_map[2],
    grepl(names(PRO_map)[3], PROCTCAE, ignore.case = TRUE) ~ PRO_map[3],
    grepl(names(PRO_map)[4], PROCTCAE, ignore.case = TRUE) ~ PRO_map[4],
    grepl(names(PRO_map)[5], PROCTCAE, ignore.case = TRUE) ~ PRO_map[5],
    grepl(names(PRO_map)[6], PROCTCAE, ignore.case = TRUE) ~ PRO_map[6],
    grepl(names(PRO_map)[7], PROCTCAE, ignore.case = TRUE) ~ PRO_map[7],
    grepl(names(PRO_map)[8], PROCTCAE, ignore.case = TRUE) ~ PRO_map[8],
    grepl(names(PRO_map)[9], PROCTCAE, ignore.case = TRUE) ~ PRO_map[9],
    grepl(names(PRO_map)[10], PROCTCAE, ignore.case = TRUE) ~ PRO_map[10],
    grepl(names(PRO_map)[11], PROCTCAE, ignore.case = TRUE) ~ PRO_map[11],
    grepl(names(PRO_map)[12], PROCTCAE, ignore.case = TRUE) ~ PRO_map[12],
    grepl(names(PRO_map)[13], PROCTCAE, ignore.case = TRUE) ~ PRO_map[13],
    grepl(names(PRO_map)[14], PROCTCAE, ignore.case = TRUE) ~ PRO_map[14]
  ))


p1 <- ggplot(vpc_basic, aes(x = TIME)) +
  geom_ribbon(aes(ymin = exp_l, ymax = exp_h),fill = "grey60", alpha = 0.5) +
  geom_line(aes(y = exp_obs), color = "royalblue4", linewidth = 1) +
  #geom_point(aes(y = exp_obs), color = "royalblue4",size = 1.2) +
  labs(x = "Time (Day)", y = "Item Score") +
  scale_x_continuous(limits = c(0, 400)) +
  ylim(0, 3) +
  facet_wrap(~PROCTCAE, ncol = 5) +
  theme_bw() +
  theme(
    axis.text.x = element_text(hjust = 1, size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    strip.text = element_text(face = "bold", size = 11),
  )

print(p1)


#===============================================================================
# VPC with removal of observation points happening after dropout
#===============================================================================
project.name <- "test7"

#sim_survival <- read.csv("Event_simulations.txt",header = TRUE) %>%
#              select(rep,ID,time)

sim_survival <- read.csv(here("data","vpc_train_data",paste0(project.name, 
                          "/ChartsData/VisualPredictiveCheck/survival_simulations.txt")), header = TRUE) %>%
                select(rep,ID,time)

dropouttable <- sim_survival[seq(2, nrow(sim_survival), by = 2), 1:3]
colnames(dropouttable)[3] <- "dropouttime"

simdata.os.list <- list()
for(i in 1:14) {
  sim_yi <- simdata.list[[i]]
  
  sim_yi <- merge(sim_yi, dropouttable, by = c("ID", "rep"))
  
  sim_yi <- sim_yi[sim_yi$time < sim_yi$dropouttime, ]
  
  simdata.os.list[[i]] <- sim_yi
}

vpc_dropout <- vpc_data_all(project.name, simdata.os.list)

vpc_dropout <- vpc_dropout %>%
  mutate(PROCTCAE = case_when(
    grepl(names(PRO_map)[1], PROCTCAE, ignore.case = TRUE) ~ PRO_map[1],
    grepl(names(PRO_map)[2], PROCTCAE, ignore.case = TRUE) ~ PRO_map[2],
    grepl(names(PRO_map)[3], PROCTCAE, ignore.case = TRUE) ~ PRO_map[3],
    grepl(names(PRO_map)[4], PROCTCAE, ignore.case = TRUE) ~ PRO_map[4],
    grepl(names(PRO_map)[5], PROCTCAE, ignore.case = TRUE) ~ PRO_map[5],
    grepl(names(PRO_map)[6], PROCTCAE, ignore.case = TRUE) ~ PRO_map[6],
    grepl(names(PRO_map)[7], PROCTCAE, ignore.case = TRUE) ~ PRO_map[7],
    grepl(names(PRO_map)[8], PROCTCAE, ignore.case = TRUE) ~ PRO_map[8],
    grepl(names(PRO_map)[9], PROCTCAE, ignore.case = TRUE) ~ PRO_map[9],
    grepl(names(PRO_map)[10], PROCTCAE, ignore.case = TRUE) ~ PRO_map[10],
    grepl(names(PRO_map)[11], PROCTCAE, ignore.case = TRUE) ~ PRO_map[11],
    grepl(names(PRO_map)[12], PROCTCAE, ignore.case = TRUE) ~ PRO_map[12],
    grepl(names(PRO_map)[13], PROCTCAE, ignore.case = TRUE) ~ PRO_map[13],
    grepl(names(PRO_map)[14], PROCTCAE, ignore.case = TRUE) ~ PRO_map[14]
  ))

vpc_dropout2 <- vpc_dropout %>%
            mutate(PROCTCAE = factor(PROCTCAE, levels = c("Activity", "Depression", "Appetite",
                              "Pain_frequency","Pain_severity","Pain_interference","Nausea_frequency",
                              "Nausea_severity","Vomit_frequency","Diarrhea_frequency","Constipation_severity",
                              "Dyspnea_severity","Dyspnea_interference","Insomnia_severity")))

strip_colors <- c(
  "Activity"              = "#AED6F1",  # Psychological
  "Depression"            = "#AED6F1",
  "Insomnia_severity"     = "#AED6F1",
  "Pain_frequency"        = "#FAD7A0",  # Pain/Physical
  "Pain_severity"         = "#FAD7A0",
  "Pain_interference"     = "#FAD7A0",
  "Dyspnea_severity"      = "#FAD7A0",
  "Dyspnea_interference"  = "#FAD7A0",
  "Appetite"              = "#A9DFBF",  # Gastrointestinal
  "Nausea_frequency"      = "#A9DFBF",
  "Nausea_severity"       = "#A9DFBF",
  "Vomit_frequency"       = "#A9DFBF",
  "Diarrhea_frequency"    = "#A9DFBF",
  "Constipation_severity" = "#A9DFBF"
)

grouped_order <- c(
  "Activity", "Depression", "Insomnia_severity",
  "Pain_frequency", "Pain_severity", "Pain_interference", "Dyspnea_severity", "Dyspnea_interference",
  "Appetite", "Nausea_frequency", "Nausea_severity", "Vomit_frequency",
  "Diarrhea_frequency", "Constipation_severity"
)

vpc_dropout2 <- vpc_dropout2 %>%
  mutate(PROCTCAE = factor(PROCTCAE, levels = grouped_order))

p2 <- ggplot(vpc_dropout2, aes(x = TIME)) +
  geom_ribbon(aes(ymin = exp_l, ymax = exp_h), fill = "grey60", alpha = 0.7) +
  geom_line(aes(y = exp_obs), color = "royalblue4", linewidth = 1) +
  #geom_point(aes(y = exp_obs), color = "royalblue4",size = 1.2) +
  labs(x = "Time (days)", y = "Item Score") +
  scale_x_continuous(limits = c(0, 400)) +
  ylim(0, 3) +
  ggh4x::facet_wrap2(
    ~PROCTCAE, ncol = 5,
    strip = strip_themed(
      background_x = elem_list_rect(fill = strip_colors[levels(vpc_dropout2$PROCTCAE)])
    )
  ) +
  theme_bw() +
  theme(
    axis.title = element_text(size = 15, face = "bold"),
    axis.text.x = element_text(size = 15, color = "black"),
    axis.text.y = element_text(size = 15, color = "black"),
    strip.text = element_text(face = "bold", size = 15, color = "black")
  )

print(p2)


ggsave("plot/figure3c.tiff",
        plot = p2,
        width = 14,
        height = 6,
        units = "in",
        dpi = 300)

poster <- vpc_dropout2 %>%
        filter(PROCTCAE %in% c("Pain_severity", "Pain_interference", "Diarrhea_frequency",
                               "Dyspnea_interference"))


p3 <- ggplot(poster, aes(x = TIME)) +
  geom_ribbon(aes(ymin = exp_l, ymax = exp_h),fill = "grey60", alpha = 0.7) +
  geom_line(aes(y = exp_obs), color = "royalblue4", linewidth = 1) +
  #geom_point(aes(y = exp_obs), color = "royalblue4",size = 1.2) +
  scale_x_continuous(limits = c(0, 400)) +
  labs(x = "Time (days)", y = "Item Score") +
  ylim(0, 3) +
  facet_wrap(~PROCTCAE, ncol = 2) +
  theme_bw() +
  theme(
    axis.title = element_text(size = 16, face = "bold"),
    axis.text.x = element_text(size = 15, color = "black"),
    axis.text.y = element_text(size = 15, color = "black"),
    strip.background = element_rect(fill = "lightgray"),
    strip.text = element_text(face = "bold", size = 15,color ="black")
  )

print(p3)


ggsave("plot/poster.tiff",
        plot = p3,
        width = 6,
        height = 6,
        units = "in",
        dpi = 300)


















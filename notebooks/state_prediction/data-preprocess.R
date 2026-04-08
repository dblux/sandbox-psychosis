library(magrittr)
source('R/subset.R')

### Load data ###

file <- 'data/astral/processed/reprocessed-data-renamed.csv'
raw <- read.csv(file, row.names = 1)
raw[raw == 0] <- NA

# Metadata
file <- 'data/astral/metadata/metadata-psy_602_16-v1.csv'
metadata <- read.csv(file, row.names = 1)

metadata$extraction_date <- factor(metadata$extraction_date)
metadata$run_datetime <- as.POSIXct(
  metadata$run_datetime,
  format = "%d/%m/%y %H:%M",
  tz = "UTC"
)
metadata$collection_datetime <- as.POSIXct(
  metadata$collection_datetime,
  format = "%d/%m/%y %H:%M",
  tz = "UTC"
)


### Subset data ###

# Data versions
# v1: Before batch correction
# TODO: v2: After correcting for continuous batch effects
# TODO: v3: After correcting for continuous batch effects followed by discrete batch effects

lyriks <- raw[, startsWith(colnames(raw), 'L')] 
lyriks_full <- log2(na.omit(lyriks))

outliers_sn <- c('L0626C', 'L0018C')
lyriks_265_396 <- subset_cols(
  lyriks_full,
  metadata,
  !(sn %in% outliers_sn)
)
stateless_sid <- 'L0365S_12'
lyriks_265_395 <- lyriks_265_396[, colnames(lyriks_265_396) != stateless_sid]

### Feature selection ###

# TODO: Calculate pearson correlation between feature and collection_datetime
# and run_datetime for control samples in 4/9/24
# Remove features with high correlation with either variable

ctrl_0409 <- subset_cols(
  lyriks_265_395,
  metadata,
  group == 'Healthy control'
    & extraction_date == '4/9/24'
    & !(sn %in% outliers_sn)
)
metadata_ctrl_0409 <- metadata[colnames(ctrl_0409), ]
# Convert to days since earliest 
metadata_ctrl_0409$collection_datetime_days <- as.numeric(difftime(
  metadata_ctrl_0409$collection_datetime,
  min(metadata_ctrl_0409$collection_datetime, na.rm = TRUE),
  units = "days"
))
metadata_ctrl_0409$run_datetime_days <- as.numeric(difftime(
  metadata_ctrl_0409$run_datetime,
  min(metadata_ctrl_0409$run_datetime, na.rm = TRUE),
  units = "days"
))

ctrl_0409_cor <- cor(
  t(ctrl_0409),
  metadata_ctrl_0409[, c("collection_datetime_days", "run_datetime_days")],
  method = "pearson",
  use = "complete.obs"
)

cor_threshold <- 0.3
has_continuous_effects <- apply(abs(ctrl_0409_cor) > cor_threshold, 1, any)
lyriks_214_395 <- lyriks_265_395[!has_continuous_effects, ]

prot_highvar <- apply(lyriks_214_395, 1, var, na.rm = TRUE) %>%
  sort(decreasing = TRUE) %>%
  head(50)
lyriks_50_395 <- lyriks_214_395[names(prot_highvar), ]

write.csv(lyriks_265_395, 'data/tmp/sehwan/lyriks_265_395.csv')
write.csv(lyriks_214_395, 'data/tmp/sehwan/lyriks_214_395.csv')
write.csv(lyriks_50_395, 'data/tmp/sehwan/lyriks_50_395.csv')

metadata_265_395 <- metadata[colnames(lyriks_265_395), ]
table(
  metadata_265_395$state,
  metadata_265_395$extraction_date
)

# metadata_cmc <- metadata[colnames(cmc), ]
# # Relevel with convert batch as reference
# metadata_cmc$extraction_date <- droplevels(metadata_cmc$extraction_date)
# print(levels(metadata_cmc$extraction_date))


# ### Subset control samples for slope estimation ###
# 
# ctrl_0409 <- subset_cols(
#   lyriks_full,
#   metadata,
#   group == 'Healthy control'
#     & extraction_date == '4/9/24'
#     & !(sn %in% outliers_sn)
# )
# metadata_ctrl_0409 <- metadata_cmc[colnames(ctrl_0409), ] 
# 
# # # Convert to days since earliest 
# # metadata_ctrl_0409$collection_datetime_days <- as.numeric(difftime(
# #   metadata_ctrl_0409$collection_datetime,
# #   min(metadata_ctrl_0409$collection_datetime, na.rm = TRUE),
# #   units = "days"
# # ))
# # metadata_ctrl_0409$run_datetime_days <- as.numeric(difftime(
# #   metadata_ctrl_0409$run_datetime,
# #   min(metadata_ctrl_0409$run_datetime, na.rm = TRUE),
# #   units = "days"
# # ))
# 
# mod <- model.matrix(
#    ~ run_datetime_days + collection_datetime_days,
#    data = metadata_ctrl_0409
# )
# 
# fit <- lmFit(ctrl_0409, mod)
# betas <- coef(fit)
# 
# # # Check for collinearity through inflation factors
# # se_run <- fit$stdev.unscaled[, "run_datetime_days"] * sqrt(fit$sigma^2)
# # mod_run <- model.matrix(~ run_datetime_days, data = metadata_ctrl_0409)
# # fit_run <- lmFit(ctrl_0409, mod_run)
# # se_run2 <- fit_run$stdev.unscaled[, "run_datetime_days"] * sqrt(fit_run$sigma^2)
# # inflation_factor_run <- se_run / se_run2
# # summary(inflation_factor_run)
# 
# ### Batch correction for continuous covariates on entire CMC subset ###
# 
# cmc_corr <- cmc - outer(
#   betas[, "run_datetime_days"],
#   metadata_cmc$run_datetime_days
# ) - outer(
#   betas[, "collection_datetime_days"],
#   metadata_cmc$collection_datetime_days
# )
# 
# file <- 'data/tmp/cmc-lm.csv'
# write.csv(cmc_corr, file)
# print(file)
# 
# ### Batch correct for extraction date ###
# 
# # ComBat correction
# print(table(metadata_cmc$extraction_date, metadata_cmc$group))
# mod_group <- model.matrix(~ group, data = metadata_cmc)
# 
# mc_combat <- ComBat(
#   cmc_corr,
#   batch = metadata_cmc$extraction_date,
#   mod = mod_group,
#   par.prior = TRUE,
#   ref.batch = '4/9/24'
# )
# 
# file <- 'data/tmp/cmc-combat_0409.csv'
# write.csv(cmc_combat, file)
# print(file)
# 
# cmc_combat <- ComBat(
#   cmc_corr,
#   batch = metadata_cmc$extraction_date,
#   mod = mod_group,
#   par.prior = TRUE,
#   ref.batch = '5/9/24'
# )
# 
# file <- 'data/tmp/cmc-combat_0509.csv'
# write.csv(cmc_combat, file)
# print(file)
# 
# # Limma correction (linear model, no Bayesian shrinkage)
# # Assumption: No interaction between batch effects due to run_datetime and extraction_date
# metadata_cmc$extraction_date <- relevel(
#   metadata_cmc$extraction_date,
#   ref = '4/9/24'
# )
# cmc_limma <- removeBatchEffect(
#   cmc_corr,
#   batch = metadata_cmc$extraction_date,
#   design = mod_group 
# )
# 
# file <- 'data/tmp/cmc-limma.csv'
# write.csv(cmc_limma, file)
# print(file)

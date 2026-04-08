library(limma)
library(sva)
source('R/subset.R')

### Load data ###

file <- 'data/astral/reprocessed-data-renamed.csv'
data <- read.csv(file, row.names = 1)
data[data == 0] <- NA

lyriks <- data[, startsWith(colnames(data), 'L')] 
lyriks_full <- log2(na.omit(lyriks))

file <- 'data/astral/metadata-psy_602_16-v1.csv'
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

### Preprocess data ###

# Subset data

### Psychosis signature (cvt, mnt, ctrl) ###

outliers_sn <- c('L0626C', 'L0018C')
cmc <- subset_cols(
  lyriks_full,
  metadata,
  group %in% c("Convert", "Maintain", "Healthy control") &
  !(sn %in% outliers_sn)
)

metadata_cmc <- metadata[colnames(cmc), ]
# Relevel with convert batch as reference
metadata_cmc$extraction_date <- droplevels(metadata_cmc$extraction_date)
print(levels(metadata_cmc$extraction_date))

# Convert to days since earliest 
metadata_cmc$run_datetime_days <- as.numeric(difftime(
  metadata_cmc$run_datetime,
  min(metadata_cmc$run_datetime, na.rm = TRUE),
  units = "days"
))
metadata_cmc$collection_datetime_days <- as.numeric(difftime(
  metadata_cmc$collection_datetime,
  min(metadata_cmc$collection_datetime, na.rm = TRUE),
  units = "days"
))
# metadata_cmc$collection_datetime_centered <- 
#   metadata_cmc$collection_datetime_days - mean(metadata_cmc$collection_datetime_days)
# mod <- model.matrix(~group, data = metadata_cmc)

### Subset control samples for slope estimation ###

ctrl_0409 <- subset_cols(
  lyriks_full,
  metadata,
  group == 'Healthy control'
    & extraction_date == '4/9/24'
    & !(sn %in% outliers_sn)
)
metadata_ctrl_0409 <- metadata_cmc[colnames(ctrl_0409), ] 

# # Convert to days since earliest 
# metadata_ctrl_0409$collection_datetime_days <- as.numeric(difftime(
#   metadata_ctrl_0409$collection_datetime,
#   min(metadata_ctrl_0409$collection_datetime, na.rm = TRUE),
#   units = "days"
# ))
# metadata_ctrl_0409$collection_datetime_days <- as.numeric(difftime(
#   metadata_ctrl_0409$collection_datetime,
#   min(metadata_ctrl_0409$collection_datetime, na.rm = TRUE),
#   units = "days"
# ))

mod <- model.matrix(
   ~ run_datetime_days + collection_datetime_days,
   data = metadata_ctrl_0409
)

fit <- lmFit(ctrl_0409, mod)
betas <- coef(fit)

# # Check for collinearity through inflation factors
# se_run <- fit$stdev.unscaled[, "run_datetime_days"] * sqrt(fit$sigma^2)
# mod_run <- model.matrix(~ run_datetime_days, data = metadata_ctrl_0409)
# fit_run <- lmFit(ctrl_0409, mod_run)
# se_run2 <- fit_run$stdev.unscaled[, "run_datetime_days"] * sqrt(fit_run$sigma^2)
# inflation_factor_run <- se_run / se_run2
# summary(inflation_factor_run)

### Batch correction for continuous covariates on entire CMC subset ###

cmc_corr <- cmc - outer(
  betas[, "run_datetime_days"],
  metadata_cmc$run_datetime_days
) - outer(
  betas[, "collection_datetime_days"],
  metadata_cmc$collection_datetime_days
)

file <- 'data/tmp/cmc-lm.csv'
write.csv(cmc_corr, file)
print(file)

### Batch correct for extraction date ###

# ComBat correction
print(table(metadata_cmc$extraction_date, metadata_cmc$group))
mod_group <- model.matrix(~ group, data = metadata_cmc)

mc_combat <- ComBat(
  cmc_corr,
  batch = metadata_cmc$extraction_date,
  mod = mod_group,
  par.prior = TRUE,
  ref.batch = '4/9/24'
)

file <- 'data/tmp/cmc-combat_0409.csv'
write.csv(cmc_combat, file)
print(file)

cmc_combat <- ComBat(
  cmc_corr,
  batch = metadata_cmc$extraction_date,
  mod = mod_group,
  par.prior = TRUE,
  ref.batch = '5/9/24'
)

file <- 'data/tmp/cmc-combat_0509.csv'
write.csv(cmc_combat, file)
print(file)

# Limma correction (linear model, no Bayesian shrinkage)
# Assumption: No interaction between batch effects due to run_datetime and extraction_date
metadata_cmc$extraction_date <- relevel(
  metadata_cmc$extraction_date,
  ref = '4/9/24'
)
cmc_limma <- removeBatchEffect(
  cmc_corr,
  batch = metadata_cmc$extraction_date,
  design = mod_group 
)

file <- 'data/tmp/cmc-limma.csv'
write.csv(cmc_limma, file)
print(file)

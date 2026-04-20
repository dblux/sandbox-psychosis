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

## Subset data

# # Identifying outlier according to runtime
# subset(
#   metadata,
#   extraction_date == '28/8/24' & group == 'Healthy control',
#   select = 'run_datetime'
# )

### Psychosis signature (cvt, mnt, ctrl) ###

outliers_sn <- c('L0076C', 'L0018C', 'L0626C')
missing_sn <- c('L0365S', 'L0417S')
lyriks387 <- subset_cols(
  lyriks_full,
  metadata,
  !(sn %in% c(outliers_sn, missing_sn))
)
print(dim(lyriks387))

metadata387 <- metadata[colnames(lyriks387), ]
# Relevel with convert batch as reference
metadata387$extraction_date <- droplevels(metadata387$extraction_date)
print(levels(metadata387$extraction_date))

# Convert to days since earliest 
metadata387$run_datetime_days <- as.numeric(difftime(
  metadata387$run_datetime,
  min(metadata387$run_datetime, na.rm = TRUE),
  units = "days"
))
metadata387$collection_datetime_days <- as.numeric(difftime(
  metadata387$collection_datetime,
  min(metadata387$collection_datetime, na.rm = TRUE),
  units = "days"
))
# metadata387$collection_datetime_centered <- 
#   metadata387$collection_datetime_days - mean(metadata387$collection_datetime_days)
# mod <- model.matrix(~group, data = metadata387)

### Subset control samples for slope estimation ###

ctrl_0409 <- subset_cols(
  lyriks387,
  metadata,
  group == 'Healthy control'
    & extraction_date == '4/9/24'
    # & !(sn %in% outliers_sn)
)
metadata_ctrl_0409 <- metadata387[colnames(ctrl_0409), ] 

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

lyriks387_corr <- lyriks387 - outer(
  betas[, "run_datetime_days"],
  metadata387$run_datetime_days
) - outer(
  betas[, "collection_datetime_days"],
  metadata387$collection_datetime_days
)


### Batch correct for extraction date ###

# ComBat correction
print(table(metadata387$extraction_date, metadata387$group))
mod_group <- model.matrix(~ group, data = metadata387)

lyriks387_combat <- ComBat(
  lyriks387_corr,
  batch = metadata387$extraction_date,
  mod = mod_group,
  par.prior = TRUE,
  ref.batch = '4/9/24'
)

if (any(is.na(lyriks387_combat)))
  cat('Missing values present in corrected data!')

file <- 'data/tmp/lyriks387-combat_0409.csv'
write.csv(lyriks387_combat, file)
print(file)


# ## Alternative: Limma correction (linear model, no Bayesian shrinkage)
# # Assumption: No interaction between batch effects due to run_datetime
# # and extraction_date
# metadata387$extraction_date <- relevel(
#   metadata387$extraction_date,
#   ref = '4/9/24'
# )
# lyriks387_limma <- removeBatchEffect(
#   lyriks387_corr,
#   batch = metadata387$extraction_date,
#   design = mod_group 
# )


# # Ad hoc - Check NAs in data
# # Conclusion: Due to missing collection dates
# file <- 'data/tmp/lyriks387-combat_0409.csv'
# lyriks387_combat <- read.csv(file, row.names = 1)
# 
# sids <- c('L0365S_12', 'L0417S_24')
# 
# lyriks387_combat[, sids]
# lyriks_full[, sids]
# lyriks387[, sids]
# lyriks387_corr[, sids]

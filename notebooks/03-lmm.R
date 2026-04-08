library(variancePartition)
library(dplyr)


#' Subset columns according to features in annotation
#'
#' Colnames of X has to match with rownames of annot.
#'
#' @example subset_cols(X, annot, subtype == 'BCR-ABL' & class_info == 'D0')
subset_cols <- function(X, metadata, ...) {
  X[, rownames(subset(metadata[colnames(X), , drop = FALSE], ...))]
}


##### Load data #####

file <- 'data/astral/reprocessed-data-renamed.csv'
raw <- read.csv(file, row.names = 1)
lyriks <- raw[, startsWith(colnames(raw), 'L')]
lyriks[lyriks == 0] <- NA
lyriks <- log2(lyriks)
lyriks_full <- lyriks[rowSums(is.na(lyriks)) == 0, ]

uniprot_gene_map <- raw$Gene
names(uniprot_gene_map) <- rownames(raw)

file <- 'data/astral/metadata-psy_602_16.csv'
metadata <- read.csv(file, row.names = 1)
metadata$group <- factor(metadata$group)
metadata$gender <- factor(metadata$gender)
metadata$ethnicity <- factor(metadata$ethnicity)
metadata$smoking <- factor(metadata$smoking)
metadata$extraction_date <- factor(metadata$extraction_date)
metadata$sn <- factor(metadata$sn)
colnames(metadata)

##### Prepare data #####

ctrl_mnt <- subset_cols(
  lyriks_full, metadata,
  group %in% c('Healthy control', 'Maintain') & study == 'LYRIKS'
)
metadata_cm <- metadata[colnames(ctrl_mnt), ]

##### Demographic table (baseline) #####

# To determine what covariates to include
baseline_cm <- droplevels(metadata_cm[metadata_cm$timepoint == 0, ])
print(table(baseline_cm$group))

ct_gender <- table(baseline_cm$gender, baseline_cm$group)
fisher_gender <- fisher.test(ct_gender)
print(ct_gender)
print(round(prop.table(ct_gender, 2) * 100, 1))
print(fisher_gender)

ct_smoking <- table(baseline_cm$smoking, baseline_cm$group)
fisher_smoking <- fisher.test(ct_smoking)
print(ct_smoking)
print(round(prop.table(ct_smoking, 2) * 100, 1))
print(fisher_smoking)

baseline_cm$ethnicity[
  baseline_cm$ethnicity %in% c('Boyanese', 'Burmese', 'Javanese', 'Indian')
] <- 'Others'
ct_ethnicity <- table(baseline_cm$ethnicity, baseline_cm$group)
fisher_ethnicity <- fisher.test(ct_ethnicity)
print(ct_ethnicity)
print(round(prop.table(ct_ethnicity, 2) * 100, 1))
print(fisher_ethnicity)

group_age <- baseline_cm %>%
  group_by(group) %>%
  summarize(mean(age), sd(age))
unpaired_ttest <- t.test(age ~ group, data = baseline_cm)
print(group_age)
print(unpaired_ttest)

group_bmi <- baseline_cm %>%
  group_by(group) %>%
  summarize(mean(bmi), sd(bmi))
unpaired_ttest <- t.test(bmi ~ group, data = baseline_cm)
print(group_bmi)
print(unpaired_ttest)

# dream uses topTable (BH) for multiple testing correction
# TODO: Look at proteins returned
print(dim(ctrl_mnt))
metadata_cm$collection_days <- as.numeric(difftime(
  metadata_cm$collection_datetime,
  min(metadata_cm$collection_datetime), units = 'days'
))
metadata_cm$run_days <- as.numeric(difftime(
  metadata_cm$run_datetime,
  min(metadata_cm$run_datetime), units = 'days'
))

formula <- ~ group + extraction_date + collection_days + (1|sn)
fit <- dream(ctrl_mnt, formula, metadata_cm)
res <- topTable(fit, coef="groupMaintain", number=Inf)
res <- mutate(res, Gene = uniprot_gene_map[rownames(res)], .before = 1)

file <- 'tmp/dream_4-uhr_signature.csv'
write.csv(res, file)

print(sum(res['P.Value'] < 0.05))
print(sum(res['adj.P.Val'] < 0.05))
uid_p005 <- rownames(res)[res['P.Value'] < 0.05]
genes_p005 <- uniprot_gene_map[uid_p005]
print(unname(genes_p005))

# TODO: Ad hoc check
file <- 'tmp/astral/conversion-pairedttest.csv'
paired_ttest <- read.csv(file, row.names = 1)
paired_ttest

qvalue = p.adjust(paired_ttest$pvalue, method = 'BH')
qvalue

# Do baseline comparison between control and UHR present in our data

# Decide to use splines for run_datetime and collection_datetime

# Assign covariates to random or fixed
# Decide on a minimal set of covariates


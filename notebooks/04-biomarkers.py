import sys

import os
import umap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
from scipy.stats import spearmanr, ttest_rel
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# sys.path.append(os.getcwd())
import biopy.utils as bp


filepath = 'data/astral/metadata/metadata-psy_602_16-v1.csv'
metadata = pd.read_csv(filepath, index_col=0)
metadata.run_datetime = pd.to_datetime(
    metadata.run_datetime,
    format='mixed'
)
metadata['run_datenum'] = mdates.date2num(metadata.run_datetime)
metadata.collection_datetime = pd.to_datetime(
    metadata.collection_datetime,
    format='mixed'
)
metadata['collection_datenum'] = mdates.date2num(metadata.collection_datetime)

filepath = 'data/astral/metadata/metadata-csa_200_37.csv'
metadata_csa = pd.read_csv(filepath, index_col=0)
metadata_csa.comorbidities.fillna('No', inplace=True)
metadata_csa.collection_datetime = pd.to_datetime(
    metadata_csa.collection_datetime,
    format='mixed'
)

# OLINKS
# filepath = 'data/astral/etc/olink-csa.csv'
# olink_stats = pd.read_csv(filepath, index_col=0)
# 
# filepath = 'data/astral/etc/annotation-olink_75.csv'
# annot_olink = pd.read_csv(filepath, index_col=0)
# map_olink_uniprot = {
#     k: v for k, v in zip(annot_olink.index, annot_olink.uniprot)
# }

# Data
filepath = 'data/astral/processed/csa-knn5.csv'
csa_knn = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/processed/lyriks_605_402_01-knn5.csv'
lyriks_knn = pd.read_csv(filepath, index_col=0)

psy_ids = csa_knn.columns.tolist()
psy_ids.extend(lyriks_knn.columns.tolist())

# with open('tmp/astral/psy-ids.txt', 'w') as f:
#     f.write('\n'.join(psy_ids))

filepath = 'data/astral/processed/lyriks_605_402_01-combat_knn5.csv'
lyriks_combat = pd.read_csv(filepath, index_col=0)

psy_knn = lyriks_knn.join(csa_knn, how='inner')
psy_combat = lyriks_combat.join(csa_knn, how='inner')

filepath = 'data/astral/processed/reprocessed-data-renamed.csv'
data = pd.read_csv(filepath, index_col=0)
data.replace(0, np.nan, inplace=True)
psy = np.log2(data.iloc[:, 2:-46])

# LYRIKS + CSA
psy_full = psy.dropna()

# LYRIKS
lyriks = psy.iloc[:, psy.columns.str.startswith('L')].copy()
lyriks_full = lyriks.dropna()

csa = psy.iloc[:, psy.columns.str.startswith('CA')].copy()
csa_full = csa.dropna()
csa_574 = csa.loc[psy_knn.index]
csa_zero_574 = csa_574.fillna(0)
csa_knn_574 = csa_knn.loc[psy_knn.index]

map_uniprot_gene = {k: v for k, v in zip(data.index, data.Gene)}

##### Check metadata coverage #####

psy_ids = csa_knn.columns.append(lyriks_knn.columns)
psy_ids.shape
psy_ids.isin(metadata.index).sum()

# CA114 missing metadata
# smoking missing for CA155 
print(metadata.info())
for feature in metadata.columns:
    print(feature)
    print(metadata.index[metadata[feature].isna()].tolist())

# L0073S_24, L0417S_24
# L0567 has missing blood collection info
# lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0073')]
# lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0417')]
# lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0365')]

##### Transform datetime data #####

# TODO: Transform according to specific model (earliest day is different for different subsets)
metadata['run_days'] = (
    metadata.run_datetime - metadata.run_datetime.min()
) / np.timedelta64(1, "D")
metadata['collection_days'] = (
    metadata.collection_datetime - metadata.collection_datetime.min()
) / np.timedelta64(1, "D")

metadata['collection_date_sec'] = mdates.date2num(metadata.collection_datetime)
metadata.columns

##### CSA #####

# CA086, CA114, CA115 not present in Astral CSA dataset
missing_samples = metadata_csa.index[~metadata_csa.index.isin(csa_full.columns)]
metadata_csa_197 = metadata_csa.loc[~metadata_csa.index.isin(missing_samples)].copy()
# TODO: Remove imputation of smoking
metadata_csa_197.loc['CA155', 'smoking'] = '0'

metadata_csa_197['collection_days'] = (
    metadata_csa_197.collection_datetime -
    metadata_csa_197.collection_datetime.min()
) / np.timedelta64(1, "D")
metadata_csa_197['collection_date_sec'] = mdates.date2num(
    metadata_csa_197.collection_datetime
)

lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0365')]

### Correlation: Expression with collection days ###

# To decide between linear or splines modelling of collection days

# collate stats in a dataframe
rows = []
for prot in csa_full.index:
    corr = spearmanr(
        csa_full.loc[prot],
        metadata_csa_197.loc[csa_full.columns, 'collection_days']
    )
    row = {
        'protein': prot,
        'spearman_r': corr.correlation,
        'p_value': corr.pvalue
    }
    rows.append(row)
    
corr_stats = pd.DataFrame(rows).set_index('protein')
corr_stats.sort_values('spearman_r', ascending=False, key=abs, inplace=True)
prots_highcorr = corr_stats.index[:20]

for prot in prots_highcorr:
    # plot expression against collection days
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=metadata_csa_197.loc[csa_full.columns, 'collection_days'],
        y=csa_full.loc[prot],
        hue=metadata_csa_197.group,
        ax=ax
    )
    ax.set_title(f'{prot} (r = {corr_stats.loc[prot, "spearman_r"]:.2f})')
    filepath = f'tmp/astral/fig/days_expr-{prot}.pdf'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

plt.close()


### Linear model with splines ###
# Model days using splines

metadata_csa_197['collection_days_centered'] = metadata_csa_197.collection_days - metadata_csa_197.collection_days.mean()

covariates = ['group', 'bmi', 'age', 'gender', 'smoking', 'collection_days_centered']
data = pd.concat([
    csa_full.transpose(),
    metadata_csa_197.loc[csa_full.columns, covariates],
], axis=1)
data.group.replace({
    'Healthy control': 0,
    'Antipsychotic responsive': 1,
    'Clozapine responsive': 1,
    'Clozapine resistant': 1
}, inplace=True)

# Fit linear model with cubic spline 
pvalues = []
coefs = []
n_cov = len(covariates)
cov_string =  'cr(collection_days_centered, df=6) + bmi + age + C(gender) + C(smoking) + C(group)'
for prot in data.columns[:-n_cov]:
    expression = f'{prot} ~ ' + cov_string
    model = ols(expression, data=data).fit()
    print(expression)
    pvalues.append(model.pvalues['C(group)[T.1]'])
    coefs.append(model.params['C(group)[T.1]'])
    # table = sm.stats.anova_lm(model, typ=2)

_, qvalues, _, _ = multipletests(
    pvalues, alpha=0.05, method='fdr_bh'
)
stats = pd.DataFrame(
    {'p': pvalues, 'q': qvalues},
    index=csa_full.index
)
genes = stats.index.map(map_uniprot_gene)
stats.insert(0, 'Gene', genes)
stats.sort_values('q', inplace=True)
# filepath = 'tmp/astral/bm-schizo-stats.csv'
stats.to_csv(filepath, index=True)


### Overlap analysis ###

filepath = 'tmp/csa/biomarkers/hgnc-schizo.txt'
with open(filepath) as f:
    hgnc_schizo = set([line.strip() for line in f.readlines()])

filepath = 'data/astral/etc/silver_standard.csv'
silver = pd.read_csv(filepath)
silver_schizo = set(silver.loc[silver.signature == 'schizophrenia', 'gene'])

filepath = 'tmp/astral/lyriks402/new/biomarkers/biomarkers-ancova.csv'
bm_ancova = pd.read_csv(filepath, index_col=0)
hgnc_ancova = set(bm_ancova.Gene)

filepath = 'tmp/astral/lyriks402/new/biomarkers/biomarkers-elasticnet.csv'
bm_enet = pd.read_csv(filepath, index_col=0)
hgnc_enet = set(bm_enet.Gene)

from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn3

plt.figure()
venn2(
    [hgnc_schizo, silver_schizo],
    set_labels=("CSA", "Silver standard")
)
plt.show()

plt.figure()
venn3(
    [hgnc_schizo, hgnc_ancova, hgnc_enet],
    set_labels=("CSA", "LYRIKS (ANCOVA)", "LYRIKS (elastic net)")
)
plt.show()

##### Plot PCA #####

metadata_csa_plot = metadata.loc[csa_full.columns]
# metadata_csa_plot['collection_days'] = (
#     metadata_csa_plot.collection_datetime - metadata_csa_plot.collection_datetime.min()
# ) / np.timedelta64(1, "D")

fig, ax = plt.subplots(figsize=(10, 8))
ax = bp.plot_pca(
    ax,
    lyriks_full.iloc[:,:20],
    metadata,
    colourbar=True,
    hue='run_datenum',
    style='extraction_date',
    alpha=0.6,
    palette='viridis',
    legend=True
)
plt.show()

filepath = 'tmp/astral/fig/pca_collection_extraction-lyriks.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

metadata.extraction_date.value_counts()
lyriks_b1_b2 = bp.subset(lyriks_full, metadata, "extraction_date != '5/9/24'")
lyriks_b1 = bp.subset(lyriks_full, metadata, "extraction_date == '28/8/24'")
lyriks_b2 = bp.subset(lyriks_full, metadata, "extraction_date == '4/9/24'")
lyriks_b3 = bp.subset(lyriks_full, metadata, "extraction_date == '5/9/24'")

sns.histplot(metadata[metadata.extraction_date == '4/9/24'].run_datetime)
filepath = 'tmp/astral/fig/hist_b2-runtime.pdf'
plt.xticks(rotation=45)
plt.savefig(filepath, dpi=300, bbox_inches='tight')

sid_b2a = metadata.index[
    (metadata.extraction_date == '4/9/24') &
    (metadata.run_datetime < pd.to_datetime('2024-09-20 12:00:00'))
]
lyriks_b2a = lyriks_full[sid_b2a]

sid_b2b = metadata.index[
    (metadata.extraction_date == '4/9/24') &
    (metadata.group == 'Healthy control') &
    (metadata.run_datetime < pd.to_datetime('2024-09-20 12:00:00'))
]
lyriks_b2b = lyriks_full[sid_b2b]

sid_b3a = metadata.index[
    (metadata.extraction_date == '5/9/24') &
    (metadata.group == 'Healthy control') &
    (metadata.study == 'LYRIKS')
]
lyriks_b3a = lyriks_full[sid_b3a]

fig, ax = plt.subplots(figsize=(10, 8))
ax = bp.plot_umap(
    ax,
    lyriks_b1_b2,
    metadata,
    colourbar=True,
    hue='run_datenum',
    style='extraction_date',
    alpha=0.6,
    palette='viridis',
    legend=True
)
filepath = 'tmp/astral/fig/pca_runtime_extraction-lyriks_b1_b2.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
ax = bp.plot_umap(
    ax,
    lyriks_b1,
    metadata,
    colourbar=True,
    hue='run_datenum',
    style='group',
    alpha=0.6,
    palette='viridis',
    legend=True
)
filepath = 'tmp/astral/fig/pca_runtime_group-lyriks_b1.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(10, 8))
ax = bp.plot_umap(
    ax,
    lyriks_b3a,
    metadata,
    colourbar=True,
    hue='collection_datenum',
    style='group',
    alpha=0.6,
    palette='viridis',
    legend=True
)
filepath = 'tmp/astral/fig/pca_collection_group-lyriks_b3a.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

metadata_b2 = metadata.query("extraction_date == '4/9/24'").copy()

metadata_b2['collection_days'] = (
    metadata_b2.collection_datetime - metadata_b2.collection_datetime.min()
) / np.timedelta64(1, "D")
metadata_b2 = metadata_b2.loc[~metadata_b2.collection_days.isna()]
sids = lyriks_b2.columns.intersection(metadata_b2.index)

metadata_b2['run_days'] = (
    metadata_b2.run_datetime - metadata_b2.run_datetime.min()
) / np.timedelta64(1, "D")
sids = lyriks_b2.columns

rows = []
for prot in lyriks_b2.index:
    corr = spearmanr(
        lyriks_b2.loc[prot, sids],
        metadata_b2.loc[sids, 'collection_days']
    )
    row = {
        'protein': prot,
        'spearman_r': corr.correlation,
        'p_value': corr.pvalue
    }
    rows.append(row)

corr_stats_b2 = pd.DataFrame(rows).set_index('protein')
corr_stats_b2.sort_values('spearman_r', ascending=False, key=abs, inplace=True)
prots_highcorr = corr_stats_b2.index[:20]

# Plot expression of whole LYRIKS dataset
for prot in prots_highcorr:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=metadata['collection_datenum'],
        y=lyriks.loc[prot, :],
        hue=metadata.group,
        style=metadata.extraction_date,
        ax=ax
    )
    ax.set_title(f'{prot} (r = {corr_stats_b2.loc[prot, "spearman_r"]:.2f})')
    filepath = f'tmp/astral/fig/{prot}_extraction-collection_expr-corr_clxn.pdf'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(prot)

metadata.columns
metadata.group.value_counts()
metadata_ctrl_mnt = metadata[
    (metadata.group.isin(['Healthy control', 'Maintain'])) &
    (metadata.study == 'LYRIKS')
]

for name, df in metadata_ctrl_mnt.groupby('sn'):
    print(name)
    print(df.group.unique())
    print(df['extraction_date'].tolist())


##### Biomarker identification #####

# ### Prognostic signature ###
# # Without explicit batch correction
# # Using entire dataset
# 
# metadata.columns
# metadata.state.value_counts()
# metadata.group.value_counts()
# 
# # Subset baseline samples from cvt v.s. non-cvt
# # TODO: Should i use only LYRIKS dataset instead (it has more features)? Depends on whether the
# # analysis is to compare against across studies
# baseline_uhr = bp.subset(
#     psyc_knn, metadata,
#     'timepoint == 0 and group in ["cvt", "mnt", "rmt"]'
# )
# covariates = ['group', 'age', 'gender', 'extraction_date']
# data = pd.concat([
#     baseline_uhr.transpose(),
#     metadata.loc[baseline_uhr.columns, covariates],
# ], axis=1)
# data.group.replace({'cvt': 1, 'mnt': 0, 'rmt': 0}, inplace=True)
# data.extraction_date.value_counts()
# # metadata[metadata.extraction_date == '28/8/24'] # only baseline controls in 28/8
# 
# # ANCOVA
# pvalues = []
# coefs = []
# n_cov = len(covariates)
# cov_string = ' + '.join(covariates)
# for prot in data.columns[:-n_cov]:
#     expression = f'{prot} ~ ' + cov_string
#     print(expression)
#     model = ols(expression, data=data).fit()
#     pvalues.append(model.pvalues['group'])
#     coefs.append(model.params['group'])
#     # table = sm.stats.anova_lm(model, typ=2)
# 
# _, qvalues, _, _ = multipletests(
#     pvalues, alpha=0.05, method='fdr_bh'
# )
# stats = pd.DataFrame(
#     {'p': pvalues, 'q': qvalues},
#     index=baseline_uhr.index
# )
# print(stats)
# 
# # prots = statvalues.index[statvalues.p < 0.01] # p-value
# prots = stats.index[stats.q < 0.05] # q-value
# prots.size
# 
# ### Short conversion signature ###
# 
# # Linear mixed models
# # TODO: Exact t2c metadata should be used
# 
# 
# ### Psychosis conversion biomarkers (UHR v.s. Schizo) ###
# 
# # Perfect confounding between state and study
# # Batch correct first before biomarker identification
# metadata.columns
# # metadata.query("study == 'CSA'")
# cond = "extraction_date == '5/9/24' and state == 'Control'"
# ctrl_metadata = metadata.query(cond)
# ctrl_metadata.extraction_date.value_counts()
# print(ctrl_metadata.study.value_counts())
# 
# ctrl = bp.subset(psyc_knn, metadata, cond)
# ctrl.head()
# ctrl.shape
# 
# covariates = ['study', 'age', 'gender']
# data = pd.concat([
#     ctrl.transpose(),
#     metadata.loc[ctrl.columns, covariates],
# ], axis=1)
# data.study.replace({'CSA': 1, 'LYRIKS': 0}, inplace=True)
# data.head()
# # metadata[metadata.extraction_date == '28/8/24'] # only baseline controls in 28/8
# 
# # ANCOVA
# pvalues = []
# coefs = []
# n_cov = len(covariates)
# cov_string = ' + '.join(covariates)
# for prot in data.columns[:-n_cov]:
#     expression = f'{prot} ~ ' + cov_string
#     print(expression)
#     model = ols(expression, data=data).fit()
#     pvalues.append(model.pvalues['study'])
#     coefs.append(model.params['study'])
#     # table = sm.stats.anova_lm(model, typ=2)
# 
# batch_effects = pd.Series(coefs, index=ctrl.index)
# 
# # Correct CSA batch effects
# corr_psyc = psyc_combat.copy()
# corr_psyc.loc[:,corr_psyc.columns.str.startswith('CA')] = corr_psyc.loc[
#     :,corr_psyc.columns.str.startswith('CA')].subtract(batch_effects, axis=0)
# # corr_psyc.equals(psyc_combat)
# 
# ### Investigate CSA ###
# 
# metadata_csa.columns
# 
# # Comorbidities
# # metadata_csa.comorbidities
# comorb_features = [
#     'group', 'comorbidities', 'comorbidities_specify', 'scid_2'
# ]
# metadata_csa[comorb_features].head(30)
# metadata_csa.columns
# metadata_csa.head()
# 
# Investigate collection datetime

sns.histplot(
    metadata_csa.collection_datetime,
    bins=50
)
filepath = 'tmp/astral/fig/hist-collection-datetime.pdf'
plt.savefig(filepath)

import matplotlib.dates as mdates
metadata_csa['collection_datenum'] = mdates.date2num(metadata_csa.collection_datetime)

fig, ax = plt.subplots(figsize=(8, 8))
ax = bp.plot_pca(
    ax,
    csa_full,
    metadata_csa,
    colourbar=True,
    hue='collection_datenum',
    style='group',
    alpha=0.6,
    palette='rocket',
    legend=False
)

# ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
ax2 = bp.plot_pca(
    ax2,
    csa_full,
    metadata_csa,
    hue='group',
    alpha=0.6
)
# ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

filepath = 'tmp/astral/fig/pca-csa.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
    
# ax = bp.plot_umap(
#     csa, metadata_csa,
#     hue='collection_datetime', style='group', alpha=0.6,
#     figsize=(8,5)
# )
# ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
# filepath = 'tmp/astral/fig/umap-csa.pdf'
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# 
# 
# ### Investigate dataset after correcting CSA ###
# 
# metadata.columns
# ax = plot_umap(
#     psyc_combat, metadata,
#     hue='study', style='group', alpha=0.6
# )
# ax.figure.set_size_inches(8, 5)
# filepath = 'tmp/astral/fig/umap-combat-psyc.pdf'
# ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()
# 
# 
# ax = plot_umap(corr_psyc, metadata, hue='study', alpha=0.6, style='group')
# ax.figure.set_size_inches(8, 5)
# filepath = 'tmp/astral/fig/umap-corr-psyc.pdf'
# ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()
# 
# # Perfect confounding between state and medication
# 
# 
# # TODO: Hierarchical clustering plots (CSA) and LYRIKS batch ()

# ### Correct for patient-level effects
# lyriks_fep = bp.subset(
#     lyriks_cvt, metadata,
#     "(timepoint  == 24) & (state == 'FEP')"
# )
# fep_means = lyriks_fep.mean(axis=1)
# corr_deltas = lyriks_fep.sub(fep_means, axis=0)
# corr_deltas.columns = [s[:6] for s in corr_deltas.columns.tolist()]
# print(corr_deltas.columns)

# # Match according to columns of lyriks_cvt (what patient they belong to)
# corr_matched = corr_deltas[lyriks_cvt.columns.str[:6]]
# # Convert column names back otherwise it will cause issues substracting
# corr_matched.columns = lyriks_cvt.columns
# lyriks_corr = lyriks_cvt - corr_matched
# lyriks_corr_int = lyriks_corr.T.join(metadata_fep_delta, how='inner')


### OLINKS ###

csa_full_gene = csa_full.rename(index=map_uniprot_gene)
olink_stats_uniprot = olink_stats.rename(index=map_olink_uniprot)

olink_stats_uniprot.index.isin(csa_full.index).sum() # 0/75 proteins in LINKS are in csa (249)
olink_stats_uniprot.index.isin(csa.index).sum() # 1/75 proteins in LINKS are in csa (1757)
csa_full.index.sort_values().tolist()
olink_stats_uniprot.index.sort_values().tolist(

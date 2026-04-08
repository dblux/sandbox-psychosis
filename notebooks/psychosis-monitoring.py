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

filepath = 'data/astral/processed/reprocessed-data-renamed.csv'
data = pd.read_csv(filepath, index_col=0)
data.replace(0, np.nan, inplace=True)
psy = np.log2(data.iloc[:, 2:-46])
# LYRIKS
lyriks = psy.iloc[:, psy.columns.str.startswith('L')].copy()
lyriks_full = lyriks.dropna()

map_uniprot_gene = {k: v for k, v in zip(data.index, data.Gene)}


##### Biomarker identification #####

### Conversion signature ###

# Detailed months to conversion
filepath = 'data/tmp/cvt-fep_delta.csv'
fep_delta = pd.read_csv(filepath, index_col=0)

metadata_fep_delta = metadata.join(
    fep_delta[['month_of_conversion', 'fep_delta']],
    how='left'
)
metadata_fep_delta['month_of_conversion'] = (
    metadata_fep_delta
        .month_of_conversion
        .astype('Int64')
)

# filepath = 'data/tmp/metadata-fep_delta.csv'
# metadata_fep_delta.to_csv(filepath, index=True)

lyriks_int = lyriks_full.T.join(metadata_fep_delta, how='inner')

# Identify the patient IDs of all timepoints from outliers
outliers = lyriks_int.index[
    (lyriks_int.extraction_date == '4/9/24') &
    (lyriks_int.run_datetime > pd.to_datetime('2024-09-20 12:00:00'))
]
outliers_sn = outliers.str.split('_').str[0] # ['L0626C', 'L0018C']

### Subset data ###

# Raw data
ctrl_raw = bp.subset(
    lyriks_full, metadata,
    "group == 'Healthy control'"
)
ctrl_raw_int = ctrl_raw.T.join(metadata_fep_delta, how='inner')

### Corrected data ###

# ComBat corrected
filepath = 'data/tmp/server/cmc-combat_0409.csv'
cmc_combat_0409 = pd.read_csv(filepath, index_col=0)

# # Limma corrected
# filepath = 'data/tmp/server/cmc-limma_0409.csv'
# cmc_limma_0409 = pd.read_csv(filepath, index_col=0)

cvt = bp.subset(
    cmc_combat_0409,
    metadata,
    "(group == 'Convert') & (sn != 'L0073S')"
)
cvt_int = cvt.T.join(metadata_fep_delta, how='inner')
mnt = bp.subset(
    cmc_combat_0409, metadata,
    "group == 'Maintain'"
)
mnt_int = mnt.T.join(metadata_fep_delta, how='inner')
ctrl = bp.subset(
    cmc_combat_0409, metadata,
    "group == 'Healthy control'"
)
ctrl_int = ctrl.T.join(metadata_fep_delta, how='inner')


### Plot trajectories ###

map_batch_color = {
    '28/8/24': 'tab:blue',
    '4/9/24': 'tab:orange',
    '5/9/24': 'tab:green',
}
map_timepoint_marker = {
    0: 'o',
    12: 's',
    24: '^'
}

### Plot: Corrected data ###

rhos = []
for prot in cmc_combat_0409.index:
    symbol = map_uniprot_gene[prot]
    rho = spearmanr(
        cvt_int.fep_delta,
        cvt_int.loc[:, prot]
    ).correlation
    rhos.append(rho)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(
    #     cvt_int.fep_delta,
    #     cvt_int.loc[:, prot],
    #     c=cvt_int.extraction_date.map(map_batch_color)
    # )
    # ax.set_title(f'{symbol} (r = {rho:.2f}) - Convert')
    # ax.set_xlabel("Months to conversion")
    # ax.set_ylabel(symbol)
    # for _, patient in cvt_int.sort_values("timepoint").groupby("sn"):
    #     ax.plot(
    #         patient["fep_delta"],
    #         patient.loc[:, prot],
    #         color="gray", alpha=0.4, linewidth=1
    #     )
    # dirpath = 'tmp/astral/fig/trajectory/cvt/combat_0409/'
    # filepath = os.path.join(
    #     dirpath,
    #     f'm2c-cvt-{int(abs(rho * 100)):02}-{prot}.pdf'
    # )
    # plt.savefig(filepath, dpi=300, bbox_inches='tight')
    # print(filepath)

# Maintain
for prot in cmc.index:
    rho = spearmanr(
        mnt_int.timepoint,
        mnt_int.loc[:, prot]
    ).correlation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    ax1.scatter(
        mnt_int.timepoint,
        mnt_int.loc[:, prot],
        c=mnt_int.extraction_date.map(map_batch_color)
    )
    ax1.set_title(f'{prot} (r = {rho:.2f}) - Maintain')
    ax1.set_xlabel("Timepoint")
    ax1.set_ylabel(prot)
    for _, patient in mnt_int.sort_values("timepoint").groupby("sn"):
        ax1.plot(
            patient["timepoint"],
            patient.loc[:, prot],
            color="gray", alpha=0.4, linewidth=1
        )
    handles = []
    for g, subdf in mnt_int.groupby("timepoint"):
        h = ax2.scatter(
            subdf["run_datenum"],
            subdf.loc[:, prot],
            c=subdf.extraction_date.map(map_batch_color),
            marker=map_timepoint_marker[g],
            label=str(g)
        )
        handles.append(h)
    ax2.set_xlabel("Run time")
    ax2.legend(handles=handles, loc="best")
    handles = []
    for g, subdf in mnt_int.groupby("timepoint"):
        h = ax3.scatter(
            subdf["collection_datenum"],
            subdf.loc[:, prot],
            c=subdf.extraction_date.map(map_batch_color),
            marker=map_timepoint_marker[g],
            label=str(g)
        )
        handles.append(h)
    ax3.set_xlabel("Collection time")
    ax3.legend(handles=handles, loc="best")
    dirpath = 'tmp/astral/fig/trajectory/mnt/limma_0409/'
    filepath = os.path.join(
        dirpath,
        f'm2c_mnt-{int(abs(rho * 100)):02}-{prot}.pdf'
    )
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(filepath)

# Control
for prot in cmc.index:
    rho = spearmanr(
        ctrl_int.timepoint,
        ctrl_int.loc[:, prot]
    ).correlation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    ax1.scatter(
        ctrl_int.timepoint,
        ctrl_int.loc[:, prot],
        c=ctrl_int.extraction_date.map(map_batch_color)
    )
    ax1.set_title(f'{prot} (r = {rho:.2f}) - Control')
    ax1.set_xlabel("Timepoint")
    ax1.set_ylabel(prot)
    for _, patient in ctrl_int.sort_values("timepoint").groupby("sn"):
        ax1.plot(
            patient["timepoint"],
            patient.loc[:, prot],
            color="gray", alpha=0.4, linewidth=1
        )
    handles = []
    for g, subdf in ctrl_int.groupby("timepoint"):
        h = ax2.scatter(
            subdf["run_datenum"],
            subdf.loc[:, prot],
            c=subdf.extraction_date.map(map_batch_color),
            marker=map_timepoint_marker[g],
            label=str(g)
        )
        handles.append(h)
    ax2.set_xlabel("Run time")
    ax2.legend(handles=handles, loc="best")
    handles = []
    for g, subdf in ctrl_int.groupby("timepoint"):
        h = ax3.scatter(
            subdf["collection_datenum"],
            subdf.loc[:, prot],
            c=subdf.extraction_date.map(map_batch_color),
            marker=map_timepoint_marker[g],
            label=str(g)
        )
        handles.append(h)
    ax3.set_xlabel("Collection time")
    ax3.legend(handles=handles, loc="best")
    dirpath = 'tmp/astral/fig/trajectory/ctrl/limma_0409/'
    filepath = os.path.join(
        dirpath,
        f'm2c_ctrl-{int(abs(rho * 100)):02}-{prot}.pdf'
    )
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(filepath)


### Plot ###

# Feature selection
# spearman_cvt = pd.DataFrame({
#     'symbol': cmc_combat_0409.index.map(map_uniprot_gene),
#     'spearman_r': rhos,
# }, index=cmc_combat_0409.index)
# spearman_cvt.sort_values('spearman_r', ascending=False, key=abs, inplace=True)
# spearman_cvt.to_csv('tmp/astral/cvt-spearman.csv', index=True)

file = 'tmp/astral/cvt-spearman.csv'
spearman_cvt = pd.read_csv(file, index_col=0)
prots_spearman = spearman_cvt.index[abs(spearman_cvt.spearman_r) > 0.4]
prots_spearman.shape

# TODO: Plot figure with three subfigures cvt, mnt, ctrl

# TODO: Plot PCA of trajectories (cvt only), color by fep_delta

def plot_pca_trajectory(
    x, metadata, filepath
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax, pca_cvt = bp.plot_pca(
        ax,
        x.loc[prots_spearman],
        metadata.loc[x.columns],
        colourbar=True,
        return_fig=False,
        hue='fep_delta',
        palette='rocket'
    )
    for _, patient in pca_cvt.sort_values("timepoint").groupby("sn"):
        ax.plot(
            patient["PC1"],
            patient["PC2"],
            color="gray", alpha=0.4, linewidth=1
        )
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

plot_pca_trajectory(
    cmc_combat_0409.loc[prots_spearman],
    metadata_fep_delta,
    'tmp/astral/fig/trajectory/pca-spearman10-cmc_combat_0409.pdf'
)


def plot_umap_trajectory(
    x, metadata, filepath
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax, umap_cvt = bp.plot_umap(
        ax,
        x,
        metadata.loc[x.columns],
        colourbar=True,
        return_fig=False,
        hue='fep_delta',
        palette='rocket'
    )
    for _, patient in umap_cvt.sort_values("timepoint").groupby("sn"):
        ax.plot(
            patient["UMAP1"],
            patient["UMAP2"],
            color="gray", alpha=0.4, linewidth=1
        )
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

plot_umap_trajectory(
    cmc_combat_0409.loc[prots_spearman],
    metadata_fep_delta,
    'tmp/astral/fig/trajectory/umap-spearman10-cmc_combat_0409.pdf'
)


# TODO: Plot PCA of trajectories (cvt, ctrl, mnt only)

### Slopes ###

# ### Aggregate trajectories ###
# 
# # investigate: timepoint v.s. run time
# metadata.columns
# md_ctrl = metadata.loc[
#     (metadata.group == 'Healthy control') &
#     (metadata.study == 'LYRIKS'),
#     ['sn', 'group', 'timepoint', 'run_datenum']
# ]
# 
# for tp, subdf in md_ctrl.groupby("timepoint"):
#     plt.hist(subdf["run_datenum"], bins=20, alpha=0.5, label=str(tp))
# 
# plt.xlabel("run_datenum")
# plt.ylabel("count")
# plt.legend(title="timepoint")
# plt.show()
# 
# # Aggregate top upregulated and downupregulated proteins
# spearman_cvt = pd.DataFrame({
#     'uniprot': cmc.index,
#     'symbol': cmc.index.map(map_uniprot_gene),
#     'spearman_r': rhos,
# }).sort_values(
#     'spearman_r', ascending=False, key=abs
# )
# 
# filepath = 'tmp/astral/cvt-spearman.csv'
# spearman_cvt.to_csv(filepath, index=False)
# 
# spearman_top = spearman_cvt.head(23)
# print(spearman_top)
# top_up = spearman_top.uniprot[spearman_top.spearman_r > 0]
# top_down = spearman_top.uniprot[spearman_top.spearman_r <= 0]
# 
# agg = metadata_fep_delta.loc[
#     cvt.columns,
#     ['sn', 'timepoint', 'fep_delta']
# ].copy()
# agg['cvt_up'] = cvt.loc[top_up].sum(axis=0)
# agg['cvt_down'] = cvt.loc[top_down].sum(axis=0)
# agg_mnt = pd.DataFrame({
#     'mnt_up': mnt.loc[top_up].sum(axis=0),
#     'mnt_down': mnt.loc[top_down].sum(axis=0)
# }).join(
#     metadata_fep_delta[['sn', 'timepoint', 'extraction_date']],
#     how='left'
# )
# agg_ctrl = pd.DataFrame({
#     'ctrl_up': ctrl.loc[top_up].sum(axis=0),
#     'ctrl_down': ctrl.loc[top_down].sum(axis=0)
# }).join(
#     metadata_fep_delta[['sn', 'timepoint', 'extraction_date']],
#     how='left'
# )
# 
# # Plot convert patients
# rho = spearmanr(agg.fep_delta, agg.cvt_up).correlation
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(
#     agg.fep_delta,
#     agg.cvt_up,
#     c='tab:green'
# )
# ax.set_title(f'Top 3 up-regulated (r = {rho:.2f})')
# ax.set_xlabel("Months to conversion")
# for _, patient in agg.sort_values("timepoint").groupby("sn"):
#     ax.plot(
#         patient.fep_delta,
#         patient.cvt_up,
#         color="gray", alpha=0.4, linewidth=1
#     )
# 
# dirpath = 'tmp/astral/fig/trajectory/'
# filepath = os.path.join(
#     dirpath,
#     f'm2c_cvt-{int(abs(rho * 100)):02}-agg_up.pdf'
# )
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# print(filepath)
# 
# # Plot maintain patients
# rho = spearmanr(agg_mnt.timepoint, agg_mnt.mnt_down).correlation
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(
#     agg_mnt.timepoint,
#     agg_mnt.mnt_down,
#     c=agg_mnt.extraction_date.map(map_batch_color)
# )
# ax.set_title(f'Top 20 down-regulated (r = {rho:.2f})')
# ax.set_xlabel('Months to conversion')
# for _, patient in agg_mnt.sort_values('timepoint').groupby('sn'):
#     ax.plot(
#         patient.timepoint,
#         patient.mnt_down,
#         color='gray', alpha=0.4, linewidth=1
#     )
# 
# dirpath = 'tmp/astral/fig/trajectory/'
# filepath = os.path.join(
#     dirpath,
#     f'm2c_mnt-{int(abs(rho * 100)):02}-agg_down.pdf'
# )
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# print(filepath)


### TODO: Paired t-test (against zero)

metadata_convert = metadata_fep_delta.query(
    "group == 'Convert' and sn != 'L0073S'"
)

before_sids = []
fep_sids = []
for _, grp in metadata_convert.groupby('sn'):
    before_sids.append(grp.index[-2])
    fep_sids.append(grp.index[-1])

lyriks_full[before_sids].shape
ttest = ttest_rel(
    lyriks_full[before_sids],
    lyriks_full[fep_sids],
    axis=1
)

conversion_stats = pd.DataFrame({
    'gene': lyriks_full.index.map(map_uniprot_gene),
    'pvalue': ttest.pvalue,
    'qvalue': multipletests(ttest.pvalue, alpha=0.05, method='fdr_bh')[1],
    't': ttest.statistic
}, index=lyriks_full.index)
conversion_stats.sort_values('pvalue', inplace=True)

filepath = 'tmp/astral/conversion-pairedttest.csv'
conversion_stats.to_csv(filepath, index=True)

# Exclude L0073S (not FEP)
# All other 12 converts: FEP and the nearest timepoint for paired t-test


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

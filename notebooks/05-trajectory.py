import os
import pickle
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


filepath = 'data/metadata/metadata-psy_602_16-v1.csv'
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

filepath = 'data/processed/reprocessed-data-renamed.csv'
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

# # Identify the patient IDs of all timepoints from outliers
# outliers = lyriks_int.index[
#     (lyriks_int.extraction_date == '4/9/24') &
#     (lyriks_int.run_datetime > pd.to_datetime('2024-09-20 12:00:00'))
# ]
# outliers_sn = outliers.str.split('_').str[0] # ['L0626C', 'L0018C']

### Subset data ###

# # Raw data
# ctrl_raw = bp.subset(
#     lyriks_full, metadata,
#     "group == 'Healthy control'"
# )
# ctrl_raw_int = ctrl_raw.T.join(metadata_fep_delta, how='inner')

### Corrected data ###

# ComBat corrected
filepath = 'data/tmp/corrected-twostep/lyriks_cmc-combat_0409.csv'
cmc_combat_0409 = pd.read_csv(filepath, index_col=0)

# # Limma corrected
# filepath = 'data/tmp/server/cmc-limma_0409.csv'
# cmc_limma_0409 = pd.read_csv(filepath, index_col=0)

# # L0073S_24 is not FEP but UHR
# metadata_fep_delta.loc[
#     metadata_fep_delta.sn == 'L0073S',
#     ['group', 'state', 'fep_delta']
# ]

cmc_int = cmc_combat_0409.T.join(metadata_fep_delta, how='inner')

cvt = bp.subset(
    cmc_combat_0409,
    metadata,
    "(group == 'Convert')" # & (sn != 'L0073S')"
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

batch_colours = {
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

# rhos = []
# for prot in cmc_combat_0409.index:
#     symbol = map_uniprot_gene[prot]
#     rho = spearmanr(
#         cvt_int.fep_delta,
#         cvt_int.loc[:, prot]
#     ).correlation
#     rhos.append(rho)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.scatter(
#         cvt_int.fep_delta,
#         cvt_int.loc[:, prot],
#         c=cvt_int.extraction_date.map(batch_colours)
#     )
#     ax.set_title(f'{symbol} (r = {rho:.2f}) - Convert')
#     ax.set_xlabel("Months to conversion")
#     ax.set_ylabel(symbol)
#     for _, patient in cvt_int.sort_values("timepoint").groupby("sn"):
#         ax.plot(
#             patient["fep_delta"],
#             patient.loc[:, prot],
#             color="gray", alpha=0.4, linewidth=1
#         )
#     dirpath = 'tmp/astral/fig/trajectory/cvt/combat_0409/'
#     filepath = os.path.join(
#         dirpath,
#         f'm2c-cvt-{int(abs(rho * 100)):02}-{prot}.pdf'
#     )
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)
# 
# # Maintain
# for prot in cmc.index:
#     rho = spearmanr(
#         mnt_int.timepoint,
#         mnt_int.loc[:, prot]
#     ).correlation
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
#     ax1.scatter(
#         mnt_int.timepoint,
#         mnt_int.loc[:, prot],
#         c=mnt_int.extraction_date.map(batch_colours)
#     )
#     ax1.set_title(f'{prot} (r = {rho:.2f}) - Maintain')
#     ax1.set_xlabel("Timepoint")
#     ax1.set_ylabel(prot)
#     for _, patient in mnt_int.sort_values("timepoint").groupby("sn"):
#         ax1.plot(
#             patient["timepoint"],
#             patient.loc[:, prot],
#             color="gray", alpha=0.4, linewidth=1
#         )
#     handles = []
#     for g, subdf in mnt_int.groupby("timepoint"):
#         h = ax2.scatter(
#             subdf["run_datenum"],
#             subdf.loc[:, prot],
#             c=subdf.extraction_date.map(batch_colours),
#             marker=map_timepoint_marker[g],
#             label=str(g)
#         )
#         handles.append(h)
#     ax2.set_xlabel("Run time")
#     ax2.legend(handles=handles, loc="best")
#     handles = []
#     for g, subdf in mnt_int.groupby("timepoint"):
#         h = ax3.scatter(
#             subdf["collection_datenum"],
#             subdf.loc[:, prot],
#             c=subdf.extraction_date.map(batch_colours),
#             marker=map_timepoint_marker[g],
#             label=str(g)
#         )
#         handles.append(h)
#     ax3.set_xlabel("Collection time")
#     ax3.legend(handles=handles, loc="best")
#     dirpath = 'tmp/astral/fig/trajectory/mnt/limma_0409/'
#     filepath = os.path.join(
#         dirpath,
#         f'm2c_mnt-{int(abs(rho * 100)):02}-{prot}.pdf'
#     )
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)
# 
# # Control
# for prot in cmc.index:
#     rho = spearmanr(
#         ctrl_int.timepoint,
#         ctrl_int.loc[:, prot]
#     ).correlation
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
#     ax1.scatter(
#         ctrl_int.timepoint,
#         ctrl_int.loc[:, prot],
#         c=ctrl_int.extraction_date.map(batch_colours)
#     )
#     ax1.set_title(f'{prot} (r = {rho:.2f}) - Control')
#     ax1.set_xlabel("Timepoint")
#     ax1.set_ylabel(prot)
#     for _, patient in ctrl_int.sort_values("timepoint").groupby("sn"):
#         ax1.plot(
#             patient["timepoint"],
#             patient.loc[:, prot],
#             color="gray", alpha=0.4, linewidth=1
#         )
#     handles = []
#     for g, subdf in ctrl_int.groupby("timepoint"):
#         h = ax2.scatter(
#             subdf["run_datenum"],
#             subdf.loc[:, prot],
#             c=subdf.extraction_date.map(batch_colours),
#             marker=map_timepoint_marker[g],
#             label=str(g)
#         )
#         handles.append(h)
#     ax2.set_xlabel("Run time")
#     ax2.legend(handles=handles, loc="best")
#     handles = []
#     for g, subdf in ctrl_int.groupby("timepoint"):
#         h = ax3.scatter(
#             subdf["collection_datenum"],
#             subdf.loc[:, prot],
#             c=subdf.extraction_date.map(batch_colours),
#             marker=map_timepoint_marker[g],
#             label=str(g)
#         )
#         handles.append(h)
#     ax3.set_xlabel("Collection time")
#     ax3.legend(handles=handles, loc="best")
#     dirpath = 'tmp/astral/fig/trajectory/ctrl/limma_0409/'
#     filepath = os.path.join(
#         dirpath,
#         f'm2c_ctrl-{int(abs(rho * 100)):02}-{prot}.pdf'
#     )
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)


### Plot ###

# %% Feature selection (spearman's rho)
# rhos = []
# for prot in cmc_combat_0409.index:
#     symbol = map_uniprot_gene[prot]
#     rho = spearmanr(
#         cvt_int.fep_delta,
#         cvt_int.loc[:, prot]
#     ).correlation
#     rhos.append(rho)
# 
# spearman_cvt = pd.DataFrame({
#     'symbol': cmc_combat_0409.index.map(map_uniprot_gene),
#     'spearman_r': rhos,
# }, index=cmc_combat_0409.index)
# spearman_cvt.sort_values('spearman_r', ascending=False, key=abs, inplace=True)
# prots_spearman = spearman_cvt.index[abs(spearman_cvt.spearman_r) > 0.4]
# 
# filepath = 'outputs/tmp/cvt-spearman.csv'
# spearman_cvt.to_csv(filepath, index=True)
# print(filepath)

# # Load proteins (spearman's)
# file = 'outputs/tmp/cvt-spearman.csv'
# spearman_cvt = pd.read_csv(file, index_col=0)
# prots_spearman = spearman_cvt.index[abs(spearman_cvt.spearman_r) > 0.4]
# prots_spearman.shape

### Plot trajectories of cvt, mnt, ctrl patients ###

# x: sn, timepoint, extraction_date, fep_delta, cca1, group

def plot_trajectory(x, feature, ylabel, batch_colours):
    """Plot 3-panel trajectory figure for a single feature.

    Parameters
    ----------
    x : DataFrame
        Combined data for all groups. Must have columns: group, fep_delta,
        timepoint, extraction_date, sn, and `feature`.
    feature : str
        Column name of the feature to plot (e.g. UniProt ID or CCA score).
    ylabel : str
        Label for the y-axis (e.g. gene symbol).
    batch_colours : dict
        Mapping from extraction_date value to colour.
    """
    cvt = x[x.group == 'Convert']
    print(cvt.shape)
    mnt = x[x.group == 'Maintain']
    ctrl = x[x.group == 'Healthy control']
    rho_cvt = spearmanr(cvt.fep_delta, cvt.loc[:, feature]).correlation
    rho_mnt = spearmanr(mnt.timepoint, mnt.loc[:, feature]).correlation
    rho_ctrl = spearmanr(ctrl.timepoint, ctrl.loc[:, feature]).correlation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    # Convert
    ax1.scatter(
        cvt.fep_delta,
        cvt.loc[:, feature],
        c=cvt.extraction_date.map(batch_colours)
    )
    ax1.set_title(f'Convert (r = {rho_cvt:.2f})')
    ax1.set_xlabel("Months to conversion")
    ax1.set_ylabel(ylabel)
    for _, patient in cvt.sort_values("timepoint").groupby("sn"):
        ax1.plot(
            patient["fep_delta"],
            patient.loc[:, feature],
            color="gray", alpha=0.4, linewidth=1
        )
    # Maintain
    ax2.scatter(
        mnt.timepoint,
        mnt.loc[:, feature],
        c=mnt.extraction_date.map(batch_colours)
    )
    ax2.set_title(f'Maintain (r = {rho_mnt:.2f})')
    ax2.set_xlabel("Timepoint")
    for _, patient in mnt.sort_values("timepoint").groupby("sn"):
        ax2.plot(
            patient["timepoint"],
            patient.loc[:, feature],
            color="gray", alpha=0.4, linewidth=1
        )
    # Control
    ax3.scatter(
        ctrl.timepoint,
        ctrl.loc[:, feature],
        c=ctrl.extraction_date.map(batch_colours)
    )
    ax3.set_title(f'Control (r = {rho_ctrl:.2f})')
    ax3.set_xlabel("Timepoint")
    for _, patient in ctrl.sort_values("timepoint").groupby("sn"):
        ax3.plot(
            patient["timepoint"],
            patient.loc[:, feature],
            color="gray", alpha=0.4, linewidth=1
        )
    return fig


# %% Plot top 10 proteins
for prot in prots_spearman:
    fig = plot_trajectory(
        cmc_int,
        prot,
        map_uniprot_gene[prot],
        batch_colours
    )
    rho_cvt = spearmanr(cvt_int.fep_delta, cvt_int.loc[:, prot]).correlation
    filepath = os.path.join(
        'outputs/figs/trajectory/corr2-combat_0409/',
        f'trajectory-{int(abs(rho_cvt * 100)):02}-{prot}.pdf'
    )
    print(filepath)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


# ### Plot CCA trajectory ###
# 
# cca_scores = pd.read_csv('data/tmp/zhihao/cca_m2c-scores.csv')
# cca_scores = cca_scores[[
#     'sn', 'timepoint', 'group', 'extraction_date', 'fep_delta', 'cca1_score']]
# 
# fig = plot_trajectory(cca_scores, 'cca1_score', 'CCA1', batch_colours)
# filepath = 'outputs/figs/trajectory/trajectory-cca1.pdf'
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()


### Plot trajectory in PCA and UMAP space ###

# def plot_pca_trajectory(x, metadata, filepath):
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax, pca_cvt = bp.plot_pca(
#         ax,
#         x.loc[prots_spearman],
#         metadata.loc[x.columns],
#         colourbar=True,
#         return_fig=False,
#         hue='fep_delta',
#         hue_label='Months to conversion',
#         palette='rocket'
#     )
#     for _, patient in pca_cvt.sort_values("timepoint").groupby("sn"):
#         ax.plot(
#             patient["PC1"],
#             patient["PC2"],
#             color="gray", alpha=0.4, linewidth=1
#         )
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
# 
# plot_pca_trajectory(
#     cmc_combat_0409.loc[prots_spearman],
#     metadata_fep_delta,
#     'outputs/figs/trajectory/pca-spearman10-cmc_combat_0409.pdf'
# )
# 
# 
# def plot_umap_trajectory(x, metadata, filepath):
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax, umap_cvt = bp.plot_umap(
#         ax,
#         x,
#         metadata.loc[x.columns],
#         colourbar=True,
#         return_fig=False,
#         hue='fep_delta',
#         hue_label='Months to conversion',
#         palette='rocket'
#     )
#     for _, patient in umap_cvt.sort_values("timepoint").groupby("sn"):
#         ax.plot(
#             patient["UMAP1"],
#             patient["UMAP2"],
#             color="gray", alpha=0.4, linewidth=1
#         )
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
# 
# plot_umap_trajectory(
#     cmc_combat_0409.loc[prots_spearman],
#     metadata_fep_delta,
#     'outputs/figs/trajectory/umap-spearman10-cmc_combat_0409.pdf'
# )


### Slopes ###

# TODO: Slope features
filepath = 'data/tmp/zhihao/slope_features.csv'
slope_features = pd.read_csv(filepath, index_col=0)
slope_features.head()
slope_features.columns.tolist()

prots_slope = ['Q7Z7G0', 'P43652', 'P02760', 'Q15113', 'P02786']
slope_features1 = slope_features.loc[
    :, slope_features.columns.str.startswith(tuple(prots_slope))]

slope_features.loc[
    slope_features.group == 'Convert',
    ['Q7Z7G0__velocity']
]

tmp = cmc_int.loc[
    cmc_int.group == 'Convert',
    ['Q7Z7G0', 'fep_delta']
]
tmp
(13.93 - 13.608) / (0.72+9) / 31

# TODO: Check whether features were calculated correctly
# TODO: Compute ANOVA p-values

# TODO: Check pickle object
filepath = 'data/tmp/zhihao/cca.pkl'
with open(filepath, 'rb') as f:
    cca = pickle.load(f)

type(cca)
dir(cca)

model = cca['cca_model']
weights = model.x_weights_
weights = pd.DataFrame({
    'symbol': cmc_combat_0409.index.map(map_uniprot_gene),
    'coef': weights[:, 0]
}, index=cmc_combat_0409.index)
weights.sort_values('coef', ascending=False, key=abs, inplace=True)
filepath = 'outputs/tmp/cca_weights.csv'
weights.to_csv(filepath)


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
#     c=agg_mnt.extraction_date.map(batch_colours)
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

# metadata_convert = metadata_fep_delta.query(
#     "group == 'Convert' and sn != 'L0073S'"
# )
# 
# before_sids = []
# fep_sids = []
# for _, grp in metadata_convert.groupby('sn'):
#     before_sids.append(grp.index[-2])
#     fep_sids.append(grp.index[-1])
# 
# lyriks_full[before_sids].shape
# ttest = ttest_rel(
#     lyriks_full[before_sids],
#     lyriks_full[fep_sids],
#     axis=1
# )
# 
# conversion_stats = pd.DataFrame({
#     'gene': lyriks_full.index.map(map_uniprot_gene),
#     'pvalue': ttest.pvalue,
#     'qvalue': multipletests(ttest.pvalue, alpha=0.05, method='fdr_bh')[1],
#     't': ttest.statistic
# }, index=lyriks_full.index)
# conversion_stats.sort_values('pvalue', inplace=True)
# 
# filepath = 'tmp/astral/conversion-pairedttest.csv'
# conversion_stats.to_csv(filepath, index=True)

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

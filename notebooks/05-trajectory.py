import os
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
from scipy.stats import spearmanr, ttest_rel, kendalltau, f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import biopy.utils as bp


filepath = 'data/metadata/metadata-psy_602_16-v2.csv'
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
map_uniprot_gene = {k: v for k, v in zip(data.index, data.Gene)}
psy = np.log2(data.iloc[:, 2:-46])
# LYRIKS
lyriks = psy.iloc[:, psy.columns.str.startswith('L')].copy()
lyriks_full = lyriks.dropna()


##### Biomarker identification #####

### Conversion signature ###

# Detailed months to conversion
filepath = 'data/tmp/cvt-fep_delta.csv'
fep_delta = pd.read_csv(filepath, index_col=0)

metadata_month = metadata.join(
    fep_delta[['month_of_conversion', 'fep_delta']],
    how='left'
)
metadata_month.rename(
    columns={'fep_delta': 'month'}, inplace=True
)
metadata_month['month_of_conversion'] = (
    metadata_month
        .month_of_conversion
        .astype('Int64')
)

# Impute missing month for LYRIKS patients
metadata_month.loc[
    (metadata_month.month.isna()) & (metadata_month.study == 'LYRIKS'),
    'month'
] = metadata_month.loc[
    (metadata_month.month.isna()) & (metadata_month.study == 'LYRIKS'),
    'timepoint'
]

# filepath = 'data/tmp/metadata-fep_delta.csv'
# metadata_month.to_csv(filepath, index=True)

lyriks_meta = lyriks_full.T.join(metadata_month, how='inner')

# # Identify the patient IDs of all timepoints from outliers
# outliers = lyriks_meta.index[
#     (lyriks_meta.extraction_date == '4/9/24') &
#     (lyriks_meta.run_datetime > pd.to_datetime('2024-09-20 12:00:00'))
# ]
# outliers_sn = outliers.str.split('_').str[0] # ['L0626C', 'L0018C']

### Subset data ###

# # Raw data
# ctrl_raw = bp.subset(
#     lyriks_full, metadata,
#     "group == 'Healthy control'"
# )
# ctrl_raw_int = ctrl_raw.T.join(metadata_month, how='inner')


### Plot: Raw data ###

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
map_group_marker = {
    'Control': 'o',
    'Early remit': 's',
    'Maintain': '^',
    'Convert': 'D',
    'Late remit': 'P',
    'Relapse': 'X'
}


def plot_batch_effects_2d(df, prot, batch_colours):
    ylabel = f"{map_uniprot_gene[prot]} ({prot})"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sns.scatterplot(
        data=df, x="collection_datenum", y=prot,
        hue="extraction_date", style="group",
        palette=batch_colours, alpha=0.7, edgecolor='none', legend=False,
        ax=ax1
    )
    sns.scatterplot(
        data=df, x="run_datenum", y=prot,
        hue="extraction_date", style="group",
        palette=batch_colours, alpha=0.7, edgecolor='none', ax=ax2
    )
    ax1.set_ylabel(ylabel)
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax1.set_xlabel("Collection time")
    ax2.set_xlabel("Run time")
    return fig


def plot_batch_effects_3d(df, prot, batch_colours, group_markers):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    for g, subdf in df.groupby("group"):
        ax.scatter(
            subdf["collection_datenum"],
            subdf["run_datenum"],
            subdf[prot],
            c=subdf.extraction_date.map(batch_colours),
            marker=group_markers[g],
            alpha=0.7,
            depthshade=False,
            label=g
        )
    ax.view_init(elev=15, azim=-30)
    ax.set_xlabel("Collection time")
    ax.set_ylabel("Run time")
    ax.set_zlabel(f"{map_uniprot_gene[prot]} ({prot})")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig


# prots_batch_effects = ['P02647', 'P00747']
# dirpath = 'outputs/figs/trajectory/batch_effects/'
# for prot in prots_batch_effects:
#     fig = plot_batch_effects_2d(lyriks_meta, prot, batch_colours)
#     filepath = os.path.join(dirpath, f'batch-lyriks387-{prot}.pdf')
#     fig.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)
# 
# for prot in prots_batch_effects:
#     fig = plot_batch_effects_3d(
#         lyriks_meta, prot, batch_colours, map_group_marker
#     )
#     filepath = os.path.join(dirpath, f'3D-batch-lyriks387-{prot}.pdf')
#     fig.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)


### Corrected data ###

# L0073S_24 is not FEP but UHR!

# ComBat corrected
filepath = 'data/tmp/corr-2step/lyriks387-combat_0409.csv'
lyriks387_cb0409 = pd.read_csv(filepath, index_col=0)
lyriks387cb_meta = lyriks387_cb0409.T.join(metadata_month, how='inner')

# prots_batch_effects = ['P02647', 'P00747']
# dirpath = 'outputs/figs/trajectory/batch_effects/'
# for prot in prots_batch_effects:
#     fig = plot_batch_effects_2d(lyriks387cb_meta, prot, batch_colours)
#     filepath = os.path.join(dirpath, f'batch-lyriks387-cb0409-{prot}.pdf')
#     fig.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)
# 
# for prot in prots_batch_effects:
#     fig = plot_batch_effects_3d(
#         lyriks387cb_meta, prot, batch_colours, map_group_marker
#     )
#     filepath = os.path.join(
#         dirpath,
#         f'3D-batch-lyriks387-cb0409-{prot}.pdf'
#     )
#     fig.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)


### Plot trajectories ###

# # Plot control samples
# for prot in prots_batch_effects:
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
#     dirpath = 'outputs/figs/trajectory/ctrl/limma_0409/'
#     filepath = os.path.join(
#         dirpath,
#         f'm2c_ctrl-{int(abs(rho * 100)):02}-{prot}.pdf'
#     )
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(filepath)


### Feature selection (spearman's rho)

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

def plot_trajectory(x):
    cvt = x[x.group == 'Convert']
    mnt = x[x.group == 'Maintain']
    ctrl = x[x.group == 'Control']
    early = x[x.group == 'Early remit']
    late = x[x.group == 'Late remit']
    relapse = x[x.group == 'Relapse']
    rows = []
    for uid in x.columns[:-3]:
        print(uid)
        rows.append({
            'uniprot': uid,
            'rho_cvt': spearmanr(cvt.month, cvt.loc[:, uid]).correlation,
            'rho_mnt': spearmanr(mnt.timepoint, mnt.loc[:, uid]).correlation,
            'rho_ctrl': spearmanr(ctrl.timepoint, ctrl.loc[:, uid]).correlation,
            'rho_early': spearmanr(early.timepoint, early.loc[:, uid]).correlation,
            'rho_late': spearmanr(late.timepoint, late.loc[:, uid]).correlation,
            'rho_relapse': spearmanr(relapse.timepoint, relapse.loc[:, uid]).correlation,
        })
    return pd.DataFrame(rows).set_index('uniprot')


lyriks387_cb0409 = pd.read_csv(filepath, index_col=0)
metadata3 = metadata_month[['group', 'month', 'timepoint']]
lyriks387_meta3 = lyriks387_cb0409.T.join(metadata3, how='inner')

# Load proteins (spearman's)
file = 'outputs/tmp/cvt-spearman.csv'
spearman_cvt = pd.read_csv(file, index_col=0)
prots_spearman = spearman_cvt.index[abs(spearman_cvt.spearman_r) > 0.4]
print(prots_spearman.shape)


### Plot ###

### Plot trajectories of patients across different groups ###

def plot_trajectory(x, feature, ylabel, batch_colours):
    """Plot 5-panel trajectory figure for a single feature.

    Parameters
    ----------
    x : DataFrame
        Combined data for all groups. Must have columns: group, month,
        timepoint, extraction_date, sn, and `feature`.
    feature : str
        Column name of the feature to plot (e.g. UniProt ID or CCA score).
    ylabel : str
        Label for the y-axis (e.g. gene symbol).
    batch_colours : dict
        Mapping from extraction_date value to colour.
    """
    cvt = x[x.group == 'Convert']
    mnt = x[x.group == 'Maintain']
    ctrl = x[x.group == 'Control']
    early = x[x.group == 'Early remit']
    late = x[x.group == 'Late remit']
    relapse = x[x.group == 'Relapse']
    rho_cvt = spearmanr(cvt.month, cvt.loc[:, feature]).correlation
    rho_mnt = spearmanr(mnt.timepoint, mnt.loc[:, feature]).correlation
    rho_ctrl = spearmanr(ctrl.timepoint, ctrl.loc[:, feature]).correlation
    rho_early = spearmanr(early.timepoint, early.loc[:, feature]).correlation
    rho_late = spearmanr(late.timepoint, late.loc[:, feature]).correlation
    rho_relapse = spearmanr(relapse.timepoint, relapse.loc[:, feature]).correlation
    ### Plot ###
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    # Convert
    ax1.scatter(
        cvt.month,
        cvt.loc[:, feature],
        c=cvt.extraction_date.map(batch_colours)
    )
    ax1.set_title(f'Convert (r = {rho_cvt:.2f})')
    ax1.set_xlabel("Months to conversion")
    ax1.set_ylabel(ylabel)
    for _, patient in cvt.sort_values("timepoint").groupby("sn"):
        ax1.plot(
            patient["month"],
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
    # Early remit
    ax4.scatter(
        early.timepoint,
        early.loc[:, feature],
        c=early.extraction_date.map(batch_colours)
    )
    ax4.set_title(f'Early remit (r = {rho_early:.2f})')
    ax4.set_xlabel("Timepoint")
    ax4.set_ylabel(ylabel)
    for _, patient in early.sort_values("timepoint").groupby("sn"):
        ax4.plot(
            patient["timepoint"],
            patient.loc[:, feature],
            color="gray", alpha=0.4, linewidth=1
        )
    # Late remit
    ax5.scatter(
        late.timepoint,
        late.loc[:, feature],
        c=late.extraction_date.map(batch_colours)
    )
    ax5.set_title(f'Late remit (r = {rho_late:.2f})')
    ax5.set_xlabel("Timepoint")
    for _, patient in late.sort_values("timepoint").groupby("sn"):
        ax5.plot(
            patient["timepoint"],
            patient.loc[:, feature],
            color="gray", alpha=0.4, linewidth=1
        )
    # Relapse
    ax6.scatter(
        relapse.timepoint,
        relapse.loc[:, feature],
        c=relapse.extraction_date.map(batch_colours)
    )
    ax6.set_title(f'Relapse (r = {rho_relapse:.2f})')
    ax6.set_xlabel("Timepoint")
    for _, patient in relapse.sort_values("timepoint").groupby("sn"):
        ax6.plot(
            patient["timepoint"],
            patient.loc[:, feature],
            color="gray", alpha=0.4, linewidth=1
        )
    return fig


# # %% Plot top 10 proteins
# for prot in prots_spearman:
#     fig = plot_trajectory(
#         lyriks387cb_meta,
#         prot,
#         map_uniprot_gene[prot],
#         batch_colours
#     )
#     rho_cvt = spearmanr(
#         lyriks387cb_meta.month,
#         lyriks387cb_meta.loc[:, prot]
#     ).correlation
#     filepath = os.path.join(
#         'outputs/figs/trajectory/corr2-combat_0409/lyriks387/',
#         f'trajectory-{int(abs(rho_cvt * 100)):02}-{prot}.pdf'
#     )
#     print(filepath)
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     plt.close()


# ### CCA ###
# 
# # ### Plot CCA trajectory ###
# # 
# # cca_scores = pd.read_csv('data/tmp/zhihao/cca_m2c-scores.csv')
# # cca_scores = cca_scores[[
# #     'sn', 'timepoint', 'group', 'extraction_date', 'month', 'cca1_score']]
# # 
# # fig = plot_trajectory(cca_scores, 'cca1_score', 'CCA1', batch_colours)
# # filepath = 'outputs/figs/trajectory/trajectory-cca1.pdf'
# # plt.savefig(filepath, dpi=300, bbox_inches='tight')
# # plt.close()
# 
# 
# # # %% Load CCA model and get weights
# # 
# # filepath = 'data/tmp/zhihao/cca.pkl'
# # with open(filepath, 'rb') as f:
# #     cca = pickle.load(f)
# # 
# # model = cca['cca_model']
# # weights = model.x_weights_
# # weights = pd.DataFrame({
# #     'symbol': cmc_combat_0409.index.map(map_uniprot_gene),
# #     'coef': weights[:, 0]
# # }, index=cmc_combat_0409.index)
# # weights.sort_values('coef', ascending=False, key=abs, inplace=True)
# # weights.head()
# # filepath = 'outputs/tmp/cca_weights.csv'
# # weights.to_csv(filepath)
# 
# # uids_cca = weights.index[weights.coef.abs() > 0.1]
# 
# ### Plot trajectory in PCA and UMAP space ###
# 
# def plot_pca_trajectory(x, metadata, filepath):
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax, pca_cvt = bp.plot_pca(
#         ax,
#         x,
#         metadata.loc[x.columns],
#         colourbar=True,
#         return_fig=False,
#         hue='month',
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
# 
# plot_pca_trajectory(
#     cmc_combat_0409.loc[uids_cca],
#     metadata_month,
#     'outputs/figs/trajectory/pca-cca23-cmc_combat_0409.pdf'
# )
# 
# plot_pca_trajectory(
#     cvt.loc[uids_cca],
#     metadata_month,
#     'outputs/figs/trajectory/pca-cca23-cvt_combat_0409.pdf'
# )
# 
# plot_pca_trajectory(
#     cvt.loc[prots_spearman],
#     metadata_month,
#     'outputs/figs/trajectory/pca-spearman8-cvt_combat_0409.pdf'
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
#         hue='month',
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
#     metadata_month,
#     'outputs/figs/trajectory/umap-spearman10-cmc_combat_0409.pdf'
# )

### Protein-protein correlations ###

# TODO: Plot correlation

group_order = [
    'Convert', 'Maintain', 'Control',
    'Early remit', 'Late remit', 'Relapse'
]

metadata_month.group.value_counts()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, g in zip(axes.flat, group_order):
    df = bp.subset(lyriks387_cb0409, metadata_month, f"group == '{g}'").T
    df = df[prots_spearman]
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title(g)

fig.tight_layout(pad=3.0)
fig.savefig('outputs/figs/corrheatmap/corr-groups.pdf', bbox_inches='tight')
plt.close(fig)

# Plot correlation between spearman8 proteins in different groups
lyriks_fep = bp.subset(
    lyriks_cvt, metadata,
    "(timepoint  == 24) & (state == 'FEP')"
)

### Slopes ###

def compute_slope_features(
    X: pd.DataFrame, metadata: pd.DataFrame, func: callable
):
    '''Computes slope features for every protein using func

    Args:
        X: DataFrame of shape (n_features, n_samples)
        metadata: indexed by sample with 'month' and 'sn' columns
        func: A function that accepts a sub-dataframe consisting of
            patient samples with protein columns including 'month' and 'sn'.
            Function has to return a pd.DataFrame with correct index.

    Returns:
        pd.DataFrame: DataFrame of shape (n_patients, n_features)
    '''
    assert X.columns.isin(metadata.index).all()
    X_meta = X.T.join(metadata_month[['month', 'sn']], how='inner')
    X_meta.sort_values('month', inplace=True)
    results = X_meta.groupby('sn').apply(func)
    return results


def compute_velocity(X_meta):
    X = X_meta.iloc[:, :-2]
    n = len(X)
    rows = {}
    for k in range(n, 1, -1):
        X_sub = X.iloc[:k]
        duration = X_meta['month'].iloc[k-1] - X_meta['month'].iloc[0]
        rows[X_sub.index[-1]] = (X_sub.iloc[-1, :] - X_sub.iloc[0, :]) / duration
    return pd.DataFrame(rows).T


def compute_speed(X_meta):
    X = X_meta.iloc[:, :-2]
    n = len(X)
    rows = {}
    for k in range(n, 1, -1):
        X_sub = X.iloc[:k]
        duration = X_meta['month'].iloc[k-1] - X_meta['month'].iloc[0]
        speed = np.abs(np.diff(X_sub, axis=0)).sum(axis=0) / duration
        rows[X_sub.index[-1]] = pd.Series(speed, index=X_sub.columns)
    return pd.DataFrame(rows).T


def compute_kendall_tau(X_meta):
    X = X_meta.iloc[:, :-2]
    n = len(X)
    rows = {}
    for k in range(n, 1, -1):
        subset = X.iloc[:k]
        ranks = np.arange(k)
        rows[subset.index[-1]] = subset.apply(
            lambda col: kendalltau(ranks, col.values).statistic
        )
    return pd.DataFrame(rows).T


def compute_sd(X_meta):
    X = X_meta.iloc[:, :-2]
    return X.std(axis=0, ddof=1)


def compute_cv(X_meta):
    X = X_meta.iloc[:, :-2]
    return X.std(axis=0, ddof=1) / X.mean(axis=0)


# ### Old code ###
# 
# def compute_velocity1(X_meta):
#     X = X_meta.iloc[:, :-2]
#     duration = X_meta['month'][-1] - X_meta['month'][0]
#     velocity = (X.iloc[-1,:] - X.iloc[0,:]) / duration
#     print(type(velocity))
#     return velocity
# 
# 
# def compute_speed1(X_meta):
#     X = X_meta.iloc[:, :-2]
#     duration = X_meta['month'][-1] - X_meta['month'][0]
#     speed = np.abs(np.diff(X, axis=0)).sum(axis=0) / duration
#     return pd.Series(speed, index=X.columns)
# 
# 
# def compute_kendall_tau1(X_meta):
#     X = X_meta.iloc[:, :-2]
#     ranks = np.arange(len(X))
#     return X.apply(lambda col: kendalltau(ranks, col.values).statistic)


### New

taus = compute_slope_features(
    lyriks387_cb0409, metadata_month, compute_kendall_tau
)
velocities = compute_slope_features(
    lyriks387_cb0409, metadata_month, compute_velocity
)
speeds = compute_slope_features(
    lyriks387_cb0409, metadata_month, compute_speed
)

# ### Old
# 
# velocities1 = compute_slope_features(
#     lyriks387_cb0409, metadata_month, compute_velocity1
# )
# speeds1 = compute_slope_features(
#     lyriks387_cb0409, metadata_month, compute_speed1
# )
# taus1 = compute_slope_features(
#     lyriks387_cb0409, metadata_month, compute_kendall_tau1
# )

velocities24 = velocities.groupby(level='sn').nth(0).droplevel(-1)
speeds24 = speeds.groupby(level='sn').nth(0).droplevel(-1)
taus24 = taus.groupby(level='sn').nth(0).droplevel(-1)
rel_velocities24 = velocities24 / speeds24


sn_group = metadata_month[['sn', 'group']].drop_duplicates('sn').set_index('sn')

group_order = [
    'Convert', 'Maintain', 'Control',
    'Early remit', 'Late remit', 'Relapse'
]
group_palette = {
    'Convert': 'tab:red',
    'Maintain': 'tab:blue',
    'Control': 'tab:green',
    'Early remit': 'tab:orange',
    'Late remit': 'tab:purple',
    'Relapse': 'tab:brown',
}

ylabels = {
    'velocity': 'Velocity',
    'speed': 'Speed',
    'rel_velocity': 'Velocity / Speed',
    'tau': "Kendall's tau",
}

for prot in prots_spearman:
    symbol = map_uniprot_gene[prot]
    print(symbol)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (src_df, label) in zip(axes.flat, [
        (velocities24, 'velocity'),
        (speeds24, 'speed'),
        (rel_velocities24, 'rel_velocity'),
        (taus24, 'tau'),
    ]):
        df = src_df[[prot]].join(sn_group, how='left')
        df.columns = [label, 'group']
        df = df[df['group'].isin(group_order)]
        sns.stripplot(
            data=df, x='group', y=label, hue='group', order=group_order,
            hue_order=group_order, palette=group_palette, jitter=True,
            alpha=0.7, legend=False, ax=ax
        )
        groups = [
            df.loc[df['group'] == g, label].dropna().values
            for g in group_order if g in df['group'].values
        ]
        _, pval = f_oneway(*groups)
        ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
        ax.set_title(f'p = {pval:.3g}')
        ax.set_xlabel('')
        ax.set_ylabel(ylabels[label])
        ax.tick_params(axis='x', labelrotation=20)
    plt.tight_layout()
    plt.savefig(
        f'outputs/figs/trajectory/features/features-{symbol}.pdf',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


### Aggregate across slope features ###

avg_features = pd.DataFrame({
    'velocity': velocities24[prots_spearman].mean(axis=1),
    'speed': speeds24[prots_spearman].mean(axis=1),
    'rel_velocity': rel_velocities24[prots_spearman].mean(axis=1),
    'tau': taus24[prots_spearman].mean(axis=1),
}).join(sn_group, how='left')
avg_features = avg_features[avg_features['group'].isin(group_order)]

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
axes = axes.flatten()
fig.suptitle("Average across 8 proteins (Spearman's)")

for ax, label in zip(axes, ylabels):
    sns.stripplot(
        data=avg_features, x='group', y=label, hue='group', order=group_order,
        hue_order=group_order, palette=group_palette, jitter=True,
        alpha=0.7, legend=False, ax=ax
    )
    ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=20)
    ax.set_ylabel(ylabels[label])

plt.tight_layout()
plt.savefig(
    'outputs/figs/trajectory/features/features-mean-spearman8.pdf',
    dpi=300, bbox_inches='tight'
)
plt.close()

# Plot all slopes from earlier timepoints
# Only 7 converters with >2 timepoints, 2 with >3 timepoints 
velocities_ex24 = velocities.groupby(level='sn').nth(slice(1, None))
speeds_ex24 = speeds.groupby(level='sn').nth(slice(1, None))
taus_ex24 = taus.groupby(level='sn').nth(slice(1, None))
rel_velocities_ex24 = velocities_ex24 / speeds_ex24

for prot in prots_spearman:
    symbol = map_uniprot_gene[prot]
    print(symbol)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (src_df, label) in zip(axes.flat, [
        (velocities_ex24, 'velocity'),
        (speeds_ex24, 'speed'),
        (rel_velocities_ex24, 'rel_velocity'),
        (taus_ex24, 'tau'),
    ]):
        print(src_df.index)
        print(sn_group.index)
        df = src_df[[prot]].join(sn_group, how='left')
        df.columns = [label, 'group']
        df = df[df['group'].isin(group_order)]
        sns.stripplot(
            data=df, x='group', y=label, hue='group', order=group_order,
            hue_order=group_order, palette=group_palette, jitter=True,
            alpha=0.7, legend=False, ax=ax
        )
        groups = [
            df.loc[df['group'] == g, label].dropna().values
            for g in group_order if g in df['group'].values
        ]
        _, pval = f_oneway(*groups)
        ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
        ax.set_title(f'p = {pval:.3g}')
        ax.set_xlabel('')
        ax.set_ylabel(ylabels[label])
        ax.tick_params(axis='x', labelrotation=20)
    plt.tight_layout()
    plt.savefig(
        f'outputs/figs/trajectory/features/features_ex24-{symbol}.pdf',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


### Aggregate across slope features ###

avg_features_ex24 = pd.DataFrame({
    'velocity': velocities_ex24[prots_spearman].mean(axis=1),
    'speed': speeds_ex24[prots_spearman].mean(axis=1),
    'rel_velocity': rel_velocities_ex24[prots_spearman].mean(axis=1),
    'tau': taus_ex24[prots_spearman].mean(axis=1),
}).join(sn_group, how='left')
avg_features_ex24 = avg_features_ex24[avg_features_ex24['group'].isin(group_order)]

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
axes = axes.flatten()
fig.suptitle("Average across 8 proteins (Spearman's) - excl. M24")

for ax, label in zip(axes, ylabels):
    print(label)
    sns.stripplot(
        data=avg_features_ex24, x='group', y=label, hue='group', order=group_order,
        hue_order=group_order, palette=group_palette, jitter=True,
        alpha=0.7, legend=False, ax=ax
    )
    ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=20)
    ax.set_ylabel(ylabels[label])

plt.tight_layout()
plt.savefig(
    'outputs/figs/trajectory/features/features_ex24-mean-spearman8.pdf',
    dpi=300, bbox_inches='tight'
)
plt.close()


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
# agg = metadata_month.loc[
#     cvt.columns,
#     ['sn', 'timepoint', 'month']
# ].copy()
# agg['cvt_up'] = cvt.loc[top_up].sum(axis=0)
# agg['cvt_down'] = cvt.loc[top_down].sum(axis=0)
# agg_mnt = pd.DataFrame({
#     'mnt_up': mnt.loc[top_up].sum(axis=0),
#     'mnt_down': mnt.loc[top_down].sum(axis=0)
# }).join(
#     metadata_month[['sn', 'timepoint', 'extraction_date']],
#     how='left'
# )
# agg_ctrl = pd.DataFrame({
#     'ctrl_up': ctrl.loc[top_up].sum(axis=0),
#     'ctrl_down': ctrl.loc[top_down].sum(axis=0)
# }).join(
#     metadata_month[['sn', 'timepoint', 'extraction_date']],
#     how='left'
# )
# 
# # Plot convert patients
# rho = spearmanr(agg.month, agg.cvt_up).correlation
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(
#     agg.month,
#     agg.cvt_up,
#     c='tab:green'
# )
# ax.set_title(f'Top 3 up-regulated (r = {rho:.2f})')
# ax.set_xlabel("Months to conversion")
# for _, patient in agg.sort_values("timepoint").groupby("sn"):
#     ax.plot(
#         patient.month,
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

# metadata_convert = metadata_month.query(
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
# lyriks_corr_int = lyriks_corr.T.join(metadata_month, how='inner')

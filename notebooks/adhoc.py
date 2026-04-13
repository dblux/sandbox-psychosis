import pandas as pd
import numpy as np


filepath = 'data/tmp/jieyin/biomarkers.csv'
biomarkers = pd.read_csv(filepath, index_col=1)

filepath = 'data/processed/reprocessed-data-renamed.csv'
data = pd.read_csv(filepath, index_col=0)
data.replace(0, np.nan, inplace=True)
lyriks = data.iloc[:, data.columns.str.startswith('L')].copy()
lyriks = np.log2(lyriks)

uids2 = biomarkers.index[biomarkers.index.isin(data.index)]
lyriks_bm2 = lyriks.loc[uids2, :]
lyriks_bm2.to_csv('outputs/tmp/jieyin/lyriks_bm2.csv')

# # %% Olink
# filepath = 'data/etc/olink-csa.csv'
# olink_stats = pd.read_csv(filepath, index_col=0)
# olink_stats.index

# filepath = 'data/astral/etc/annotation-olink_75.csv'
# annot_olink = pd.read_csv(filepath, index_col=0)
# map_olink_uniprot = {
#     k: v for k, v in zip(annot_olink.index, annot_olink.uniprot)
# }

# csa_full_gene = csa_full.rename(index=map_uniprot_gene)
# olink_stats_uniprot = olink_stats.rename(index=map_olink_uniprot)
# 
# olink_stats_uniprot.index.isin(csa_full.index).sum() # 0/75 proteins in LINKS are in csa (249)
# olink_stats_uniprot.index.isin(csa.index).sum() # 1/75 proteins in LINKS are in csa (1757)
# csa_full.index.sort_values().tolist()
# olink_stats_uniprot.index.sort_values().tolist(

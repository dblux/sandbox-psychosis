import pandas as pd
import numpy as np


# Extraction date
filepath = 'data/metadata/metadata_experimental-all_645_13.csv'
expt_metadata = pd.read_csv(filepath, index_col=0)
expt_metadata.rename(columns={'Sample.Name': 'sn'}, inplace=True)
expt_metadata.drop(columns='sn', inplace=True)
# expt_metadata.sn = expt_metadata.sn.str.replace('_00', '_0')
# expt_metadata.sn = expt_metadata.sn.str.replace('_06', '_6')

# State
filepath = 'data/metadata/ZH-states-all.csv'
states = pd.read_csv(filepath, index_col=0)
states['sn'] = states.index
states.sn = states.sn.str.replace('_.*$', '', regex=True)

### LYRIKS metadata ###

# filepath = 'data/processed/lyriks_605_402_01-knn5.csv'
# lyriks_data = pd.read_csv(filepath, index_col=0)

filepath = 'data/metadata/LYRIKS/lyriks-baseline_medication.csv'
baseline_med = pd.read_csv(filepath, index_col=0)

# three uncategorised drugs
filepath = 'data/metadata/LYRIKS/metadata_392_57.csv'
metadata57 = pd.read_csv(filepath, index_col=0)
metadata57.shape

filepath = 'data/metadata/LYRIKS/metadata_2277_73.csv'
metadata73 = pd.read_csv(filepath, index_col=0)

filepath = 'data/metadata/metadata10-lyriks.csv'
metadata10 = pd.read_csv(filepath, index_col=0)

# antidepressant and anxiolytics use
filepath = 'data/metadata/LYRIKS/metadata_65_60-antidepressant_anxiolytics-JY.csv'
lyriks_jy = pd.read_csv(filepath, index_col=0)

filepath = 'data/metadata/metadata_blood_collection-lyriks.csv'
lyriks_collection = pd.read_csv(filepath)
lyriks_collection.index = lyriks_collection.sn + \
    lyriks_collection.is_control + '_' + \
    lyriks_collection.timepoint.astype(str)
lyriks_collection['date'] = pd.to_datetime(lyriks_collection.date, format='mixed')

# metadata10.index[~metadata10.index.isin(lyriks_collection.index)]
# lyriks_collection.iloc[200:250]

### CSA metadata ###

filepath = 'data/metadata/metadata-csa_200_37.csv'
csa_metadata = pd.read_csv(filepath, index_col=0)
csa_metadata.head()
csa_metadata.columns

### Change L0673_18 to L0673S_24 ###
states.index = states.index.str.replace('L0673S_18', 'L0673S_24')
metadata10.index = metadata10.index.str.replace('L0673S_18', 'L0673S_24')
expt_metadata.index = expt_metadata.index.str.replace('L0673S_18', 'L0673S_24')

### Integrate experimental metadata ###
metadata_expt = expt_metadata[[
    'Concentration.ng.ul.', 'Volume.ul.', 'Total.amount.ug.',
    'Study', 'Extraction.Date', 'Run.DateTime'
]].copy()
metadata_expt.columns = [
    'concentration', 'volume', 'total_amount',
    'study', 'extraction_date', 'run_datetime'
]
metadata_expt = metadata_expt[~metadata_expt.index.str.startswith('QC')]
metadata_expt.extraction_date.value_counts()
# Integrate collection datetime (missing for bipolar cohort)
collection_datetime = pd.to_datetime(pd.concat([
    lyriks_collection.date,
    csa_metadata.collection_datetime
]), format='mixed')
collection_datetime = collection_datetime.rename('collection_datetime')
metadata_expt = metadata_expt.join(collection_datetime, how='left')

# CSA
csa = csa_metadata[['group', 'age', 'bmi', 'gender', 'ethnicity', 'smoking']].copy()
csa.smoking.value_counts()
csa.smoking.replace({'0': False, '1': True, ' ': pd.NA}, inplace=True)
csa.insert(0, 'timepoint', 0) 
csa.insert(0, 'sn', csa.index)
csa.smoking.tolist()
csa.index[csa.smoking.isna()] # CA114, CA155

# LYRIKS
lyriks = metadata73[[
    'sn', 'Period', 'age', 'bmi', 'gend', 'eth', 'smoke_stat'
]].join(metadata10[['label']], how='inner')
lyriks.gend = np.where(lyriks.gend == 2, 'Male', 'Female')
lyriks.eth = lyriks.eth.str.capitalize()
lyriks.smoke_stat = lyriks.smoke_stat.map({
    'non_smoker': False, 'quitted': False,
    'light': True, 'moderate': True, 'heavy': True
})
# Reorder columns
lyriks = lyriks[[
    'sn', 'Period', 'label',
    'age', 'bmi', 'gend', 'eth', 'smoke_stat'
]]
lyriks.label = lyriks.label.str.replace('_', ' ').str.capitalize()
print(pd.DataFrame({'lyriks': lyriks.columns, 'csa': csa.columns}))
lyriks.columns = csa.columns

# Integrate
psy_demo = pd.concat([lyriks, csa])
psy = psy_demo.join(metadata_expt, how='left')
print(psy.columns)
print(psy.group.value_counts())

### Retrieve state of sample from metadata ###

caarms_map = {2: 'Control', 1: 'UHR', 4: 'FEP'}
schizo_groups = [
    'Antipsychotic responsive',
    'Clozapine responsive',
    'Clozapine resistant'
]
group_map = {
    'Control': 'Control',
    'Healthy control': 'Control',
    **{g: 'Schizophrenia' for g in schizo_groups}
}
psy['state'] = psy.group.map(group_map).fillna(
    metadata73.caarms_stat.map(caarms_map)
)
print(psy.state.value_counts())
# L0073S_24 sample is actually six months before FEP
# Relabel state of L0073S_24 as UHR instead of FEP
psy.loc['L0073S_24', 'state'] = 'UHR'
# print(psy.loc[psy.state.isna(), 'group'])
print(psy.shape)

filepath = 'data/metadata/metadata-psy_602_16-v2.csv'
psy.to_csv(filepath, index=True)

### Sanity checks ###

filepath = 'data/metadata/metadata-psy_602_16-v1.csv'
psy_v1 = pd.read_csv(filepath, index_col=0)

filepath = 'data/metadata/metadata-psy_602_16-v2.csv'
psy_v2 = pd.read_csv(filepath, index_col=0)

state_df = pd.DataFrame({'v1': psy_v1.state, 'v2': psy_v2.state})
print(state_df[state_df.v1 != state_df.v2])
# Conclusion: States of v2 are correct!
    

# # Check with Astral dataset
# filepath = 'data/metadata/metadata-psy_602_16.csv'
# metadata = pd.read_csv(filepath, index_col=0)
# metadata.head()
# metadata.loc['L0073S_24']
# 
# # lyriks_data.columns.str.startswith('L0417S_24').any()
# # lyriks_collection.loc['L0417S_24']
# 
# ### Compute m2c from collection months ###
# 
# cvt_collection = lyriks_collection.query('is_convert == True').dropna()
# # Remove L0167S as it is wrongly indicated as convert. It is not in Astral data.
# cvt_collection = cvt_collection.loc[~(cvt_collection.sn == 'L0167')]
# 
# def calc_time_diff(group):
#     group = group.copy()
#     group['collection_months'] = (group['date'] - group['date'].min()) / pd.Timedelta(days=30.44)
#     return group
# 
# cvt_patients = cvt_collection.groupby('sn', group_keys=False)
# cvt_collection = cvt_patients.apply(calc_time_diff)
# cvt_collection['month_of_conversion'] = metadata73.loc[
#     cvt_collection.index, 'month_of_conversion'
# ]
# cvt_collection['fep_delta'] = cvt_collection['collection_months'] - cvt_collection['month_of_conversion']
# 
# filepath = 'tmp/cvt-fep_delta.csv'
# cvt_collection.to_csv(filepath)
# 
# # TODO: Compute m2c from collection_months for maintain patients as well
# 
# ### Misc. ###
# 
# lyriks_collection.shape
# lyriks_collection_406 = lyriks_collection[~lyriks_collection.date.isna()]
# 
# cvt24 = cvt_collection.loc[
#     cvt_collection.timepoint == 24,
#     ['sn', 'timepoint', 'month_of_conversion', 'collection_months']
# ]
# 
# filepath = 'tmp/lyriks-collection_timediff_month_of_conversion.csv'
# cvt24.to_csv(filepath, index=False)
# 
# # TODO: medication
# csa_metadata
# 
# scid = lyriks_jy.iloc[:, 2:58:].copy()
# scid.replace({-9999: 0}, inplace=True)
# (scid == 1).any(axis=1).sum()
# 
# # TODO: comorbidities
# # csa_metadata has data on comorbidities
# # where is the LYRIKS one?
# 
# ### Check possible mislabelling of L0673S_18 in proteomics data ###
# 
# mask = (metadata73['month_of_conversion'].notna()) # & (metadata73['Period'] == 24)
# cvt_meta = metadata73.loc[mask, metadata73.columns[:7]]
# cvt_meta_size = cvt_meta.groupby('sn').size().to_frame()
# 
# states_cvt = states[(states.stage_label == 'convert')]
# cvt_size = states_cvt.groupby('sn').size().to_frame()
# 
# cvt_size.join(cvt_meta_size, how='outer', lsuffix='_states', rsuffix='_metadata')
# 
# # Patients (convert) that do not have proteomics data
# missing_cvt = ['L0333S', 'L0336S', 'L0435S', 'L0635S', 'L0651S']
# 
# # Patients (convert) with lesser timepoint data than expected
# lesser_cvt = ['L0073S', 'L0141S', 'L0561S', 'L0609S', 'L0673S']
# # Only L0673 has sample 18 (which is not recorded in EMR) but lacks 24
# # L0673 converted at M19 (M18 'should be' labelled as 'M24')
# 
# metadata73.loc[metadata73.sn.isin(lesser_cvt), metadata73.columns[:7]]
# states[states.sn.isin(lesser_cvt)]
# states.cohort.value_counts()

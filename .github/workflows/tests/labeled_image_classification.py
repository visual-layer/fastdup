import pandas as pd
data_dir = 'imagenette2-160/'
csv_path = 'imagenette2-160/noisy_imagenette.csv'

label_map = {
    'n02979186': 'cassette_player', 
    'n03417042': 'garbage_truck', 
    'n01440764': 'tench', 
    'n02102040': 'English_springer', 
    'n03028079': 'church',
    'n03888257': 'parachute', 
    'n03394916': 'French_horn', 
    'n03000684': 'chain_saw', 
    'n03445777': 'golf_ball', 
    'n03425413': 'gas_pump'
}

df_annot = pd.read_csv(csv_path)
# take relevant columns
df_annot = df_annot[['path', 'noisy_labels_0']]

# rename columns to fastdup's column names
df_annot = df_annot.rename({'noisy_labels_0': 'label', 'path': 'filename'}, axis='columns')

# append datadir
df_annot['filename'] = df_annot['filename'].apply(lambda x: data_dir + x)

# create split column
df_annot['split'] = df_annot['filename'].apply(lambda x: x.split("/")[1])

# map label ids to regular labels
df_annot['label'] = df_annot['label'].map(label_map)


import fastdup
print(f'fastdup version: {fastdup.__version__}')

work_dir = 'fastdup_work_dir'
fd = fastdup.create(work_dir=work_dir, input_dir=data_dir) 
fd.run(annotations=df_annot, ccthreshold=0.9, threshold=0.8)

fd.vis.duplicates_gallery(num_images=5)
fd.vis.component_gallery(num_images=5)
fd.vis.component_gallery(slice='chain_saw')
fd.vis.outliers_gallery(num_images=5)
fd.vis.similarity_gallery() 

fd.vis.stats_gallery(metric='dark', num_images=5)
fd.vis.stats_gallery(metric='bright', num_images=5)
fd.vis.stats_gallery(metric='blur', num_images=5)
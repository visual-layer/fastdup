import fastdup
print(f'fastdup version: {fastdup.__version__}')

fd = fastdup.create(work_dir="fastdup_work_dir/", input_dir="images/")
fd.run(num_images=10000)

fd.vis.duplicates_gallery()
fd.vis.outliers_gallery() 
fd.vis.stats_gallery(metric='dark')
fd.vis.component_gallery()
fd.vis.similarity_gallery()
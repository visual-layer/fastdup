# Fastdup documentation page

The main function is `fastdup.run`. It runs on a folder or list of images and computes the artifacts needed to compute image similrity, outliers, componetns etc.

:::fastdup.run


## Fastdup visualization of results
Visualization of the output data is done using the following functions:

`fastdup.create_duplicates_gallery`
:::fastdup.create_duplicates_gallery

`fastdup.create_outliers_gallery`
:::fastdup.create_outliers_gallery

`fastdup.create_components_gallery`
:::fastdup.create_components_gallery

`fastdup.create_kmeans_clusters_gallery`
:::fastdup.create_kmeans_clusters_gallery

`fastdup.create_stats_gallery`
:::fastdup.create_stats_gallery

`fastdup.create_similarity_gallery`
:::fastdup.create_similarity_gallery

`fastdup.create_aspect_ratio_gallery`
:::fastdup.create_aspect_ratio_gallery

## Fastdup classifiers
Given fastdup output compute a baseline lightweight classifier
`fastdup.create_knn_classifier`
:::fastdup.create_knn_classifier

`fastdup.create_kmeans_classifier`
:::fastdup.create_kmeans_classifier

## Fastdup utilities
Loading the binary feature resulting in fastdup run can be done by `fastdup.load_binary_features`.

`fastdup.load_binary_feature`
:::fastdup.load_binary_feature

`fastdup.save_binary_feature`
:::fastdup.save_binary_feature

`fastdup.generate_sprite_image`
:::fastdup.generate_sprite_image

`fastdup.export_to_tensorboard_projector`
:::fastdup.export_to_tensorboard_projector


`fastdup.export_to_cvat`
:::fastdup.export_to_cvat

`fastdup.export_to_labelImg`
:::fastdup.export_to_labelImg

## Fastdup utilities to remove images

`fastdup.delete_components`
:::fastdup.delete_components

`fastdup.delete_or_retag_stats_outliers`
:::fastdup.delete_or_retag_stats_outliers




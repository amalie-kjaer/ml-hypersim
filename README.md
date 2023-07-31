# Research in Data Science: Graph Neural Networks for Scene Understanding

## Setting up the environment

If you're using Anaconda, you can install all of the required Python libraries using our `requirements.txt` file.

```
conda create --name ds --file requirements.txt
conda activate ds
conda install pyg -c pyg
```

## Downloading the processed Hypersim dataset

To obtain our image dataset, you can run the following download script. On Windows, you'll need to modify the script so it doesn't depend on the `curl` and `unzip` command-line utilities.

```
python code/python/tools/dataset_download_images.py --downloads_dir /Volumes/portable_hard_drive/downloads --decompress_dir /Volumes/portable_hard_drive/evermotion_dataset/scenes
```

Note that our dataset is roughly 1.9TB. We have partitioned the dataset into a few hundred separate ZIP files, where each ZIP file is between 1GB and 20GB. Our [download script](code/python/tools/dataset_download_images.py) contains the URLs for each ZIP file. [Thomas Germer](https://github.com/99991) has generously contributed an [alternative download script](contrib/99991) that can be used to download subsets of files from within each ZIP archive.

Note also that we manually excluded images containing people and prominent logos from our public release, and therefore our public release contains 74,619 images, rather than 77,400 images. We list all the images we manually excluded in `ml-hypersim/evermotion_dataset/analysis/metadata_images.csv`.

To obtain the ground truth triangle meshes for each scene, you must purchase the asset files [here](https://www.turbosquid.com/Search/3D-Models?include_artist=evermotion).

&nbsp;

## Working with the raw Hypersim dataset

The Hypersim Dataset consists of a collection of synthetic scenes. Each scene has a name of the form `ai_VVV_NNN` where `VVV` is the volume number, and `NNN` is the scene number within the volume. For each scene, there are one or more camera trajectories named {`cam_00`, `cam_01`, ...}. Each camera trajectory has one or more images named {`frame.0000`, `frame.0001`, ...}. Each scene is stored in its own ZIP file according to the following data layout:

```
ai_VVV_NNN
├── _detail
│   ├── metadata_cameras.csv                     # list of all the camera trajectories for this scene
│   ├── metadata_node_strings.csv                # all human-readable strings in the definition of each V-Ray node
│   ├── metadata_nodes.csv                       # establishes a correspondence between the object names in an exported OBJ file, and the V-Ray node IDs that are stored in our render_entity_id images
│   ├── metadata_scene.csv                       # includes the scale factor to convert asset units into meters
│   ├── cam_XX                                   # camera trajectory information
│   │   ├── camera_keyframe_orientations.hdf5    # camera orientations
│   │   └── camera_keyframe_positions.hdf5       # camera positions (in asset coordinates)
│   ├── ...
│   └── mesh                                                                            # mesh information
│       ├── mesh_objects_si.hdf5                                                        # NYU40 semantic label for each object ID (available in our public code repository)
│       ├── mesh_objects_sii.hdf5                                                       # semantic instance ID for each object ID (available in our public code repository)
│       ├── metadata_objects.csv                                                        # object name for each object ID (available in our public code repository)
│       ├── metadata_scene_annotation_tool.log                                          # log of the time spent annotating each scene (available in our public code repository)
│       ├── metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5      # length (in asset units) of each dimension of the 3D bounding for each semantic instance ID
│       ├── metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5 # orientation of the 3D bounding box for each semantic instance ID
│       └── metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5    # position (in asset coordinates) of the 3D bounding box for each semantic instance ID
└── images
    ├── scene_cam_XX_final_hdf5                  # lossless HDR image data that requires accurate shading
    │   ├── frame.IIII.color.hdf5                # color image before any tone mapping has been applied
    │   ├── frame.IIII.diffuse_illumination.hdf5 # diffuse illumination
    │   ├── frame.IIII.diffuse_reflectance.hdf5  # diffuse reflectance (many authors refer to this modality as "albedo")
    │   ├── frame.IIII.residual.hdf5             # non-diffuse residual
    │   └── ...
    ├── scene_cam_XX_final_preview               # preview images
    |   └── ...
    ├── scene_cam_XX_geometry_hdf5               # lossless HDR image data that does not require accurate shading
    │   ├── frame.IIII.depth_meters.hdf5         # Euclidean distances (in meters) to the optical center of the camera
    │   ├── frame.IIII.position.hdf5             # world-space positions (in asset coordinates)
    │   ├── frame.IIII.normal_cam.hdf5           # surface normals in camera-space (ignores bump mapping)
    │   ├── frame.IIII.normal_world.hdf5         # surface normals in world-space (ignores bump mapping)
    │   ├── frame.IIII.normal_bump_cam.hdf5      # surface normals in camera-space (takes bump mapping into account)
    │   ├── frame.IIII.normal_bump_world.hdf5    # surface normals in world-space (takes bump mapping into account)
    │   ├── frame.IIII.render_entity_id.hdf5     # fine-grained segmentation where each V-Ray node has a unique ID
    │   ├── frame.IIII.semantic.hdf5             # NYU40 semantic labels
    │   ├── frame.IIII.semantic_instance.hdf5    # semantic instance IDs
    │   ├── frame.IIII.tex_coord.hdf5            # texture coordinates
    │   └── ...
    ├── scene_cam_XX_geometry_preview            # preview images
    |   └── ...
    └── ...
```

### Dataset split

We include a standard train/val/test split in `ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv`. We refer to this split as the _v1_ split of our dataset. We generated this split by randomly partitioning our data at the granularity of scenes, rather than images or camera trajectories, to minimize the probability of very similar images ending up in different partitions. In order to maximize reproducibilty, we only include publicly released images in our split.

In the Hypersim dataset, there is a small amount of asset reuse across scenes. This asset reuse is difficult to detect by analyzing metadata in the original scene assets, but it is evident when manually browsing through our rendered images. We do not attempt to address this issue when generating our split. Therefore, individual objects in our training images are occasionally also present in our validation and test images.
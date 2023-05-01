#!/usr/bin/env bash
python download.py --contains metadata_semantic_instance_bounding_box_object_aligned_2d_ --directory /cluster/scratch/akjaer/ds_data --silent &&
python download.py --contains metadata_scene --directory /cluster/scratch/akjaer/ds_data --silent &&
python download.py --contains camera_keyframe_ --directory /cluster/scratch/akjaer/ds_data --silent &&
python download.py --contains tonemap --directory /cluster/scratch/akjaer/ds_data --silent &&
python download.py --contains semantic_instance --directory /cluster/scratch/akjaer/ds_data --silent
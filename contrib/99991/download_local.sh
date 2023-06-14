#!/usr/bin/env bash
python download.py --contains ai_001_007 --contains metadata_semantic_instance_bounding_box_object_aligned_2d_ --silent &&
python download.py --contains ai_001_007 --contains metadata_scene --silent &&
python download.py --contains ai_001_007 --contains camera_keyframe_ --silent &&
python download.py --contains ai_001_007 --contains tonemap.jpg --silent &&
python download.py --contains ai_001_007 --contains semantic_instance.hdf5 --silent
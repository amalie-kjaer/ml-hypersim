import h5py
import json
import pandas as pd
import os
import numpy as np
import glob
import io
from configparser import ConfigParser
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import wandb
from PIL import Image

def load_config():
    config = ConfigParser()
    config.read('./config.yaml')
    return config

def construct_M(height_pixels, width_pixels):
    fov_x = np.pi/3.0
    fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x/2.0) / width_pixels)
    near  = 1.0
    far   = 1000.0

    f_h    = np.tan(fov_y/2.0)*near
    f_w    = f_h*width_pixels/height_pixels
    left   = -f_w
    right  = f_w
    bottom = -f_h
    top    = f_h

    M_proj      = np.matrix(np.zeros((4,4)))
    M_proj[0,0] = (2.0*near)/(right - left)
    M_proj[1,1] = (2.0*near)/(top - bottom)
    M_proj[0,2] = (right + left)/(right - left)
    M_proj[1,2] = (top + bottom)/(top - bottom)
    M_proj[2,2] = -(far + near)/(far - near)
    M_proj[3,2] = -1.0
    M_proj[2,3] = -(2.0*far*near)/(far - near)
    
    return M_proj

def world2screen_proj(p_world, camera_pos, camera_rot, height_pixels, width_pixels):
    R_cam2world = np.matrix(camera_rot)
    t_cam2world = np.matrix(camera_pos).T
    R_world2cam = R_cam2world.T
    t_world2cam = -R_world2cam*t_cam2world

    M = construct_M(height_pixels, width_pixels)

    p_cam      = t_world2cam + R_world2cam*p_world
    p_cam_     = np.matrix(np.r_[ p_cam.A1, 1 ]).T
    p_clip     = M * p_cam_
    p_ndc      = p_clip/p_clip[3]
    p_ndc_     = p_ndc.A1
    p_screen_x = 0.5*(p_ndc_[0]+1)*(width_pixels-1)
    p_screen_y = (1 - 0.5*(p_ndc_[1]+1))*(height_pixels-1)
    p_screen_z = (p_ndc_[2]+1)/2.0
    p_screen   = np.matrix([p_screen_x, p_screen_y, p_screen_z]).T
    
    return p_screen

def visualize_graph(dataset, idx, display=False):
    config = load_config()
    
    # Print which data_idx.pt sample is plotted (idx is different for full and small datasets)
    if config['dataset']['dataset_type'] !=  'dataset_full':
        file = './checkpoints/graphs.json'
        with open(file, 'r') as f:
            graphs = json.load(f)
        old_i = int(graphs[idx].split('_')[1].split('.')[0])
        print(f'Current graph visualised: data_{old_i}.pt')
    else:
        print(f'Current graph visualised: data_{idx}.pt')

    # Determine scene name, camera name and frame no. from idx
    file_path = './processed/data_info.csv'
    temp = pd.read_csv(file_path)
    data = temp.to_numpy()
    scene_name = data[old_i][1]
    camera_name = data[old_i][2]
    frame_idx = data[old_i][3]
    
    # Get current frame number (different from frame idx as some frames are missing)
    download_dir = config['path']['download_path']
    scene_dir = os.path.join(download_dir, scene_name)
    images_dir = os.path.join(scene_dir, "images")
    tonemaps_dir = os.path.join(images_dir, "scene_" + camera_name + "_final_preview")
    frames_in_folder = [int(os.path.basename(f).split('.')[1]) for f in glob.glob(os.path.join(tonemaps_dir, '*'))]
    current_frame = frames_in_folder[frame_idx]

    # Plot corresponding image
    tonemap_dir = os.path.join(download_dir, scene_name, "images", f"scene_{camera_name}_final_preview", f"frame.{current_frame:04d}.tonemap.jpg")
    tonemap = mpimg.imread(tonemap_dir)
    y_lim, x_lim = tonemap.shape[:-1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(tonemap, extent=extent, interpolation='nearest')

    # Load 3D positions of bounding boxes in the scene
    bb_pos_dir = os.path.join(download_dir, scene_name, "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5")
    with h5py.File(bb_pos_dir, "r") as f: bb_pos = f['dataset'][:]

    # Load segmentation .hdf5 files
    # and determine which bounding boxes from the scene are present in the frame
    geometry_files_dir = os.path.join(download_dir, scene_name, "images/scene_" + camera_name + "_geometry_hdf5")
    segmentation_file = f'frame.{current_frame:04d}.semantic_instance.hdf5'
    segmentation_dir =  os.path.join(geometry_files_dir, segmentation_file)
    with h5py.File(segmentation_dir, "r") as f: segmentation = f['dataset'][:]

    bb_in_sample = np.unique(segmentation)
    if bb_in_sample[0] == -1: bb_in_sample = bb_in_sample[1:] # Discard -1 label (pixels in segmentation map with unidentified BB)

    if eval(config['dataset']['remove_no_nyu']):
        n_bb = bb_pos.shape[0]-1
        mesh_objects_si_dir = os.path.join("evermotion_dataset", "scenes", scene_name, "_detail", "mesh", "mesh_objects_si.hdf5")
        mesh_objects_sii_dir = os.path.join("evermotion_dataset", "scenes", scene_name, "_detail", "mesh", "mesh_objects_sii.hdf5")
        with h5py.File(mesh_objects_si_dir, "r") as f: mesh_objects_si = f['dataset'][:]
        with h5py.File(mesh_objects_sii_dir, "r") as f: mesh_objects_sii = f['dataset'][:]

        # Define no_nyu
        no_nyu = []
        bb_labels = []
        for i in range(n_bb):
            lowlvl_instances_in_current_bb = np.where(mesh_objects_sii == i+1)[0]
            if lowlvl_instances_in_current_bb.size > 0:
                nyu_id = mesh_objects_si[lowlvl_instances_in_current_bb[0]]
                if nyu_id == 40 or nyu_id == 39 or nyu_id == -1 or nyu_id == 23:
                    no_nyu.append(i+1)
                bb_labels.append(nyu_id)
            else:
                bb_labels.append(np.array([-1], dtype=np.int64))

        # Remove no_nyu
        temp = np.arange(1, n_bb+1)
        bb_not_in_sample = np.delete(temp, bb_in_sample - np.ones(len(bb_in_sample), dtype=int))

        bb_not_in_sample = bb_not_in_sample.tolist()
        for i in range(len(no_nyu)):
            if no_nyu[i] not in bb_not_in_sample:
                bb_not_in_sample.append(no_nyu[i])
        bb_not_in_sample.sort()
        bb_not_in_sample = np.array(bb_not_in_sample)

        bb_in_sample = np.delete(temp.astype(int), bb_not_in_sample.astype(int) - np.ones(len(bb_not_in_sample), dtype=int)).astype(int)

    # Load camera rotations and translations, and select relevant transform
    camera_pos_dir = os.path.join(download_dir, scene_name, "_detail", camera_name, "camera_keyframe_positions.hdf5")
    camera_rot_dir = os.path.join(download_dir, scene_name, "_detail", camera_name, "camera_keyframe_orientations.hdf5")
    with h5py.File(camera_pos_dir, "r") as f: camera_pos_all = f['dataset'][:]
    with h5py.File(camera_rot_dir, "r") as f: camera_rot_all = f['dataset'][:]

    camera_pos = camera_pos_all[current_frame]
    camera_rot = camera_rot_all[current_frame]
    height_pixels = y_lim
    width_pixels = x_lim

    # Calculate 2D postions of bounding box centres
    bb_pos_nodes = {}
    for j, k in enumerate(bb_in_sample):
        bb_pos_ = np.expand_dims(bb_pos[k], 1) # homogenous coordinates
        bb_pos_nodes_screen = world2screen_proj(bb_pos_, camera_pos, camera_rot, height_pixels, width_pixels)

        x = np.ravel(bb_pos_nodes_screen[0])
        x = x.astype(int)
        x = x.item()
        x = np.clip(x, 0, width_pixels)

        y = np.ravel(bb_pos_nodes_screen[1])
        y = y.astype(int)
        y = y.item()
        y = - y + height_pixels
        y = np.clip(y, 0, height_pixels)

        bb_pos_nodes[j] = (x, y)
    
    # Construct labels for plotting
    semantic_dict = {row[0]: row[1] for row in dataset.nyu_labels[1:]}
    semantic_dict['-1'] = 'n/a'
    custom_labels_dict = {j: semantic_dict[str(label.item())] for j, label in enumerate(dataset[idx][0].x)}
    nodes_to_remove = {'otherprop', 'otherfurniture', 'otherstructure'} # not plotting 'other' node labels to avoid clutter
    selected_labels_dict = {node: label for node, label in custom_labels_dict.items() if label not in nodes_to_remove}

    # Plot graph on top of image
    vis = to_networkx(dataset[idx][0])
    nx.draw_networkx(vis, pos=bb_pos_nodes, node_size=10, width=0.5, with_labels=True, labels=selected_labels_dict, font_color='red', edge_color='white')
    # nx.draw_networkx(vis, pos=bb_pos_nodes, node_size=10, width=0.1, edge_color='white')
    classes = [int(c) for c in config['dataset']['classes'].split(',')]
    plt.title(dataset.scene_names[classes][dataset[idx][0].y.item()])
    plt.axis('off')
    
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png', pad_inches=0)
    
    if display == True:
        plt.show()
    
    plt.close()
    image_data = image_buffer.getvalue()

    return image_data

def log_visualization_wandb(dataset, idx, caption):
    image_data = visualize_graph(dataset, idx)
    image = Image.open(io.BytesIO(image_data))
    image_array = np.array(image)
    return wandb.Image(image_array, caption=caption)
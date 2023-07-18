import numpy as np
import os
import glob
import h5py
import csv
import torch
from scipy.spatial import distance
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from utils import *

class HypersimDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, classes=False, presaved_graphs=None):
        self.config = load_config()
        self.nyu_labels = np.genfromtxt(os.path.join("code", "cpp", "tools", "scene_annotation_tool", "semantic_label_descs.csv"), delimiter=',', dtype=None, encoding=None, autostrip=True)
        self.y_labels, self.scene_metadata, self.scene_metadata_w_cams = self._get_y()
        self.scene_names = np.unique(self.scene_metadata[:,2])
        self.processed_dir = self.config['path']['processed_path']

        if classes:

            self.classes = classes
            if presaved_graphs:
                self.graphs = presaved_graphs
            else:
                self.graphs = [f'data_{idx}.pt' for idx in range(self.len())]
                self.graphs = [graph for graph in self.graphs if self._is_valid(graph)]

            self.class_mapping = {class_id: index for index, class_id in enumerate(classes)}
        
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        download_dir = self.config['path']['download_path']
        scenes = [x[-10:] for x in glob.glob(download_dir + "/*")]
        return scenes

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.len())]

    def download(self):
        pass

    def process(self):
        print('PROCESSING FILES...')
        idx = 0

        download_dir = self.config['path']['download_path']

        csv_path = os.path.join(self.processed_dir, 'data_info.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Data File', 'Scene Name', 'Camera Name', 'Frame ID'])


        for scene_name in tqdm(self.raw_file_names, total=len(self.raw_file_names)):

            scene_dir = os.path.join(download_dir, scene_name)
            images_dir = os.path.join(scene_dir, "images")
            detail_dir = os.path.join(scene_dir, "_detail")
            n_cams = len(glob.glob(detail_dir + "/*/"))-1
            camera_names = ["cam_0" + str(n) for n in np.arange(n_cams)]

            try:
                bb_pos, mesh_objects_si, mesh_objects_sii, metadata_objects, a2m = self._import_scene(scene_name, scene_dir)
            except:
                continue
            n_bb = bb_pos.shape[0] - 1
            bb_labels, bb_error = self._assign_labels(mesh_objects_sii, mesh_objects_si, metadata_objects, self.nyu_labels, bb_pos)
            
            for camera_name in camera_names:
                geometry_files_dir = os.path.join(images_dir, "scene_" + camera_name + "_geometry_hdf5")                

                segmentation_files_dir = os.path.join(images_dir, "scene_" + camera_name + "_geometry_hdf5", "frame.*.semantic_instance.hdf5")
                filenames_segmentation = [ os.path.basename(f) for f in np.sort(glob.glob(segmentation_files_dir)) ]
                n_frames = len(filenames_segmentation)

                threshold = 1.5
                distance_mask = self._calculate_distance(threshold, a2m, bb_pos, bb_error)

                for segmentation_file, frame_id in zip(filenames_segmentation[:n_frames], np.arange(n_frames)):
                    segmentation_dir =  os.path.join(geometry_files_dir, segmentation_file)

                    # Load segmentation for current frame
                    try:
                        segmentation = self._import_frame(segmentation_dir)
                    except:
                        continue

                    # Select BB that are present in current frame
                    bb_in_sample = np.unique(segmentation)
                    if bb_in_sample[0] == -1:
                        bb_in_sample = bb_in_sample[1:] # discard -1 label (pixels in segmentation map with unidentified BB)

                    # Construct adjacency matrix
                    adjacency_matrix = self._construct_adjacency_matrix(n_bb, bb_in_sample, distance_mask)

                    # Transform adjacency matrix to adjacency list in COO format
                    row, col = np.where(adjacency_matrix)
                    edge_index_np = np.array(list(zip(row, col)))
                    
                    # Filter out graphs that have 0 edges (either because no objects detected or objects far apart)
                    if edge_index_np.size > 0:

                        nodes_present = np.unique(edge_index_np) # refers to BB number, indexing starts at 0
                        new_idx = []
                        for node0, node1 in edge_index_np:
                            new_node0 = np.where(node0 == nodes_present)[0][0]
                            new_node1 = np.where(node1 == nodes_present)[0][0]
                            new_idx.append((new_node0, new_node1))
                        edge_index = torch.tensor(np.array(new_idx)).t()
                        
                        # Transform bb_labels to node features (tensor matrix with shape [num_nodes, num_node_features])
                        bb_labels_in_sample = np.array([bb_labels[i-1] for i in bb_in_sample]) # indexing starts at 0
                        x = torch.tensor(np.array(bb_labels_in_sample)) # indexing starts at 0

                        # Get relevant graph label
                        y_scene = self.y_labels[np.where(scene_name == self.scene_metadata[:,0])]

                        # Construct Data object
                        data = Data(edge_index=edge_index, x=x, y=y_scene)
                        torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

                        with open(csv_path, 'a', newline='') as csvfile_append:
                            writer = csv.writer(csvfile_append)
                            writer.writerow([f'data_{idx}.pt', scene_name, camera_name, frame_id])

                        idx += 1

    def _get_y(self):
        scene_labels = np.genfromtxt(os.path.join("evermotion_dataset", "analysis", "metadata_camera_trajectories.csv"), delimiter=',', dtype=None, encoding=None, autostrip=True)[1:,[0,7,8]]
        
        # Clean up first column (split scene and camera number)
        scene_names = [y[:-7] for y in scene_labels[:,0]]
        camera_names = [y[-6:] for y in scene_labels[:,0]]
        scene_metadata = np.c_[scene_names, camera_names, scene_labels[:,1:]]

        # Remove camera views annotated with 'OUTSIDE' in metadata file
        # (these have been removed form the online dataset, files no longer exist)
        remove = []
        for i, s in enumerate(scene_metadata[:,2]):
            if 'OUTSIDE' in s:
                remove.append(i)
        scene_metadata_clean = np.delete(scene_metadata, np.array(remove), axis=0)

        # Assign integer to each scene label
        scene_ids = np.zeros_like(scene_metadata_clean[:,0], dtype=int)
        for i, y_scene in enumerate(scene_metadata_clean[:,2]):
            y_id = np.where(np.unique(scene_metadata_clean[:,2]) == y_scene)
            scene_ids[i] = y_id[0]
        scene_metadata_clean = np.concatenate((scene_metadata_clean, scene_ids[:, np.newaxis]), axis=1)

        # Remove duplicate scene annotations (different cameras within the same scene still have same label)
        remove = []
        scene = None
        for i, row in enumerate(scene_metadata_clean):
            if row[0] == scene:
                remove.append(i)
            else:
                scene = row[0]
        scene_metadata_clean_nocam = np.delete(scene_metadata_clean, np.array(remove), axis=0)

        y = torch.tensor(scene_metadata_clean_nocam[:,4].astype(int))
        return y, scene_metadata_clean_nocam, scene_metadata_clean
    
    def _assign_labels(self, mesh_objects_sii, mesh_objects_si, bb_pos):
        bb_labels = []
        bb_error = []
        n_bb = bb_pos.shape[0] - 1

        for i in range(n_bb):
            lowlvl_instances_in_current_bb = np.where(mesh_objects_sii == i+1)[0]
            if lowlvl_instances_in_current_bb.size > 0:
                nyu_id = mesh_objects_si[lowlvl_instances_in_current_bb[0]]
                bb_labels.append(nyu_id)
            else:
                bb_error.append(i+1)
                bb_labels.append(np.array([-1], dtype=int64))
        
        return bb_labels, bb_error
    
    def _calculate_distance(self, threshold, a2m, bb_pos, bb_error):
        n_bb = bb_pos.shape[0] - 1
        bb_pos_m = bb_pos*a2m

        distances = np.zeros((n_bb, n_bb))
        distances[:, :] = np.nan

        for i in range(n_bb):
            for j in range(n_bb):
                if i+1 not in bb_error and j+1 not in bb_error:
                    distances[i, j] = distance.euclidean(bb_pos_m[i+1, :], bb_pos_m[j+1, :])

        distance_mask = (distances <= threshold)*int(1) # True for objects that are within [threshold] m of other objects
        np.fill_diagonal(distance_mask, 0)
        
        return distance_mask
    
    def _construct_adjacency_matrix(self, n_bb, bb_in_sample, distance_mask):    
        temp = np.arange(1, n_bb+1)
        bb_not_in_sample = np.delete(temp, bb_in_sample - np.ones(len(bb_in_sample), dtype=int))
        index = bb_not_in_sample - np.ones(len(bb_not_in_sample), dtype=int)
        
        adjacency_matrix = np.copy(distance_mask)
        adjacency_matrix[index,:] = np.zeros((len(bb_not_in_sample), n_bb))
        adjacency_matrix[:, index] = np.zeros((n_bb, len(bb_not_in_sample)))
        return adjacency_matrix

    def _import_scene(self, scene_name, scene_dir):
        detail_dir = os.path.join(scene_dir, "_detail")

        bb_pos_dir = os.path.join(detail_dir, "mesh", "metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5")

        mesh_objects_si_dir = os.path.join("evermotion_dataset", "scenes", scene_name, "_detail", "mesh", "mesh_objects_si.hdf5")
        mesh_objects_sii_dir = os.path.join("evermotion_dataset", "scenes", scene_name, "_detail", "mesh", "mesh_objects_sii.hdf5")
        metadata_objects_dir = os.path.join("evermotion_dataset", "scenes", scene_name, "_detail", "mesh", "metadata_objects.csv")
        a2m_dir = os.path.join(detail_dir, "metadata_scene.csv")

        with h5py.File(bb_pos_dir, "r") as f: bb_pos = f['dataset'][:]

        with h5py.File(mesh_objects_si_dir, "r") as f: mesh_objects_si = f['dataset'][:]
        with h5py.File(mesh_objects_sii_dir, "r") as f: mesh_objects_sii = f['dataset'][:]
        metadata_objects = np.genfromtxt(metadata_objects_dir, delimiter=None, dtype=str)

        with open(a2m_dir, newline='') as csvfile:
            metadata_scene = list(csv.reader(csvfile))
            a2m = float(metadata_scene[1][1])
        
        return bb_pos, mesh_objects_si, mesh_objects_sii, metadata_objects, a2m

    def _import_camera(self, scene_dir, camera_name):
        detail_dir = os.path.join(scene_dir, "_detail")
        camera_pos_dir = os.path.join(detail_dir, camera_name, "camera_keyframe_positions.hdf5")
        camera_rot_dir = os.path.join(detail_dir, camera_name, "camera_keyframe_orientations.hdf5")
        with h5py.File(camera_pos_dir, "r") as f: camera_pos_all = f['dataset'][:]
        with h5py.File(camera_rot_dir, "r") as f: camera_rot_all = f['dataset'][:]
        return camera_pos_all, camera_rot_all
    
    def _import_frame(self, segmentation_dir):
        with h5py.File(segmentation_dir, "r") as f: segmentation = f['dataset'][:]
        return segmentation

    def _is_valid(self, filename):
        data = torch.load(os.path.join(self.processed_dir, filename))
        y = data.y.item()
        return y in self.classes
    
    def _update_label(self, data):
        old_label = data.y.item()
        new_label = self.class_mapping[old_label]
        data.y = torch.tensor([new_label], dtype=torch.long)
    
    def _remove_weird(self):
        pass
    
    def len(self):
        if self.classes:
            length = len(self.graphs)
        else:
            length = 61936
        return length
    
    def get(self, idx):
        if self.classes:
            filename = self.graphs[idx]
            data = torch.load(os.path.join(self.processed_dir, filename))
            self._update_label(data)
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

if __name__ == "__main__":
    dataset = HypersimDataset('./')
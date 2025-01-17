import os
import numpy as np
from .generic import SceneFlowDataset

import pickle
import io
from functools import partial

def remove_out_of_bounds_points(pc, y, x_min, x_max, y_min, y_max, z_min, z_max):
    # Max needs to be exclusive because the last grid cell on each axis contains
    # [((grid_size - 1) * cell_size) + *_min, *_max).
    #   E.g grid_size=512, cell_size = 170/512 with min=-85 and max=85
    # For z-axis this is not necessary, but we do it for consistency
    mask = (pc[:, 0] >= x_min) & (pc[:, 0] < x_max) \
           & (pc[:, 1] >= y_min) & (pc[:, 1] < y_max) \
           & (pc[:, 2] >= z_min) & (pc[:, 2] < z_max)
    pc_valid = pc[mask]
    y_valid = None
    if y is not None:
        y_valid = y[mask]
    return pc_valid, y_valid

class WaymoDataset(SceneFlowDataset):
    def __init__(self, root_dir, nb_points=None, mode=None, compensate_ego_motion=False,):
        """
        Construct the KITTI scene flow datatset as in:
        Gu, X., Wang, Y., Wu, C., Lee, Y.J., Wang, P., HPLFlowNet: Hierarchical
        Permutohedral Lattice FlowNet for scene ﬂow estimation on large-scale 
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition 
        (CVPR). pp. 3254–3263 (2019) 

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.

        """

        super(WaymoDataset, self).__init__(nb_points)
        
        # It has information regarding the files and transformations
        print("[Dataset]Use waymo dataset, where compensate_ego_motion={}".format(compensate_ego_motion))
        self.compensate_ego_motion_to_pcl = compensate_ego_motion
        self.data_path = root_dir
        self.ph = None
        self.mode = mode
        if self.mode is None:
            self.mode = "train"
        if self.mode == "train":
            self.data_path = os.path.join(root_dir, 'train')
            metadata_path = os.path.join(root_dir, 'train', 'metadata')
        elif self.mode == "val":
            self.data_path = os.path.join(root_dir, 'valid')
            metadata_path = os.path.join(root_dir, 'valid', 'metadata')

        
        if self.data_path.startswith("s3://"):
            from petrel_helper import PetrelHelper
            self.ph = PetrelHelper()
            
            
        try:
            if metadata_path.startswith("s3://"):
                metadata_file = self.ph.open(metadata_path, 'rb')
                self.metadata = pickle.load(metadata_file)
            else:
                with open(metadata_path, 'rb') as metadata_file:
                    self.metadata = pickle.load(metadata_file)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found, please create it by running preprocess.py")
        

        # self._pillarization_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
        #                                                    y_min=y_min, z_min=z_min, z_max=z_max,
        #                                                    n_pillars_x=n_pillars_x)

        # This returns a function that removes points that should not be included in the pillarization.
        # It also removes the labels if given.
        self._drop_invalid_point_function = self.drop_points_function(x_min=-35,
                                                          x_max=35, y_min=-35, y_max=35,
                                                          z_min=0.3, z_max=3)

    def __len__(self):
        return len(self.metadata['look_up_table'])



    # def make_dataset(self):
    #     """
    #     Find and filter out paths to all examples in the dataset. 
        
    #     """

    #     #
    #     root = os.path.realpath(os.path.expanduser(self.root_dir))
    #     all_paths = sorted(os.walk(root))
    #     useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
    #     assert len(useful_paths) == 200, "Problem with size of kitti dataset"

    #     # Mapping / Filtering of scans as in HPLFlowNet code
    #     mapping_path = os.path.join(os.path.dirname(__file__), "KITTI_mapping.txt")
    #     with open(mapping_path) as fd:
    #         lines = fd.readlines()
    #         lines = [line.strip() for line in lines]
    #     useful_paths = [
    #         path for path in useful_paths if lines[int(os.path.split(path)[-1])] != ""
    #     ]

    #     return useful_paths
    
    def read_point_cloud_pair(self, index):
        """
        Read from disk the current and previous point cloud given an index
        """
        # In the lookup table entries with (current_frame, previous_frame) are stored
        #print(self.metadata['look_up_table'][index][0][0])
        data_path = os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0])
        data_str = None
        if data_path.startswith("s3://"):
            # np load need the file-like object supports the seek operation
            data_str = io.BytesIO(self.ph.open(data_path, 'rb').read())
        else:
            data_str = open(data_path, 'rb')
        
        current_frame = np.load(data_str)['frame']
        
        
        data_path = os.path.join(self.data_path, self.metadata['look_up_table'][index][1][0])
        data_str = None
        if data_path.startswith("s3://"):
            data_str = io.BytesIO(self.ph.open(data_path, 'rb').read())
        else:
            data_str = open(data_path, 'rb')
        previous_frame = np.load(data_str)['frame']
        return current_frame, previous_frame

    def get_pose_transform(self, index):
        """
        Return the frame poses of the current and previous point clouds given an index
        """
        current_frame_pose = self.metadata['look_up_table'][index][0][1]
        previous_frame_pose = self.metadata['look_up_table'][index][1][1]
        return current_frame_pose, previous_frame_pose

    def get_flows(self, frame):
        """
        Return the flows given a point cloud
        """
        flows = frame[:, -4:]
        return flows

    # def subsample_points(self, current_frame, previous_frame, flows):
    #     # current_frame.shape[0] == flows.shape[0]
    #     if current_frame.shape[0] > self.nb_points:
    #         indexes_current_frame = np.linspace(0, current_frame.shape[0]-1, num=self.nb_points).astype(int)
    #         current_frame = current_frame[indexes_current_frame, :]
    #         flows = flows[indexes_current_frame, :]
    #     if previous_frame.shape[0] > self.nb_points:
    #         indexes_previous_frame = np.linspace(0, previous_frame.shape[0]-1, num=self.nb_points).astype(int)
    #         previous_frame = previous_frame[indexes_previous_frame, :]
    #     return current_frame, previous_frame, flows
    
    
    @staticmethod
    def get_coordinates_and_features(point_cloud, transform=None):
        """
        Parse a point clound into coordinates and features.
        :param point_cloud: Full [N, 9] point cloud
        :param transform: Optional parameter. Transformation matrix to apply
        to the coordinates of the point cloud
        :return: [N, 5] where N is the number of points and 5 is [x, y, z, intensity, elongation]
        """
        points_coord, features, flows = point_cloud[:, 0:3], point_cloud[:, 3:5], point_cloud[:, 5:]
        if transform is not None:
            ones = np.ones((points_coord.shape[0], 1))
            points_coord = np.hstack((points_coord, ones))
            points_coord = transform @ points_coord.T
            points_coord = points_coord[0:-1, :]
            points_coord = points_coord.T
        point_cloud = np.hstack((points_coord, features))
        return point_cloud
    
    @staticmethod
    def drop_points_function(x_min, x_max, y_min, y_max, z_min, z_max):
        inner = partial(remove_out_of_bounds_points,
                                            x_min=x_min,
                                            y_min=y_min,
                                            z_min=z_min,
                                            z_max=z_max,
                                            x_max=x_max,
                                            y_max=y_max
                                            )

        return inner

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        # # Load data
        # sequence = [np.load(os.path.join(self.paths[idx], "pc1.npy"))]
        # sequence.append(np.load(os.path.join(self.paths[idx], "pc2.npy")))

        # # Remove ground points
        # is_ground = np.logical_and(sequence[0][:, 1] < -1.4, sequence[1][:, 1] < -1.4)
        # not_ground = np.logical_not(is_ground)
        # sequence = [sequence[i][not_ground] for i in range(2)]

        # # Remove points further than 35 meter away as in HPLFlowNet code
        # is_close = np.logical_and(sequence[0][:, 2] < 35, sequence[1][:, 2] < 35)
        # sequence = [sequence[i][is_close] for i in range(2)]

        # # Scene flow
        # ground_truth = [
        #     np.ones_like(sequence[0][:, 0:1]),
        #     sequence[1] - sequence[0],
        # ]  # [Occlusion mask, scene flow]

        # return sequence, ground_truth
        
        
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, F], being N the number of points and the
        F to the number of features, which is [x, y, z, intensity, elongation]
        """
        
        current_frame, previous_frame = self.read_point_cloud_pair(idx)
        current_frame_pose, previous_frame_pose = self.get_pose_transform(idx)
        flows = self.get_flows(current_frame)

        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:
            current_frame, flows = self._drop_invalid_point_function(current_frame, flows)
            previous_frame, _ = self._drop_invalid_point_function(previous_frame, None)
    
        # if self.nb_points is not None:
        #     current_frame, previous_frame, flows = self.subsample_points(current_frame, previous_frame, flows)

        # G_T_C -> Global_TransformMatrix_Current
        G_T_C = np.reshape(np.array(current_frame_pose), [4, 4])

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame_pose), [4, 4])
        C_T_P = np.linalg.inv(G_T_C) @ G_T_P
        # https://github.com/waymo-research/waymo-open-dataset/blob/bbcd77fc503622a292f0928bfa455f190ca5946e/waymo_open_dataset/utils/box_utils.py#L179
        
        # compensate ego
        if self.compensate_ego_motion_to_pcl:
            previous_frame = self.get_coordinates_and_features(previous_frame, transform=C_T_P)
        # do not compensate ego
        else:
            C_T_P_inv = np.linalg.inv(C_T_P)
            previous_frame = self.get_coordinates_and_features(previous_frame, transform=None)
            # retrieve the initial flow
            # flow is the velocity, frame rate = 10hz
            frm_time_interval = 0.10 # unit: s
            flows[:, :3] = (current_frame[:, :3] - self.get_coordinates_and_features(current_frame[:, :3] - flows[:, :3] * frm_time_interval, transform=C_T_P_inv)) / frm_time_interval
            # note : if u decide not to compensate the flow as the code above, the flow information saving in the metadata is not valid.
            # because in this repo, there is no effect to the the training procedure, we 【do not】 update the data saved in the metadata. 
        current_frame = self.get_coordinates_and_features(current_frame, transform=None)



        # Perform the pillarization of the point_cloud
        # if self._point_cloud_transform is not None and self._apply_pillarization:
        #     current_frame = self._point_cloud_transform(current_frame)
        #     previous_frame = self._point_cloud_transform(previous_frame)
        # else:
            # output must be a tuple
        # previous_frame = (previous_frame, None)
        # current_frame = (current_frame, None)
        # This returns a tuple of augmented pointcloud and grid indices
        previous_frame = previous_frame[:, :3]
        current_frame = current_frame[:, :3]
        # print(previous_frame.shape, current_frame.shape, flows.shape)
        
        # initial flow： current = previous + flow
        # now we swap the current/previous frm for the definition of point num. so that the flow should be inversed
        flows = -flows[:, :3]
        # mask = flows.sum(axis=1)
        # mask[mask > 0] = 1
        # mask = mask.reshape(-1, 1)
        mask = np.ones_like(current_frame[:, 0:1])
        return [current_frame, previous_frame], [mask, flows]
    
    
    def subsample_points(self, sequence, ground_truth):
        """
        Subsample point clouds randomly.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x N x 3 and pc2 has size 1 x M x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x N x 1 and pc1 has size 
            1 x N x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3. The n 
            points are chosen randomly among the N available ones. The m points
            are chosen randomly among the M available ones. 
            If N, M >= 
            self.nb_point then n, m = self.nb_points. If N, M < 
            self.nb_point then n, m = self.nb_points. 
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size 
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        """
        if sequence[0].shape[0] >= self.nb_points:
            # Choose points in first scan
            ind1 = np.random.permutation(sequence[0].shape[0])[: self.nb_points]
            sequence[0] = sequence[0][ind1]
            ground_truth = [g[ind1] for g in ground_truth]
        else:
            n1 = sequence[0].shape[0]
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.nb_points - n1, replace=True)),
                                         axis=-1)
            sequence[0] = sequence[0][sample_idx1]
            ground_truth = [g[sample_idx1] for g in ground_truth]
            
        if sequence[1].shape[0] >= self.nb_points:
            # Choose point in second scan
            ind2 = np.random.permutation(sequence[1].shape[0])[: self.nb_points]
            sequence[1] = sequence[1][ind2]
        else:
            n2 = sequence[1].shape[0]
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.nb_points - n2, replace=True)),
                                         axis=-1)
            sequence[1] = sequence[1][sample_idx2]

        return sequence, ground_truth

if __name__ == '__main__':
    dataset = WaymoDataset(root_dir="/home/xlju/data/waymo_sf_processed", nb_points=8192)
    sample = dataset[141773]
    pass

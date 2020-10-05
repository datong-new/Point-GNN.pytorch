import numpy as np
import torch
from models import preprocess
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn,\
    get_encoding_len
from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from multiprocessing import Pool, Queue, Process
import os
import argparse
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config

def fetch_data(dataset, frame_idx, train_config, config):
    aug_fn = preprocess.get_data_aug(train_config['data_aug_configs'])
    BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
    box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
    box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])
    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx,
        config['downsample_by_voxel_size'])

    box_label_list = dataset.get_label(frame_idx)
    if 'crop_aug' in train_config:
        cam_rgb_points, box_label_list = sampler.crop_aug(cam_rgb_points,
            box_label_list,
            sample_rate=train_config['crop_aug']['sample_rate'],
            parser_kwargs=train_config['crop_aug']['parser_kwargs'])

    cam_rgb_points, box_label_list = aug_fn(cam_rgb_points, box_label_list)

    (vertex_coord_list, keypoint_indices_list, edges_list) = \
        graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])
    if config['input_features'] == 'irgb':
        input_v = cam_rgb_points.attr
    elif config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
            cam_rgb_points.attr[:, 1:]])
    elif config['input_features'] == '0000':
        input_v = np.zeros_like(cam_rgb_points.attr)
    elif config['input_features'] == 'i000':
        input_v = np.hstack([cam_rgb_points.attr[:, [0]],
            np.zeros((cam_rgb_points.attr.shape[0], 3))])
    elif config['input_features'] == 'i':
        input_v = cam_rgb_points.attr[:, [0]]
    elif config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
    last_layer_graph_level = config['model_kwargs'][
        'layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
    if config['label_method'] == 'yaw':
        cls_labels, boxes_3d, valid_boxes, label_map = \
            dataset.assign_classaware_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    if config['label_method'] == 'Car':
        cls_labels, boxes_3d, valid_boxes, label_map = \
            dataset.assign_classaware_car_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    if config['label_method'] == 'Pedestrian_and_Cyclist':
        (cls_labels, boxes_3d, valid_boxes, label_map) =\
            dataset.assign_classaware_ped_and_cyc_label_to_points(
            box_label_list, last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    encoded_boxes = box_encoding_fn(cls_labels, last_layer_points_xyz,
        boxes_3d, label_map)
    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, encoded_boxes, valid_boxes)


def batch_data(batch_list):
    N_input_v, N_vertex_coord_list, N_keypoint_indices_list, N_edges_list,\
    N_cls_labels, N_encoded_boxes, N_valid_boxes = zip(*batch_list)
    batch_size = len(batch_list)
    level_num = len(N_vertex_coord_list[0])
    batched_keypoint_indices_list = []
    batched_edges_list = []
    for level_idx in range(level_num-1):
        centers = []
        vertices = []
        point_counter = 0
        center_counter = 0
        for batch_idx in range(batch_size):
            centers.append(
                N_keypoint_indices_list[batch_idx][level_idx]+point_counter)
            vertices.append(np.hstack(
                [N_edges_list[batch_idx][level_idx][:,[0]]+point_counter,
                 N_edges_list[batch_idx][level_idx][:,[1]]+center_counter]))
            point_counter += N_vertex_coord_list[batch_idx][level_idx].shape[0]
            center_counter += \
                N_keypoint_indices_list[batch_idx][level_idx].shape[0]
        batched_keypoint_indices_list.append(np.vstack(centers))
        batched_edges_list.append(np.vstack(vertices))
    batched_vertex_coord_list = []
    for level_idx in range(level_num):
        points = []
        counter = 0
        for batch_idx in range(batch_size):
            points.append(N_vertex_coord_list[batch_idx][level_idx])
        batched_vertex_coord_list.append(np.vstack(points))
    batched_input_v = np.vstack(N_input_v)
    batched_cls_labels = np.vstack(N_cls_labels)
    batched_encoded_boxes = np.vstack(N_encoded_boxes)
    batched_valid_boxes = np.vstack(N_valid_boxes)

    batched_input_v = torch.from_numpy(batched_input_v)
    batched_vertex_coord_list = [torch.from_numpy(item) for item in batched_vertex_coord_list]
    batched_keypoint_indices_list = [torch.from_numpy(item).long() for item in batched_keypoint_indices_list]
    batched_edges_list = [torch.from_numpy(item).long() for item in batched_edges_list]
    batched_cls_labels = torch.from_numpy(batched_cls_labels)
    batched_encoded_boxes = torch.from_numpy(batched_encoded_boxes)
    batched_valid_boxes = torch.from_numpy(batched_valid_boxes)

    return (batched_input_v, batched_vertex_coord_list,
        batched_keypoint_indices_list, batched_edges_list, batched_cls_labels,
        batched_encoded_boxes, batched_valid_boxes)

class DataProvider(object):
    """This class provides input data to training.
    It has option to load dataset in memory so that preprocessing does not
    repeat every time.
    Note, if there is randomness inside graph creation, dataset should be
    reloaded.
    """
    def __init__(self, dataset, train_config, config, async_load_rate=1.0, result_pool_limit=10000):
        if 'NUM_TEST_SAMPLE' not in train_config:
            self.NUM_TEST_SAMPLE = dataset.num_files
        else:
            if train_config['NUM_TEST_SAMPLE'] < 0:
                self.NUM_TEST_SAMPLE = dataset.num_files
            else:
                self.NUM_TEST_SAMPLE = train_config['NUM_TEST_SAMPLE']
        load_dataset_to_mem=train_config['load_dataset_to_mem']
        load_dataset_every_N_time=train_config['load_dataset_every_N_time']
        capacity=train_config['capacity']
        num_workers=train_config['num_load_dataset_workers']
        preload_list=list(range(self.NUM_TEST_SAMPLE))

        self.dataset = dataset
        self.train_config = train_config
        self.config = config
        self._fetch_data = fetch_data
        self._batch_data = batch_data
        self._buffer = {}
        self._results = {}
        self._load_dataset_to_mem = load_dataset_to_mem
        self._load_every_N_time = load_dataset_every_N_time
        self._capacity = capacity
        self._worker_pool = Pool(processes=num_workers)
        self._preload_list = preload_list
        self._async_load_rate = async_load_rate
        self._result_pool_limit = result_pool_limit
        #if len(self._preload_list) > 0:
        #    self.preload(self._preload_list)

    def preload(self, frame_idx_list):
        """async load dataset into memory."""
        for frame_idx in frame_idx_list:
            result = self._worker_pool.apply_async(
                self._fetch_data, (self.dataset, frame_idx, self.train_config, self.config))
            self._results[frame_idx] = result

    def async_load(self, frame_idx):
        """async load a data into memory"""
        if frame_idx in self._results:
            data = self._results[frame_idx].get()
            del self._results[frame_idx]
        else:
            data = self._fetch_data(self.dataset, frame_idx, self.train_config, self.config)
        if np.random.random() < self._async_load_rate:
            if len(self._results) < self._result_pool_limit:
                result = self._worker_pool.apply_async(
                    self._fetch_data, (self.dataset, frame_idx, self.train_config, self.config))
                self._results[frame_idx] = result
        return data

    def provide(self, frame_idx):
        if self._load_dataset_to_mem:
            if self._load_every_N_time >= 1:
                extend_frame_idx = frame_idx+np.random.choice(
                    self._capacity)*self.NUM_TEST_SAMPLE
                if extend_frame_idx not in self._buffer:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                if ctr == self._load_every_N_time:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                self._buffer[extend_frame_idx] = (data, ctr+1)
                return data
            else:
                # do not buffer
                return self.async_load(frame_idx)
        else:
            return self._fetch_data(self.dataset, frame_idx, self.train_config, self.config)

    def provide_batch(self, frame_idx_list):
        batch_list = []
        for frame_idx in frame_idx_list:
            batch_list.append(self.provide(frame_idx))
        return self._batch_data(batch_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of PointGNN')
    parser.add_argument('train_config_path', type=str,
                       help='Path to train_config')
    parser.add_argument('config_path', type=str,
                       help='Path to config')
    parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                       help='Path to KITTI dataset. Default="../dataset/kitti/"')
    parser.add_argument('--dataset_split_file', type=str,
                        default='',
                       help='Path to KITTI dataset split file.'
                       'Default="DATASET_ROOT_DIR/3DOP_splits'
                       '/train_config["train_dataset"]"')
    
    args = parser.parse_args()
    train_config = load_train_config(args.train_config_path)
    DATASET_DIR = args.dataset_root_dir
    config_complete = load_config(args.config_path)
    if 'train' in config_complete:
        config = config_complete['train']
    else:
        config = config_complete

    if args.dataset_split_file == '':
        DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
            './3DOP_splits/'+train_config['train_dataset'])
    else:
        DATASET_SPLIT_FILE = args.dataset_split_file

    # input function ==============================================================
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])

    data_provider = DataProvider(dataset, train_config, config)

    input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
            cls_labels, encoded_boxes, valid_boxes = data_provider.provide_batch([1545, 1546])


    #batch_list = []
    #batch_list += [fetch_data(dataset, 1545, train_config, config)]
    #batch_list += [fetch_data(dataset, 1546, train_config, config)]
    #input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
    #        cls_labels, encoded_boxes, valid_boxes = batch_data(batch_list)


    print(f"input_v: {input_v.shape}")
    for i, vertex_coord in enumerate(vertex_coord_list):
        print(f"vertex_coord: {i}: {vertex_coord.shape}")

    for i, indices in enumerate(keypoint_indices_list):
        print(f"indices: {i}: {indices.shape}")
        print(indices)
    for i, edge in enumerate(edges_list):
        print(f"edge: {i}: {edge.shape}")
        print(edge)
        #for item in edge:
        #    if item[0]==item[1]: print(item)
    print(f"cls_labels:{cls_labels.shape}")
    print(f"encoded_boxes: {encoded_boxes.shape}")
    print(f"valid_boxes: {valid_boxes.shape}")
    print(valid_boxes)
    print(f"max: {valid_boxes.max()}, min:{valid_boxes.min()}, sum: {valid_boxes.sum()}")








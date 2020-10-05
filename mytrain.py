import torch
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, get_encoding_len
import os
from dataset.kitti_dataset import KittiDataset
from kitty_dataset import DataProvider
from model import *

import argparse



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
    #input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
    #        cls_labels, encoded_boxes, valid_boxes = data_provider.provide_batch([1545, 1546])

    batch = data_provider.provide_batch([1545, 1546])
    input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
            cls_labels, encoded_boxes, valid_boxes = batch


    NUM_CLASSES = dataset.num_classes
    BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])

    model = MultiLayerFastLocalGraphModelV2(num_classes=NUM_CLASSES,
                box_encoding_len=BOX_ENCODING_LEN, mode='train',
                **config['model_kwargs'])

    logits, box_encoding = model(batch, is_training=True)
    loss_dict = model.loss(logits, cls_labels, box_encoding, encoded_boxes, valid_boxes)



import torch
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, get_encoding_len
import os
from dataset.kitti_dataset import KittiDataset
from kitty_dataset import DataProvider
from model import *
import numpy as np
import argparse
from util.metrics import recall_precisions, mAP
from tqdm import trange
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of PointGNN')
    parser.add_argument('train_config_path', type=str,
                       help='Path to train_config')
    parser.add_argument('config_path', type=str,
                       help='Path to config')
    parser.add_argument('--device', type=str, default='cuda:0',
            help="Device for training, cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1,
            help='Batch size')
    parser.add_argument('--epoches', type=int, default=100,
            help='Training epoches')
    parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                       help='Path to KITTI dataset. Default="../dataset/kitti/"')
    parser.add_argument('--dataset_split_file', type=str,
                        default='',
                       help='Path to KITTI dataset split file.'
                       'Default="DATASET_ROOT_DIR/3DOP_splits'
                       '/train_config["train_dataset"]"')
    
    args = parser.parse_args()
    epoches = args.epoches
    batch_size = args.batch_size
    device = args.device
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
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    NUM_TEST_SAMPLE = dataset.num_files

    os.system("mkdir saved_models")


    for epoch in range(1, epoches):
        recalls_list, precisions_list, mAP_list = {}, {}, {}
        for i in range(NUM_CLASSES): recalls_list[i], precisions_list[i], mAP_list[i] = [], [], []

        frame_idx_list = np.random.permutation(NUM_TEST_SAMPLE)

        pbar = tqdm(list(range(0, NUM_TEST_SAMPLE-batch_size+1, batch_size)), desc="start training", leave=True)

        for batch_idx in pbar:
        #for batch_idx in range(0, NUM_TEST_SAMPLE-batch_size+1, batch_size):
            batch_frame_idx_list = frame_idx_list[batch_idx: batch_idx+batch_size]
            batch = data_provider.provide_batch(batch_frame_idx_list)
            input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
                    cls_labels, encoded_boxes, valid_boxes = batch

            new_batch = []
            for item in batch:
                if not isinstance(item, torch.Tensor):
                    item = [x.to(device) for x in item]
                else: item = item.to(device)
                new_batch += [item]
            batch = new_batch
            input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
                    cls_labels, encoded_boxes, valid_boxes = batch

            logits, box_encoding = model(batch, is_training=True)
            predictions = torch.argmax(logits, dim=1)

            loss_dict = model.loss(logits, cls_labels, box_encoding, encoded_boxes, valid_boxes)
            t_cls_loss, t_loc_loss, t_reg_loss = loss_dict['cls_loss'], loss_dict['loc_loss'], loss_dict['reg_loss']
            pbar.set_description(f"{epoch}, t_cls_loss: {t_cls_loss}, t_loc_loss: {t_loc_loss}, t_reg_loss: {t_reg_loss}")
            t_total_loss = t_cls_loss + t_loc_loss + t_reg_loss
            optimizer.zero_grad()
            t_total_loss.backward()
            optimizer.step()

            # record metrics
            recalls, precisions = recall_precisions(cls_labels, predictions, NUM_CLASSES)
            #mAPs = mAP(cls_labels, logits, NUM_CLASSES)
            mAPs = mAP(cls_labels, logits.sigmoid(), NUM_CLASSES)
            for i in range(NUM_CLASSES):
                recalls_list[i] += [recalls[i]]
                precisions_list[i] += [precisions[i]]
                mAP_list[i] += [mAPs[i]]

        # print metrics
        for class_idx in range(NUM_CLASSES):
            print(f"class_idx:{class_idx}, recall: {np.mean(recalls_list[class_idx])}, precision: {np.mean(precisions_list[class_idx])}, mAP: {np.mean(mAP_list[class_idx])}")

        # save model
        torch.save(model.state_dict(), "saved_models/model_{}.pt".format(epoch))







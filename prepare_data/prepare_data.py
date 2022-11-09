import os
import sys
import copy
import random
import pickle
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

sys.path.append('/home/warrenzhao/PointNet-AE-3DMM')
from utils.read_object import *

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='shapenet', help='dataset to use (only support shapenet now) [default: shapenet]')
parser.add_argument('--data_class', default='car', help='the class of dataset to use [default: shapenet]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--var_range', type=float, default=1.0, help='Range of Variance Values [default: 1.0]')
FLAGS = parser.parse_args()

DATASET = FLAGS.dataset
DATA_CLASS = cate_to_synsetid[FLAGS.data_class]
NUM_POINT = FLAGS.num_point
VAR_RANGE = FLAGS.var_range

def read_mesh(filename):
    return o3d.io.read_triangle_mesh(filename)

def read_pointcloud(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    return pcd

def prepare_point_cloud(filename, var_range=VAR_RANGE, target_size=NUM_POINT):
    """
    preprocessing point cloud
    @param filename: mesh file name
    @param var_range: float
    @param target_size: int
    @return: downsampled np array of shape (3, target_szie)
    """
    # downsample to target size
    pcd = o3d.io.read_point_cloud(filename, format='xyz')
    X = np.asarray(pcd.points)

    idx_lst = random.sample(range(X.shape[0]), target_size)
    X = X[idx_lst]
    
    all_scale = np.array([np.var(X[:, 0]), np.var(X[:, 1]), np.var(X[:, 2])])
    max_scale = np.max(all_scale)
    s = var_range/max_scale

    centroid = np.mean(X, axis=0)
    X = X - centroid
    X = X * s
    return X.T

def preprocessAll(filenames):
    """
    filenames: list of filenames (output of listFileNames)
    return list of scaled and downsampled data
    """
    
    pcd_all = []
    for filename in tqdm(filenames):
        pcd = prepare_point_cloud(filename)
        pcd_all.append(pcd)

    return pcd_all

def listFileNames(folder):
    """Walk through every files in a directory"""
    filenames = []
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            filenames.append(os.path.abspath(os.path.join(dirpath, filename)))

    return filenames

def preprocess_and_save(folder, save_path):
    """
    Unify object sizes and orientation to the reference object
    @param    folder: (in form of (data)/train/... and (data)/test/...)
    @param    ref_name: name of the reference object
    @param    save_path: preprocessed data will be saved here as 'train.txt' and 'test.txt'
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    filenames = listFileNames(folder)

    data_all = preprocessAll(filenames)

    train_idx = random.sample(range(len(data_all)), (int)(len(data_all) * 9 / 10))
    test_idx = [x for x in range(len(data_all)) if x not in train_idx]

    train_all = np.asarray(data_all)[train_idx]
    test_all = np.asarray(data_all)[test_idx]

    train_save_filename = open(os.path.join(save_path, 'train.txt'), 'wb')
    pickle.dump(train_all, train_save_filename)

    test_save_filename = open(os.path.join(save_path, 'test.txt'), 'wb')
    pickle.dump(test_all, test_save_filename)

if __name__ == "__main__":
    # download dataset if dataset does not exist
    if not os.path.exists(os.path.join("../data/raw", DATASET)):
        os.makedirs("../data/raw")
        os.system("bash ./download_" + DATASET + ".sh")

    data_folder = os.path.join("../data/raw", DATASET, DATA_CLASS, 'points/')
    save_folder = os.path.join("../data/preprocessed", DATASET, FLAGS.data_class)

    preprocess_and_save(folder = data_folder, save_path = save_folder)
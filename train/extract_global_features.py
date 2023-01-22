import argparse
import os, sys
import gitpath
HOME_PATH = gitpath.root()
sys.path.append(HOME_PATH)

import pickle

import numpy as np

# Import pytorch dependencies
import torch
from torch.utils.data import DataLoader

# Import toolkits
from utils.visualization_3D_objects import *

from nn.pointnetae import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='shapenet', help='dataset to use [default: shapenet]')
parser.add_argument('--model_name', default='PointNetAE', help='Model name [default: PointNetAE]')
parser.add_argument('--category', default='car', help='sub-category in dataset to use [default: car]')
parser.add_argument('--checkpoint_path', default=os.path.join(HOME_PATH, 'saved_nn'), help='Path to saved models [default: repo_root/saved_nn]')
FLAGS = parser.parse_args()

DATASET = FLAGS.dataset
MODEL_NAME = FLAGS.model_name +'_' + FLAGS.category
DATA_PATH = os.path.join(HOME_PATH, 'data/preprocessed', DATASET, FLAGS.category)
CHECKPOINT_PATH = FLAGS.checkpoint_path
# SAVE_PATH = os.path.join(HOME_PATH, 'evaluate_model', DATASET, FLAGS.category)
SAVE_PATH = os.path.join(HOME_PATH, 'evaluate_model', DATASET, 'test')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#load all preprocessed data
f1 = open(os.path.join(DATA_PATH,'train.txt'),'rb')
X_train = pickle.load(f1)
f2 = open(os.path.join(DATA_PATH,'test.txt'),'rb')
X_test = pickle.load(f2)

BATCH_SIZE = 16

# construct dataloader
full_loader = DataLoader(
    np.vstack([X_train, X_test]), 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    drop_last=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
model_test = PointNet_AE(3, 2048)
state_dict = torch.load(os.path.join(CHECKPOINT_PATH, MODEL_NAME + '.pth')) # change the path to your own checkpoint file
model_test.cuda()
model_test.load_state_dict(state_dict['state_dict'])
model_test.eval()

global_feat_lst = []
for batch_idx, (inputs) in enumerate(full_loader):
    # copy inputs to device
    inputs = inputs.float().to(device)
    # compute the output and loss
    outputs, global_feat_i = model_test(inputs)
    global_feat_lst.append(global_feat_i.cpu().detach().numpy())
        
global_feat = np.vstack(global_feat_lst)
np.savetxt(os.path.join(SAVE_PATH, "global_feat.csv"), global_feat, delimiter=",")
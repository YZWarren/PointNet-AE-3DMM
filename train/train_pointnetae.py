import argparse
import os, sys
sys.path.append('../')

import time
import copy
import math
import pickle
import statistics
from matplotlib.image import imread

import numpy as np
import pandas as pd
import open3d as o3d
import torchvision
import pytorch3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _single, _pair, _triple
from torchsummary import summary

from pytorch3d.loss import chamfer_distance

# Import toolkits
from utils.visualization_3D_objects import *

from nn.model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='shapenet', help='dataset to use [default: shapenet]')
parser.add_argument('--model_name', default='PointNetAE', help='Model name [default: PointNetAE]')
parser.add_argument('--category', default='car', help='Which single class to train on [default: car]')
parser.add_argument('--checkpoint_path', default='../saved_nn', help='Path to save model checkpoint [default: ../saved_nn]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--log_epoch', type=int, default=10, help='Epoch to log results [default: 10]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate [default: 1e-5]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=30, help='Decay step for lr decay [default: 30]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--resume', choices=['yes', 'no'], default='no', help='Resume previous training (yes/no) [default: no]')
FLAGS = parser.parse_args()

DATASET = FLAGS.dataset
MODEL_NAME = FLAGS.model_name
DATA_PATH = os.path.join("../data/preprocessed", DATASET, FLAGS.category)
CHECKPOINT_PATH = FLAGS.checkpoint_path
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch_size
INITIAL_LR = FLAGS.initial_lr
MOMENTUM = FLAGS.momentum
EPOCHS = FLAGS.max_epoch
LOG_EPOCHS = FLAGS.log_epoch
DECAY_STEPS = FLAGS.decay_step
DECAY_EPOCHS = np.arange(DECAY_STEPS, DECAY_STEPS * (np.round(EPOCHS/DECAY_STEPS)), DECAY_STEPS)
DECAY_RATE = FLAGS.decay_rate

RESUME = FLAGS.resume == 'yes'

#load all aligned cars
f1 = open(os.path.join(DATA_PATH,'train.txt'),'rb')
X_train = pickle.load(f1)
f2 = open(os.path.join(DATA_PATH,'test.txt'),'rb')
X_test = pickle.load(f2)

# construct dataloader
train_loader = DataLoader(
    X_train, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4
)
val_loader = DataLoader(
    X_test, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4
)

# GPU check                
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Run on GPU...")
else:
    print("Run on CPU...")

# Model Definition  
model = PointNet_AE(3, NUM_POINT)
model = model.to(device)

# Check if on GPU
assert(next(model.parameters()).is_cuda)

# create loss function: Chamfer distance
criterion = lambda recon_x, x: chamfer_distance(recon_x, x)

# Add optimizer
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCHS, gamma=DECAY_RATE, verbose=True)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/' + MODEL_NAME)

best_loss = 1e20
START_EPOCH = 0

# store loss learning curve
train_loss_lst = []
valid_loss_lst = []

if RESUME:
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, MODEL_NAME + '.pth'))

    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler']) 
    
    START_EPOCH = checkpoint['epoch']
    best_loss = checkpoint['best_loss']

    train_loss_lst = checkpoint['train_loss_lst']
    valid_loss_lst = checkpoint['train_loss_lst']

    for i in range(len(train_loss_lst)): 
        writer.add_scalar('training loss', train_loss_lst[i], i)
        writer.add_scalar('validation loss', valid_loss_lst[i], i)
    
    current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print("==> Resuming from epoch: %d, learning rate %.4f, best loss: %.4f"%(START_EPOCH, current_learning_rate, best_loss))
    print("="*50)


# start the training/validation process
start = time.time()
print("==> Training starts!")
print("="*50)

for i in range(START_EPOCH, EPOCHS):    
    # switch to train mode
    model.train()
    
    print("Epoch %d:" %i)

    train_loss = 0 # track training loss if you want
    
    # Train the model for 1 epoch.
    for batch_idx, (inputs) in enumerate(train_loader):
        # copy inputs to device
        inputs = inputs.float().to(device)

        # compute the output and loss
        outputs, _ = model(inputs)

        inputs = inputs.transpose(1, 2)
        outputs = outputs.transpose(1, 2)
        dist1, _ = criterion(outputs, inputs)

        loss = (torch.mean(dist1))
        train_loss += loss.to('cpu').detach().numpy()

        # zero the gradient
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # apply gradient and update the weights
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    train_loss_lst.append(avg_train_loss)
    writer.add_scalar('training loss', avg_train_loss, i)
    # switch to eval mode
    model.eval()

    # this help you compute the validation accuracy
    total_examples = 0
    correct_examples = 0
    
    val_loss = 0 # again, track the validation loss if you want
    
    # disable gradient during validation, which can save GPU memory
    with torch.no_grad():
        for batch_idx, (inputs) in enumerate(val_loader):
            
            inputs = inputs.float().to(device)
            outputs, _ = model(inputs)

            inputs = inputs.transpose(1, 2)
            outputs = outputs.transpose(1, 2)

            dist1, _ = criterion(outputs, inputs)

            loss = (torch.mean(dist1))
            val_loss += loss
            
            # log first batch outcome plots
            if (batch_idx == 0 and i % LOG_EPOCHS == 0):
                print("Logging generate shape quality")
                fig = draw3DPoints(outputs.cpu().detach().numpy()[0].T, point_size = 5)
                save3DPointsImage(fig, save_path = os.path.join("image", MODEL_NAME), title = "epoch_%d"%i)
                
                # img = imread(os.path.join("image", MODEL_NAME, "epoch_%d"%i + ".png"))
                # writer.add_image("epoch_%d"%i + ".png", img, i)


    avg_val_loss = val_loss / len(val_loader)    
    valid_loss_lst.append(avg_val_loss.cpu().detach().numpy())
    writer.add_scalar('validation loss', avg_val_loss, i)
    
    print("Training loss: %.4f, Validation loss: %.4f" % (avg_train_loss, avg_val_loss))

    # save the model checkpoint
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print("Saving Model...")
        state = {'state_dict': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': i,
                'best_loss': best_loss,
                'train_loss_lst': train_loss_lst,
                'valid_loss_lst': valid_loss_lst}
        torch.save(state, os.path.join(CHECKPOINT_PATH, MODEL_NAME + '.pth'))
        
    print('')

    scheduler.step()

print("="*50)
print(f"==> Optimization finished in {time.time() - start:.2f}s! Best validation loss: {best_loss:.4f}")

plt.plot(train_loss_lst, label='Train Loss')
plt.plot(valid_loss_lst, label='Test Loss')
plt.title(MODEL_NAME + " Learning Curve")
plt.legend()
plt.xlabel('epoch')
plt.yscale('log')
plt.ylabel('Loss')
plt.savefig(MODEL_NAME + '_learning_curve.png')

writer.close()

# run 'tensorboard --logdir=runs' in terminal to get tensorboard session link
# os.system('tensorboard --logdir=runs')
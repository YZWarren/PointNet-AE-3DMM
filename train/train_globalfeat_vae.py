import argparse
import os, sys
sys.path.append('../')

import time

import numpy as np
import matplotlib.pyplot as plt

# Import pytorch dependencies
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Import toolkits
from utils.visualization_3D_objects import *

from nn.gf_vae import *
from nn.pointnetae import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='shapenet', help='Dataset to use [default: shapenet]')
parser.add_argument('--model_name', default='VAE_global_feat', help='Model name [default: VAE_global_feat]')
parser.add_argument('--point_decoder_name', default='PointNetAE', help='Point cloud decoder name [default: PointNetAE]')
parser.add_argument('--category', default='car', help='Which single class to train on [default: car]')
parser.add_argument('--checkpoint_path', default='../saved_nn', help='Path to save model checkpoint [default: ../saved_nn]')
parser.add_argument('--num_point', type=int, default=2048, help='Number of Points [default: 2048]')
parser.add_argument('--input_dim', type=int, default=1024, help='Input vector dimension [default: 1024]')
parser.add_argument('--inter_dim', type=int, default=512, help='Intermediate feature space dimension [default: 1024]')
parser.add_argument('--latent_dim', type=int, default=128, help='Latent vector dimension [default: 128]')
parser.add_argument('--max_epoch', type=int, default=350, help='Epoch to run [default: 350]')
parser.add_argument('--log_epoch', type=int, default=30, help='Epoch to log results [default: 30]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 12]')
parser.add_argument('--beta', type=float, default=1, help='beta for VAE KL regularization [default: 1]')
parser.add_argument('--initial_lr', type=float, default=1e-3, help='Initial learning rate [default: 1e-3]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for lr decay [default: 100]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
parser.add_argument('--resume', choices=['yes', 'no'], default='no', help='Resume previous training (yes/no) [default: no]')
FLAGS = parser.parse_args()

DATASET = FLAGS.dataset
MODEL_NAME = FLAGS.model_name
PCDECODER = FLAGS.point_decoder_name
DATA_PATH = os.path.join("../evaluate_model/", DATASET +"_" + FLAGS.category + "_global_feat.csv")
CHECKPOINT_PATH = FLAGS.checkpoint_path
NUM_POINT = FLAGS.num_point
INPUT_DIM = FLAGS.input_dim
INTER_DIM = FLAGS.inter_dim
LATENT_DIM = FLAGS.latent_dim
BATCH_SIZE = FLAGS.batch_size
INITIAL_LR = FLAGS.initial_lr
BETA = FLAGS.beta
EPOCHS = FLAGS.max_epoch
LOG_EPOCHS = FLAGS.log_epoch
DECAY_STEPS = FLAGS.decay_step
DECAY_EPOCHS = np.arange(DECAY_STEPS, DECAY_STEPS * (np.round(EPOCHS/DECAY_STEPS)), DECAY_STEPS)
DECAY_RATE = FLAGS.decay_rate

RESUME = FLAGS.resume == 'yes'

global_feat = np.loadtxt(DATA_PATH, delimiter=",", dtype=float)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(global_feat, test_size = 0.1)

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
model = VAE()
model = model.to(device)

# Check if on GPU
assert(next(model.parameters()).is_cuda)

pointnetae = PointNet_AE(3, NUM_POINT)
state_dict = torch.load(os.path.join(CHECKPOINT_PATH, PCDECODER + '.pth')) # change the path to your own checkpoint file
pointnetae.cuda()
pointnetae.load_state_dict(state_dict['state_dict'])

# create loss function: Chamfer distance
criterion = lambda recon_x, x: F.mse_loss(recon_x, x, reduction='sum')

# Add optimizer
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCHS, gamma=DECAY_RATE, verbose=True)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/' + MODEL_NAME)

best_loss = 1e20
START_EPOCH = 0

if RESUME:
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, MODEL_NAME + '.pth'))

    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler']) 
    
    START_EPOCH = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print("Resuming from epoch: %d, learning rate %.4f, best loss: %.4f"%(START_EPOCH, current_learning_rate, best_loss))

# dummy input latent vector to check training result
random_latent_vector = torch.randn(BATCH_SIZE, LATENT_DIM).cuda()

# start the training/validation process
start = time.time()
print("==> Training starts!")
print("="*50)

# store loss learning curve
train_loss_lst = []
valid_loss_lst = []

for i in range(START_EPOCH, EPOCHS):    
    # switch to train mode
    model.train()
    
    print("Epoch %d:" %i)

    train_loss = 0 # track training loss if you want
    train_recon_loss = 0
    
    # Train the model for 1 epoch.
    for batch_idx, (inputs) in enumerate(train_loader):
        # copy inputs to device
        inputs = inputs.float().to(device)

        # compute the output and loss
        outputs = model(inputs)

        recon_loss = criterion(outputs, inputs)
        loss = recon_loss + BETA * model.kl

        train_loss += loss.to('cpu').detach().numpy()
        train_recon_loss += recon_loss.to('cpu').detach().numpy()
        # zero the gradient
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # apply gradient and update the weights
        optimizer.step()

    avg_loss = train_loss / len(train_loader)
    avg_recon_loss = train_recon_loss / len(train_loader)
    print("Training loss: %.4f (Recon: %.4f, KL: %.4f)" %(avg_loss, avg_recon_loss, avg_loss - avg_recon_loss))
    train_loss_lst.append(avg_loss)

    # switch to eval mode
    model.eval()

    # this help you compute the validation accuracy
    total_examples = 0
    correct_examples = 0
    
    val_loss = 0 # again, track the validation loss if you want
    val_recon_loss = 0
    # disable gradient during validation, which can save GPU memory
    with torch.no_grad():
        for batch_idx, (inputs) in enumerate(val_loader):
            # copy inputs to device
            inputs = inputs.float().to(device)
            # compute the output and loss
            outputs = model(inputs)

            recon_loss = criterion(outputs, inputs)
            loss = recon_loss + model.kl
            val_loss += loss
            
            val_recon_loss += recon_loss.to('cpu').detach().numpy()

    avg_loss = val_loss / len(val_loader)
    avg_recon_loss = val_recon_loss / len(val_loader)
    print("Validation loss: %.4f (Recon: %.4f, KL: %.4f)" %(avg_loss, avg_recon_loss, avg_loss - avg_recon_loss))
    valid_loss_lst.append(avg_loss.cpu().detach().numpy())
    
    
    # log outcome plots
    if (i % LOG_EPOCHS == 0):
        print("Logging generate shape quality")
        X_sample = pointnetae.decoder(model.decoder(random_latent_vector)).cpu().detach().numpy()
        fig = draw3DPoints(X_sample[0], point_size = 5)
        save3DPointsImage(fig, save_path = os.path.join("image", MODEL_NAME), title = "epoch_%d"%i)

    # save the model checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print("Saving ...")
        state = {'state_dict': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': i,
                'best_loss': best_loss}
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCAutoEncoder(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: Batch of Point Cloud B x 3 x N
    Output: reconstructed points
    """
    def __init__(self, point_dim, num_points, 
                    xavier = False, bn_momentum = 0.1, 
                    bn_encoder = False, bn_decoder = False):
        super(PCAutoEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_points*3)

        #batch norm
        self.bn1 = nn.BatchNorm1d(64, momentum = bn_momentum) if bn_encoder else nn.Sequential()
        self.bn2 = nn.BatchNorm1d(128, momentum = bn_momentum) if bn_encoder else nn.Sequential()
        self.bn3 = nn.BatchNorm1d(1024, momentum = bn_momentum) if bn_encoder else nn.Sequential()

        self.bn_fc = nn.BatchNorm1d(1024, momentum = bn_momentum) if bn_decoder else nn.Sequential()

        if xavier: self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder, MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))
        
        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = F.relu(self.bn_fc(self.fc2(x)))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points , global_feat

if __name__ == '__main__':
    # GPU check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    # Model Sanity Check
    net = PCAutoEncoder(3, 2048)
    net = net.to(device)
    summary(net, (3,2048))

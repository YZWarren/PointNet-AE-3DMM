import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PointNet_Encoder(nn.Module):
    """ PointNet Encoder for Point Cloud (following PointNet by Charles Q. et al.)
    Input: Batch of Point Cloud B x 3 x N
    Output: global features B x 1088 x N
    """
    def __init__(self):
        super(PointNet_Encoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        #batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    
    def forward(self, x):
        n_pts = x.shape[2]
        #encoder, MLP
        x = F.relu(self.bn1(self.conv1(x)))

        pointfeat = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024, 1).repeat(1,1,n_pts)
        x = torch.cat([x, pointfeat], 1)
        # get the global embedding
        return x

class Decoder(nn.Module):
    """Convolutional Decoder for Point Cloud AE
    Input: Batch of global features B x 1088 x N
    Output: reconstructed points B x 3 x N
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        #decoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x

class PointNet_AE(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: Batch of Point Cloud B x 3 x N
    Output: Batch of reconstructed points, extracted global features
    """
    def __init__(self):
        super(PointNet_AE, self).__init__()

        self.encoder = PointNet_Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # encode
        global_feat = self.encoder(x)
        # decode
        reconstructed_points = self.decoder(global_feat)

        return reconstructed_points , global_feat

if __name__ == "__main__":
    ## Sanity Check
    model = PointNet_AE()
    
    # GPU check                
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print("Run on GPU...")
    else:
        print("Run on CPU...")

    # Generator Definition  
    n_pts = 4096
    model = model.to(device)
    summary(model, (3,n_pts))

    # Test forward pass
    z = torch.randn(5,3,n_pts)
    z = z.to(device)
    out, gf = model(z)
    # Check output shape
    print(out.detach().cpu().numpy().shape)
    print(gf.detach().cpu().numpy().shape)
    print("Forward pass successful")
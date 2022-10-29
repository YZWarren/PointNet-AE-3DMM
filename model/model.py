import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet_Encoder(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: Batch of Point Cloud B x 3 x N
    Output: reconstructed points
    """
    def __init__(self, point_dim):
        super(PointNet_Encoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)
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
        #encoder, MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        # print(x.shape)
        x = x.view(-1, 1024)
        # get the global embedding
        return x

class Decoder(nn.Module):
    """ Decoder for Point Cloud VAE
    Input: Batch of Point Cloud B x 1024 x 1
    Output: reconstructed points B x 3 x N
    """
    def __init__(self, point_dim, num_points):
        super(Decoder, self).__init__()
        self.point_dim = point_dim
        self.num_points = num_points
        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_points*3)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batch_size = x.shape[0]

        #decoder
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, self.point_dim, self.num_points)

        return reconstructed_points

class PointNet_AE(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: Batch of Point Cloud B x 3 x N
    Output: reconstructed points
    """
    def __init__(self, point_dim, num_points):
        super(PointNet_AE, self).__init__()

        self.encoder = PointNet_Encoder(point_dim)
        self.decoder = Decoder(point_dim, num_points)

    def forward(self, x):
        # encode
        global_feat = self.encoder(x)
        # decode
        reconstructed_points = self.decoder(global_feat)

        return reconstructed_points , global_feat

class PointNet_AE_lowerDim(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: Batch of Point Cloud B x 3 x N
    Output: reconstructed points
    """
    def __init__(self, point_dim, num_points, latent_dim):
        super(PointNet_AE_lowerDim, self).__init__()

        self.encoder = PointNet_Encoder(point_dim)
        self.decoder = Decoder(point_dim, num_points)

        self.fc_encode = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=latent_dim)
        )

        self.fc_decode = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        # encode
        global_feat = self.encoder(x)

        latent_space = self.fc_encode(global_feat)

        decoded_global_feat = self.fc_decode(latent_space)

        # decode
        reconstructed_points = self.decoder(decoded_global_feat)

        return reconstructed_points, latent_space

class PointNet_VAE(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: Batch of Point Cloud B x 3 x N
    Output: reconstructed points
    """
    def __init__(self, point_dim, num_points):
        super(PointNet_VAE, self).__init__()

        self.encoder = PointNet_Encoder(point_dim)

        self.fc_mu = nn.Linear(in_features=1024, out_features=1024)
        self.fc_var = nn.Linear(in_features=1024, out_features=1024)

        self.decoder = Decoder(point_dim, num_points)

        self.kl = 0
    
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)
    
    def forward(self, x):
        # encode
        global_feat = self.encoder(x)

        # get variational distribution
        mu = self.fc_mu(global_feat)
        logvar = self.fc_var(global_feat)

        self.kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        z = self.reparameterise(mu, logvar)

        # decode
        reconstructed_points = self.decoder(z)

        return reconstructed_points, mu, torch.exp(logvar / 2)
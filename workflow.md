# Preprocess shapenet data
python prepare_data.py --help

# Train PointNet autoencoder
python train_pointnetae.py --help

# Extract global features and save
Extract in folder evaluate_model

# Train Variational AutoEncoder on global features
python train_globalfeat_vae.py --help

# Sample from latent distribution
### pass to VAE decoder
### pass to PointNet AE decoder
### plot generated object


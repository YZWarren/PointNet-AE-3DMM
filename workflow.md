# Preprocess shapenet data
python prepare_data.py --dataset {dataset to use [default: shapenet]} --data_class {the class of dataset to use [default: shapenet]} --num_point {Point Number [default: 2048]} --var_range {Range of Variance Values [default: 1.0]}

# Train PointNet autoencoder
Todo: find best hyperparameter combination and make default

# Extract global features and save
Todo: in a seperate python script

# Train Variational AutoEncoder on global features

# Sample from latent distribution
### pass to VAE decoder
### pass to PointNet AE decoder
### plot generated object


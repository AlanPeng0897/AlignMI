# Architechture
lpips_type = 'vgg'  # 'alex'
first_inv_type = 'w'
optim_type = 'adam'

# Locality regularization
use_locality_regularization = False
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
# regulizer_alpha = 0.5

# Loss
pt_l2_lambda = 0  # 1
pt_lpips_lambda = 1
dist_lambda = 1

# Steps
LPIPS_value_threshold = -1  # 0.06
first_inv_steps = 300  # Point-wise-300  # CT-300  # 450
max_pti_steps = 150  # 150  # 100  # 350

# Optimization
first_inv_lr = 5e-3
pti_learning_rate = 3e-4  # 3e-4

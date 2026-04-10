method = 'PROBKAN_V2'

# model
hid_S = 32

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

# probkan_v2 loss weights
alpha_mse = 1.0
lambda_sigma_reg = 1e-3

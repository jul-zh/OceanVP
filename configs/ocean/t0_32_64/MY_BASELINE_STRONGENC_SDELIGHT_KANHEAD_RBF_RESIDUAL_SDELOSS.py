method = 'my_baseline_strongenc_sdelight_kanhead_rbf_residual_sdeloss'

# model
hid_S = 32
kan_head_hidden_dim = 64
kan_head_num_centers = 16
kan_head_init_gamma = 1.0
kan_head_dropout = 0.05

# sde loss
lambda_sde = 1e-4
sde_alpha = 1.0
sde_beta = 0.1

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

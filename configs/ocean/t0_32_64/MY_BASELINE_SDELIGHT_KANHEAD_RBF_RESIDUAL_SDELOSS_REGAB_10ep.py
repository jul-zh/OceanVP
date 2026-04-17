method = 'my_baseline_sdelight_kanhead_rbf_residual_sdeloss_regab'

# model
hid_S = 32
kan_head_hidden_dim = 64
kan_head_num_centers = 16
kan_head_init_gamma = 1.0
kan_head_dropout = 0.05

# sde loss from regression-estimated alpha/beta
lambda_sde = 1e-4
sde_alpha = 0.1051857
sde_beta = 6.1e-05

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

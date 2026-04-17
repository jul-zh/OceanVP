method = 'my_baseline_sdelight_kanhead_rbf_residual_sdeloss_bins_v2'

# model
hid_S = 32
kan_head_hidden_dim = 64
kan_head_num_centers = 16
kan_head_init_gamma = 1.0
kan_head_dropout = 0.05

# soft delayed bins-SDE
lambda_sde = 1e-6
sde_start_epoch = 5
sde_bins_path = '/home/yzhidkova/projects/OceanVP/logs/sde_bins_t0_train_1000.json'

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

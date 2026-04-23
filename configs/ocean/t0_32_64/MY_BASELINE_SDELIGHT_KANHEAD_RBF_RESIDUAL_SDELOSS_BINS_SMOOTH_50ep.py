method = 'my_baseline_sdelight_kanhead_rbf_residual_sdeloss_bins_smooth'

hid_S = 32
kan_head_hidden_dim = 96
kan_head_num_centers = 24
kan_head_init_gamma = 1.0
kan_head_dropout = 0.05

lambda_sde = 1e-4
lambda_smooth = 1e-7
sde_bins_path = '/home/yzhidkova/projects/OceanVP/logs/sde_bins_t0_train_1000.json'

lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

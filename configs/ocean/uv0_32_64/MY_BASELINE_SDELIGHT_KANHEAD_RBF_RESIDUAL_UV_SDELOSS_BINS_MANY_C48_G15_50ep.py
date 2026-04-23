method = 'my_baseline_sdelight_kanhead_rbf_residual_uv_sdeloss_bins'

hid_S = 32
kan_head_hidden_dim = 96
kan_head_num_centers = 48
kan_head_init_gamma = 1.5
kan_head_dropout = 0.05

lambda_sde = 1e-4
sde_bins_u_path = '/home/yzhidkova/projects/OceanVP/logs/sde_bins_u0_train_1000.json'
sde_bins_v_path = '/home/yzhidkova/projects/OceanVP/logs/sde_bins_v0_train_1000.json'

lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

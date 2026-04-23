method = 'my_baseline_sdelight_kanhead_rbf_residual_sdeloss_bins_topk'

hid_S = 32
kan_head_hidden_dim = 96
kan_head_num_centers = 40
kan_head_init_gamma = 1.5
kan_head_dropout = 0.05

lambda_topk = 0.2
topk_frac = 0.1
lambda_sde = 1e-4
sde_bins_path = '/home/yzhidkova/projects/OceanVP/logs/sde_bins_t0_train_1000.json'

lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

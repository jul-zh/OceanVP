method = 'my_baseline_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins'

hid_S = 32
kan_head_hidden_dim = 96
kan_head_num_centers = 48
kan_head_init_gamma = 1.5
kan_head_dropout = 0.05

lambda_sde = 5e-5
sde_eps = 1e-6
sde_bins_path = '/home/yzhidkova/projects/OceanVP/logs/sde_bins_s0_train_1000.json'

lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

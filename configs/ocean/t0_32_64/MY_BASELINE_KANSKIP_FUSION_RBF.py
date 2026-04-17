method = 'MY_BASELINE_KANSKIP_FUSION_RBF'

# model
hid_S = 32
kan_fusion_hidden_dim = 64
kan_fusion_num_centers = 16
kan_fusion_init_gamma = 1.0
kan_fusion_dropout = 0.05

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

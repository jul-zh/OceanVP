method = 'MY_BASELINE_KANHEAD_RBF_RESIDUAL'

# model
hid_S = 32
kan_head_hidden_dim = 96
kan_head_num_centers = 24
kan_head_init_gamma = 0.5
kan_head_dropout = 0.0

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

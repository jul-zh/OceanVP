method = 'MY_BASELINE_KANHEAD_SPLINE'

# model
hid_S = 32
kan_head_hidden_dim = 64
kan_head_num_knots = 16
kan_head_xmin = -2.0
kan_head_xmax = 2.0
kan_head_dropout = 0.05

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

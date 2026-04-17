method = 'MY_BASELINE_KANDECODER_GATE_RBF'

# model
hid_S = 32
kan_gate_hidden_dim = 64
kan_gate_num_centers = 16
kan_gate_init_gamma = 1.0
kan_gate_dropout = 0.05

# training
lr = 5e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 0

from lib.datasets.dataloader_ocean import load_data

train_loader, vali_loader, test_loader = load_data(
    batch_size=2,
    val_batch_size=2,
    data_root='/home/yzhidkova/datasets/oceanvp_raw/OceanVP_HYCOM_32_64',
    data_split='32_64',
    data_name='ocean_t0',
    train_time=['1994', '2013'],
    val_time=['2014', '2014'],
    test_time=['2015', '2015'],
    idx_in=[i for i in range(-15, 1)],
    idx_out=[i for i in range(1, 17)],
    step=1,
    level=1,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    temp_stride=1,
)

print("train years unique:", sorted(set(train_loader.dataset.year.values.tolist()))[:3], "...", sorted(set(train_loader.dataset.year.values.tolist()))[-3:])
print("vali years unique:", sorted(set(vali_loader.dataset.year.values.tolist())))
print("test years unique:", sorted(set(test_loader.dataset.year.values.tolist())))
print("train len:", len(train_loader.dataset))
print("vali len:", len(vali_loader.dataset))
print("test len:", len(test_loader.dataset))

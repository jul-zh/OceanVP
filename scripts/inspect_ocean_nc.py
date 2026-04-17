import xarray as xr

path = "/home/yzhidkova/datasets/oceanvp_raw/OceanVP_HYCOM_32_64/water_temp_depth_0m/water_temp_depth_0m_1994_32_64.nc"
ds = xr.open_dataset(path)

print("PATH:", path)
print("DATA_VARS:", list(ds.data_vars))
print("COORDS:", list(ds.coords))
print(ds)

import cfg_target
from data_load_new import DataLoader
data = DataLoader(cfg_target)

target = data.data_download_target()
covariates_us, covariates_sea, covariates_global, spatial_covariates, temporal_covariates = data.data_download_cov()

#spatial_covariates, temporal_covariates = data.data_download_cov()

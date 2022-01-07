"""
To create training-test sets and training - test sets
"""
import pandas as pd
import preprocess
import cfg_target as cfg
import progressbar

def get_enso_varnames():
	return ['mei', 'nao', 'mjo_phase', 'mjo_amplitude', 
			'nino3', 'nino4', 'nino3.4', 'nino1+2']

def latlon_convert(lats, lons):
	return lats, lons + 360

def main(flags):	

	mode = flags.mode

	idx = pd.IndexSlice
	date_col = 'start_date'

	spatial_data = pd.read_hdf(cfg.data_target_file)
	spatial_data = spatial_data.reset_index().set_index('start_date') # for faster search in .loc
	
	global_data = pd.read_hdf(cfg.data_cov_file)
	cols = []
	for c in global_data.columns:
		if cfg.target_var in c:
			cols.append(c)
		for var in (cfg.vars + cfg.temporal_set):
			if var in c:
				cols.append(c)

	global_data = global_data[cols]
	
	train_start_date = cfg.train_start_date
	end_date = cfg.end_date


	time_index = pd.date_range(train_start_date, end_date, freq='1D')

	spatial_data = spatial_data.loc[idx[time_index]]
	spatial_data = spatial_data.reset_index().set_index(['lat', 'lon', 'start_date'])
	global_data = global_data.loc[idx[time_index]] 
	global_data = global_data.loc[:,~global_data.columns.duplicated()]

	clims = pd.read_hdf('tmp2m_2degree_clim_per_monthday.h5')
	clims = clims.rename({'tmp2m': 'tmp2m_clim'}, axis=1)
	spatial_data = spatial_data.reset_index().set_index(['month', 'day', 'lat', 'lon'])
	spatial_data = pd.merge(spatial_data, clims, on=['month', 'day', 'lat', 'lon'])
	spatial_data = spatial_data.reset_index().set_index(['lat', 'lon', 'start_date'])

	cv_path = cfg.rootpath_cv
	# forecast_path = cfg.forecast_rootpath
	forecast_path = '/SSF/data/tmp2m_34w_xgboost_climappend/'

	target_var = cfg.target_var

	val_years = cfg.val_years
	test_years = cfg.test_years

	val_train_range = cfg.val_train_range
	test_train_range = cfg.test_train_range

	past_years = cfg.past_kyears

	val_range = cfg.val_range
	val_freq = cfg.val_freq


	test_start_date = cfg.test_start_date
	test_time_index_all = pd.date_range(test_start_date, end_date, freq='7D')


	# By default collect date, lat-lon and temperature time series
	global_ids = ['month', 'day']
	# global_ids = []
	# Append ENSO indices for collection
	global_ids += get_enso_varnames() 
	spatial_ids = ['tmp2m_clim', 'tmp2m_timeseries', 'tmp2m_daily_mean',
					]
	input_ids = spatial_ids + global_ids

	target_data = spatial_data[['tmp2m']]
	spatial_data = spatial_data.drop('tmp2m', axis=1)

	# merge spatial data and global data (fill in global data for each location)
	spatial_data = spatial_data.reset_index().set_index('start_date') # match indices
	
	combined_data = pd.merge(global_data, spatial_data, how='left', on='start_date')
	combined_data = combined_data.reset_index().set_index(['lat', 'lon', 'start_date'])
	# combined_data = combined_data[input_ids]

	combined_data = combined_data.sort_index()
	target_data = target_data.sort_index()

	# Filter by lat x lon
	combined_data = combined_data.reset_index()
	
	# Create XGBoost dataset
	if mode == 'combo':
		preprocess.train_test_split_combo(forecast_path,
									target_data,
									combined_data.set_index(['lat', 'lon', 'start_date']),
									test_time_index_all,
									0, 0,
									train_range=test_train_range,
									past_years=past_years,
									all_test=True,
									n_jobs=20)

	else:
		# TODO: Move regional filtering out of this script
		import numpy as np
		lons = np.array([-110, -93])
		lats = np.array([24, 36])
		lats, lons = latlon_convert(lats, lons)
		
		combined_data = combined_data[
									  (combined_data['lon'] >= lons[0]) &
									  (combined_data['lon'] <= lons[1]) &
									  (combined_data['lat'] >= lats[0]) &
									  (combined_data['lat'] <= lats[1])
									]
		
		target_data = target_data.reset_index()
		target_data = target_data[
									  (target_data['lon'] >= lons[0]) &
									  (target_data['lon'] <= lons[1]) &
									  (target_data['lat'] >= lats[0]) &
									  (target_data['lat'] <= lats[1])
									]

		data = {
			'inputs': combined_data.set_index(['lat', 'lon', 'start_date']),
			'target': target_data.set_index(['lat', 'lon', 'start_date'])
		}


		preprocess.train_test_split_target_ar(
												forecast_path,
												data,
												global_ids + ['lat', 'lon'],
												test_time_index_all,
												None, None,
												train_range=test_train_range,
												past_years=past_years,
												n_jobs=20
											)




if __name__ == "__main__":
	main()

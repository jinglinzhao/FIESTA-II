import numpy as np
import matplotlib.pyplot as plt
from FIESTA_II import *

#----------------------------------
# Define a plot function
#----------------------------------
def plot_all(k_mode, t, rv, erv, ind, eind, ts_xlabel, rv_xlabel, pe_xlabel, ind_yalbel, file_name):

	'''
	e.g. 
		k_mode 		= 11
		t 			= bjd_daily
		rv 			= rv_daily
		erv 		= erv_daily
		ind 		= shift_function
		eind 	 	= err_shift_spectrum
		ts_xlabel 	= 'BJD - 2400000 [d]'
		rv_xlabel 	= '$RV_{HARPS}$'
		pe_xlabel 	= 'Period [days]'
		ind_yalbel	= 'A'
		file_name 	= 'time-series_and_shift_correlation.png'

	'''

	def new_periodogram(x, y, dy,
					plot_min_t=2, max_f=1, spp=100):
		
		from scipy.signal import find_peaks
		from astropy.timeseries import LombScargle

		time_span = (max(x) - min(x))
		min_f   = 1/time_span

		frequency, power = LombScargle(x, y, dy).autopower(minimum_frequency=min_f,
													   maximum_frequency=max_f,
													   samples_per_peak=spp)

		plot_x = 1/frequency
		idxx = (plot_x>plot_min_t) & (plot_x<time_span/2)
		height = max(power[idxx])*0.4
		ax.plot(plot_x[idxx], power[idxx], 'k-', label=r'$\xi$'+str(i+1), alpha=0.5)
		peaks, _ = find_peaks(power[idxx], height=height)
		ax.plot(plot_x[idxx][peaks], power[idxx][peaks], "ro")

		for n in range(len(plot_x[idxx][peaks])):
			ax.text(plot_x[idxx][peaks][n], power[idxx][peaks][n], '%.1f' % plot_x[idxx][peaks][n], fontsize=10)

		ax.set_xlim([plot_min_t,time_span/2])
		ax.set_ylim([0, 3*height])
		ax.set_xscale('log')

	from sklearn.linear_model import LinearRegression

	# set up the plotting configureations
	alpha1, alpha2 = [0.5,0.2]
	widths 	= [7,1,7]
	heights = [1]*(k_mode+1)
	gs_kw 	= dict(width_ratios=widths, height_ratios=heights)
	plt.rcParams.update({'font.size': 12})
	fig6, f6_axes = plt.subplots(figsize=(16, k_mode+1), ncols=3, nrows=k_mode+1, constrained_layout=True,
	                             gridspec_kw=gs_kw)

	# plots 
	for r, row in enumerate(f6_axes):
		for c, ax in enumerate(row):	

			# time-series 
			if c==0:
				if r==0:
					ax.errorbar(t, rv-np.mean(rv), erv, marker='.', ms=5, color='black', ls='none', alpha=alpha1)
					ax.set_title('Time-series')
					ax.set_ylabel(rv_xlabel)
				else:				
					ax.errorbar(t, ind[r-1,:], eind[r-1,:],  marker='.', ms=5, color='black', ls='none', alpha=alpha1)
					ax.set_ylabel(ind_yalbel + '$_{' + str(r) + '}$')
				if r!=k_mode:
					ax.set_xticks([])
				else:
					ax.set_xlabel(ts_xlabel)

			if c==1:
				if r==0:
					reg = LinearRegression().fit(rv.reshape(-1, 1), rv.reshape(-1, 1))
					score = reg.score(rv.reshape(-1, 1), rv.reshape(-1, 1))
					ax.set_title('score = {:.2f}'.format(score))
					ax.plot(rv-np.mean(rv), rv-np.mean(rv), 'k.', alpha = alpha2)				
				if r>0:
					reg = LinearRegression().fit(rv.reshape(-1, 1), ind[r-1,:].reshape(-1, 1))
					score = reg.score(rv.reshape(-1, 1), ind[r-1,:].reshape(-1, 1))
					ax.set_title('score = {:.2f}'.format(score))
					ax.plot(rv-np.mean(rv), ind[r-1,:], 'k.', alpha = alpha2)
				if r!=k_mode:
					ax.set_xticks([])
				else:
					ax.set_xlabel(rv_xlabel)
				ax.yaxis.tick_right()

			if c==2:
				if r==0:
					new_periodogram(t, rv, erv)
					ax.set_title('Periodogram')
				if r>0:
					new_periodogram(t, ind[r-1,:], eind[r-1,:])
				if r!=k_mode:
					ax.set_xticks([])
				if r==k_mode:
					ax.set_xlabel(pe_xlabel)

	plt.savefig(file_name)
	plt.close('all')

#----------------------------------
# Import pre-processed daily binned data
#----------------------------------
V_grid 		= np.loadtxt('./daily_binned_HARPS-N_data/V_grid.txt')
CCF_daily 	= np.loadtxt('./daily_binned_HARPS-N_data/CCF_daily.txt')
eCCF_daily 	= np.loadtxt('./daily_binned_HARPS-N_data/eCCF_daily.txt')
bjd_daily 	= np.loadtxt('./daily_binned_HARPS-N_data/bjd_daily.txt')
rv_daily 	= np.loadtxt('./daily_binned_HARPS-N_data/rv_daily.txt')
rv_raw_daily= np.loadtxt('./daily_binned_HARPS-N_data/rv_raw_daily.txt')
erv_daily 	= np.loadtxt('./daily_binned_HARPS-N_data/erv_daily.txt')

#----------------------------------
# Go FIESTA II
#----------------------------------
k_max = 16
df, shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(V_grid, CCF_daily, eCCF_daily, k_max=k_max)
shift_spectrum 		*= 1000
err_shift_spectrum 	*= 1000
RV_gauss 			*= 1000

shift_function 	= np.zeros(shift_spectrum.shape)
for i in range(shift_spectrum.shape[0]):
	shift_function[i,:] = shift_spectrum[i,:] - rv_raw_daily

plot_all(k_mode=k_max, t=bjd_daily, rv=rv_daily, erv=erv_daily, 
	ind=power_spectrum, eind=err_power_spectrum, 
	ts_xlabel='BJD - 2400000 [d]', 
	rv_xlabel='$RV_{HARPS}$', 
	pe_xlabel='Period [days]',
	ind_yalbel=r'$A$',
	file_name='Amplitude_time-series_correlation_periodogram.png')
plt.close()

plot_all(k_mode=k_max, t=bjd_daily, rv=rv_daily, erv=erv_daily, 
	ind=shift_function, eind=err_shift_spectrum, 
	ts_xlabel='BJD - 2400000 [d]', 
	rv_xlabel='$RV_{HARPS}$', 
	pe_xlabel='Period [days]',
	ind_yalbel=r'$\Delta RV$',
	file_name='shift_time-series_correlation_periodogram.png')
plt.close()


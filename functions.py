import matplotlib.pyplot as plt
import numpy as np

def periodogram6(ax, x, y, vlines, plot_min_t=1, max_f=1, spp=100):

	from scipy.signal import find_peaks
	from astropy.timeseries import LombScargle

	time_span = (max(x) - min(x))
	min_f   = 1/time_span

	frequency, power = LombScargle(x, y).autopower(minimum_frequency=min_f,
												   maximum_frequency=max_f,
												   samples_per_peak=spp)

	plot_x = 1/frequency
	idxx = (plot_x>plot_min_t) & (plot_x<100)
	height = max(power[idxx])*0.9
	ax.plot(plot_x[idxx], power[idxx], 'k-', label=r'$\xi$'+str(i+1), alpha=0.7)
	peaks, _ = find_peaks(power[idxx], height=height)
	ax.plot(plot_x[idxx][peaks], power[idxx][peaks], "ro")

	for n in range(len(plot_x[idxx][peaks])):
		ax.text(plot_x[idxx][peaks][n], power[idxx][peaks][n], '%.1f' % plot_x[idxx][peaks][n], fontsize=10)

	for xc in vlines:
		ax.axvline(x=xc, color='r', linestyle='-', lw=2, alpha=0.3)

	ax.set_xlim([plot_min_t,100])
	ax.set_ylim([0,1.7*height])
	ax.set_xscale('log')

def plot_all(k_mode, t, rv, erv, ind, eind, ts_xlabel, rv_xlabel, pe_xlabel, ind_yalbel, file_name, height_ratio=0.7, vlines=[], HARPS=True):

	'''
	e.g. 
		k_mode 		= 11
		t 			= bjd_daily
		rv 			= rv_daily
		erv 		= erv_daily
		ind 		= ΔRV_k
		eind 	 	= eRV_FT_k
		ts_xlabel 	= 'BJD - 2400000 [d]'
		rv_xlabel 	= '$RV_{HARPS}$'
		pe_xlabel 	= 'Period [days]'
		ind_yalbel	= 'A'
		file_name 	= 'time-series_and_shift_correlation.png'

	'''

	def new_periodogram(x, y, dy, vlines, height_ratio=height_ratio, plot_min_t=2, max_f=1, spp=100):
	
		from scipy.signal import find_peaks
		from astropy.timeseries import LombScargle

		time_span = (max(x) - min(x))
		min_f   = 1/time_span

		frequency, power = LombScargle(x, y, dy).autopower(minimum_frequency=min_f,
													   maximum_frequency=max_f,
													   samples_per_peak=spp)

		plot_x = 1/frequency
		idxx = (plot_x>plot_min_t) & (plot_x<time_span/2)
		height = max(power[idxx])*height_ratio
		ax.plot(plot_x[idxx], power[idxx], 'k-', alpha=0.5)
		peaks, _ = find_peaks(power[idxx], height=height)
		ax.plot(plot_x[idxx][peaks], power[idxx][peaks], "ro")

		for n in range(len(plot_x[idxx][peaks])):
			ax.text(plot_x[idxx][peaks][n], power[idxx][peaks][n], '%.1f' % plot_x[idxx][peaks][n], fontsize=10)

		ax.set_xlim([plot_min_t,time_span/2])
		ax.set_ylim([0, 1.25*max(power[idxx])])

		if vlines!=[]:        
			for xc in vlines:
				ax.axvline(x=xc, color='r', linestyle='-', lw=2, alpha=0.2)        
		if HARPS == True:
			ax.axvspan(200, 220, facecolor='r', alpha=0.2)

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
					ax.tick_params(labelbottom=False)
				else:
					ax.set_xlabel(ts_xlabel)

			if c==1:
				if r==0:
					reg = LinearRegression().fit(rv.reshape(-1, 1), rv.reshape(-1, 1))
					score = reg.score(rv.reshape(-1, 1), rv.reshape(-1, 1))
					adjust_R2 = 1-(1-score)*(len(t)-1)/(len(t)-1-1)
					title = r'$\bar{R}$' + r'$^2$'
					ax.set_title(title + ' = {:.2f}'.format(adjust_R2))					
					ax.plot(rv-np.mean(rv), rv-np.mean(rv), 'k.', alpha = alpha2)				
				if r>0:
					reg = LinearRegression().fit(rv.reshape(-1, 1), ind[r-1,:].reshape(-1, 1))
					score = reg.score(rv.reshape(-1, 1), ind[r-1,:].reshape(-1, 1))
					adjust_R2 = 1-(1-score)*(len(t)-1)/(len(t)-1-1)
					title = r'$\bar{R}$' + r'$^2$'
					ax.set_title(title + ' = {:.2f}'.format(adjust_R2))
					ax.plot(rv-np.mean(rv), ind[r-1,:], 'k.', alpha = alpha2)
				if r!=k_mode:
					ax.tick_params(labelbottom=False)
				else:
					ax.set_xlabel(rv_xlabel)
				ax.yaxis.tick_right()

			if c==2:
				if r==0:
					new_periodogram(t, rv, erv, vlines)
					ax.set_title('Periodogram')
				if r>0:
					new_periodogram(t, ind[r-1,:], eind[r-1,:], vlines)
				if r!=k_mode:
					ax.tick_params(labelbottom=False)
				if r==k_mode:
					ax.set_xlabel(pe_xlabel)

	fig6.align_ylabels(f6_axes[:, 0])
	plt.savefig(file_name)
	plt.show()    
	plt.close('all')
    
    
def weighted_pca(X, X_err, n_pca=None, nor=False):
	'''
	X.shape 	= (n_samples, n_features)
	X_err.shape = (n_samples, n_features)
	nor: normalization = True / False
	'''
	from wpca import WPCA

	weights = 1/X_err
	kwds 	= {'weights': weights}

	# subtract the weighted mean for each measurement type (dimension) of X
	X_new 	= np.zeros(X.shape)
	mean 	= np.zeros(X.shape[1])
	std 	= np.zeros(X.shape[1])

	for i in range(X.shape[1]):
		mean[i] = np.average(X[:,i], weights=weights[:,i])
		std[i] 	= np.average((X[:,i]-mean[i])**2, weights=weights[:,i])**0.5

	for i in range(X.shape[1]):
		if nor == True:
			X_new[:,i] 		= (X[:,i] - mean[i]) / std[i]
			weights[:,i]	= weights[:,i] * std[i] 	# may need to include the Fourier power later
		else:
			X_new[:,i] = X[:,i] - mean[i]

	if n_pca==None:
		n_pca = X.shape[1]

	# Compute the PCA vectors & variance
	pca 		= WPCA(n_components=n_pca).fit(X_new, **kwds)
	pca_score 	= pca.transform(X_new, **kwds)
	P 			= pca.components_

	# The following computes the errorbars
	# The transpose is only intended to be consistent with the matrix format from Ludovic's paper
	X_wpca 	= X_new.T
	C 		= np.zeros(X_wpca.shape)
	err_C 	= np.zeros(X_wpca.shape)

	W 		= weights.T
	P 		= P.T

	for i in range(X_wpca.shape[1]):
		w = np.diag(W[:, i]) ** 2
		C[:, i] = np.linalg.inv(P.T @ w @ P) @ (P.T @ w @ X_wpca[:, i])

		# the error is described by a covariance matrix of C
		Cov_C = np.linalg.inv(P.T @ w @ P)
		diag_C = np.diag(Cov_C)
		err_C[:, i] = diag_C ** 0.5

	P = pca.components_
	err_pca_score = err_C.T

	#---------#
	# results #
	#---------#
	cumulative_variance_explained = np.cumsum(
		pca.explained_variance_ratio_) * 100  # look again the difference calculated from the other way
	print('Cumulative variance explained vs PCA components')
	for i in range(len(cumulative_variance_explained)):
		print(i+1, '\t', '{:.3f}'.format(cumulative_variance_explained[i]))

	for i in range(len(cumulative_variance_explained)):
		if cumulative_variance_explained[i] < 89.5:
			n_pca = i
	n_pca += 2
	if cumulative_variance_explained[0] > 89.5:
		n_pca = 1

	print('{:d} pca scores account for {:.2f}% variance explained'.format(n_pca,
			cumulative_variance_explained[n_pca - 1]))

	print('Standard deviations of each component and the midean uncertainty are\n',
		  np.around(np.std(C[0:n_pca, :], axis=1), decimals=1), '\n',
		  np.around(np.median(err_C[0:n_pca, :], axis=1), decimals=1))

	return P, pca_score, err_pca_score, n_pca
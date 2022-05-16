import matplotlib.pyplot as plt
import numpy as np

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

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
	ax.plot(plot_x[idxx], power[idxx], 'k-', alpha=0.7)
	peaks, _ = find_peaks(power[idxx], height=height)
	ax.plot(plot_x[idxx][peaks], power[idxx][peaks], "ro")

	for n in range(len(plot_x[idxx][peaks])):
		ax.text(plot_x[idxx][peaks][n], power[idxx][peaks][n], '%.1f' % plot_x[idxx][peaks][n], fontsize=10)

	for xc in vlines:
		ax.axvline(x=xc, color='r', linestyle='-', lw=2, alpha=0.3)

	ax.set_xlim([plot_min_t,100])
	ax.set_ylim([0,1.7*height])
	ax.set_xscale('log')

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------    

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
		file_name 	= 'time-series_and_shift_correlation.pdf'

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

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def plot_all_but_corr(k_mode, t, ind, eind, height_ratio, ts_xlabel, pe_xlabel, ind_yalbel, file_name):

	'''
	e.g. 
		k_mode 		= 11
		t 			= bjd_daily
		ind 		= shift_function
		eind 	 	= err_shift_spectrum
		ts_xlabel 	= 'BJD - 2400000 [d]'
		pe_xlabel 	= 'Period [days]'
		ind_yalbel	= 'A'
		file_name 	= 'time-series_and_shift_correlation.png'

	'''
	def new_periodogram(x, y, dy, height_ratio=height_ratio, plot_min_t=2, max_f=1, spp=100):
	
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

		ax.set_xscale('log')

	from sklearn.linear_model import LinearRegression

	# set up the plotting configureations
	alpha1, alpha2 = [0.5,0.2]
	widths 	= [7,7]
	heights = [1]*k_mode
	gs_kw 	= dict(width_ratios=widths, height_ratios=heights)
	plt.rcParams.update({'font.size': 12})
	fig6, f6_axes = plt.subplots(figsize=(16, k_mode), ncols=2, nrows=k_mode, constrained_layout=True,
	                             gridspec_kw=gs_kw)

	# plots 
	for r, row in enumerate(f6_axes):
		for c, ax in enumerate(row):	

			# time-series 
			if c==0:
				ax.errorbar(t, ind[r,:], eind[r,:],  marker='.', ms=5, color='black', ls='none', alpha=alpha1)
				if len(ind_yalbel)==1:
					ax.set_ylabel(ind_yalbel[0] + '$_{' + str(r+1) + '}$')
				else:
					ax.set_ylabel(ind_yalbel[r])
				if r==0:
					ax.set_title('Time-series')
				if r!=(k_mode-1):
					ax.set_xticks([])
				if r==(k_mode-1):
					ax.set_xlabel(ts_xlabel)

			if c==1:
				new_periodogram(t, ind[r,:], eind[r,:])
				if r==0:
					ax.set_title('Periodogram')
				if r!=(k_mode-1):
					ax.set_xticks([])
				if r==(k_mode-1):
					ax.set_xlabel(pe_xlabel)

	fig6.align_ylabels(f6_axes[:, 0])
	plt.savefig(file_name)
	plt.show() 
	plt.close('all')
    
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

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

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def long_short_divide(x, y, yerr, r):
	'''
	x 		= bjd_daily
	y 		= shift_function[i,:]
	yerr 	= err_shift_spectrum[i,:]
	'''

	import george
	from george import kernels

	kernel 	= np.var(y) * kernels.Matern52Kernel(r**2)
	gp 		= george.GP(kernel)
	gp.compute(x, yerr)

	y_pred, _ 	= gp.predict(y, x, return_var=True)
	long_term 	= y_pred
	short_term 	= y - y_pred

	return gp, short_term, long_term

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def lasso_lambda(X, Y, Weight, alphas, n_folds=5, N=100, title='', file_name=None):

	from sklearn.linear_model import Lasso
	from sklearn.model_selection import GridSearchCV
	from sklearn.utils import shuffle
	from datetime import datetime
	from alive_progress import alive_bar

	start_time  = datetime.now()

	alpha_size 	= len(alphas)
	array_size 	= len(Y)

	alpna_n 		= np.zeros(N)
	# coef 			= np.zeros((36,N))
	scores_test 	= np.zeros((alpha_size, N))
	scores_train 	= np.zeros((alpha_size, N))
	scores_test_std = np.zeros((alpha_size, N))
	scores_train_std= np.zeros((alpha_size, N))

	tuned_parameters 	= [{"alpha": alphas}]

	with alive_bar(N) as bar:

		for n in range(N):
			idx_random 	= shuffle(np.arange(array_size))
			X 			= X[idx_random,:]
			Y 			= Y[idx_random]
			sw 			= Weight[idx_random]

			lasso 		= Lasso()
			clf 		= GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_root_mean_squared_error', refit=True, return_train_score=True)
			clf.fit(X, Y, sample_weight=sw)
			# coef[:,n] 	= clf.best_estimator_.coef_
			alpna_n[n] 	= clf.best_estimator_.alpha

			scores_test[:,n] 		= -clf.cv_results_["mean_test_score"]
			scores_train[:,n] 		= -clf.cv_results_["mean_train_score"]
			scores_test_std[:,n]	= clf.cv_results_["std_test_score"] / np.sqrt(n_folds)
			scores_train_std[:,n] 	= clf.cv_results_["std_train_score"] / np.sqrt(n_folds)

			bar()

	scores_test_mean 		= np.mean(scores_test, axis=1)
	scores_train_mean 		= np.mean(scores_train, axis=1)
	scores_test_std_mean 	= np.mean(scores_test_std, axis=1)
	scores_train_std_mean 	= np.mean(scores_train_std, axis=1)

	end_time = datetime.now()
	print('Duration: {}'.format(end_time - start_time))

	# plots
	if file_name != None:
		plt.figure().set_size_inches(8, 6)
		plt.rcParams['font.size'] = '16'
		plt.gcf().subplots_adjust(left=0.15)

		# plot error lines showing +/- std. errors of the scores
		plt.title(title)
		plt.semilogx(alphas, scores_test_mean, 'r', label='testing')
		plt.semilogx(alphas, scores_train_mean, 'b', label='training')
		plt.semilogx(alphas, scores_test_mean + scores_test_std_mean, "r--")
		plt.semilogx(alphas, scores_test_mean - scores_test_std_mean, "r--")
		plt.semilogx(alphas, scores_train_mean + scores_train_std_mean, "b--")
		plt.semilogx(alphas, scores_train_mean - scores_train_std_mean, "b--")	
		plt.fill_between(alphas, scores_test_mean + scores_test_std_mean, scores_test_mean - scores_test_std_mean, color='r', alpha=0.2)
		plt.fill_between(alphas, scores_train_mean + scores_train_std_mean, scores_train_mean - scores_train_std_mean, color='b', alpha=0.2)
		
		xx 	= np.median(alpna_n)
		plt.axvline(xx, color='k', alpha=0.5)
		plt.text(xx, scores_test_mean[alphas==xx], r'$\hat\lambda_1$ = {:.3f}'.format(xx))

		# determine a more robust lambda using the 1-sigma rule
		scores_1sigma = (scores_test_mean+scores_test_std_mean)[alphas==xx][0]
		for i in range(alpha_size-1):
			if (scores_test_mean[i] < scores_1sigma) & (scores_test_mean[i+1] > scores_1sigma):
				alpha_1sigma = alphas[i]
		plt.axvline(alpha_1sigma, color='k', alpha=0.5)			
		plt.axhline(scores_1sigma, linestyle="--", color=".5")
		plt.text(alpha_1sigma, scores_1sigma, r'$\hat\lambda_2$ = {:.3f}'.format(alpha_1sigma))
		plt.ylabel("Residual WRMS [m/s]")
		plt.xlabel(r"$\lambda$")
		plt.axhline(np.min(scores_test_mean), linestyle="--", color=".5")
		plt.xlim([alphas[0], alphas[-1]])
		plt.legend()
		
		plt.savefig(file_name + '.pdf')
		plt.show()
		plt.close()
	
	return scores_test_mean, scores_train_mean, scores_test_std_mean, scores_train_std_mean, alpna_n

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def mlr(feature_matrix, target_vector, etarget_vector, alpha, lag='True', day=5, feature_matrix2=None):

	'''
		Multiple linear regression
		no lag involved 
	'''		

	from sklearn import linear_model
	from sklearn.linear_model import Lasso
	
	regr 	= linear_model.LinearRegression()
	lasso 	= Lasso(alpha=alpha, max_iter=10000).fit(feature_matrix, target_vector, sample_weight=1/etarget_vector**2)

	y_hat 			= lasso.predict(feature_matrix)
	_, w_std_all 	= weighted_avg_and_std(target_vector, 1/etarget_vector**2)
	_, res_wrms 	= weighted_avg_and_std((y_hat - target_vector), weights=1/etarget_vector**2)
	_, model_wrms 	= weighted_avg_and_std(y_hat, weights=1/etarget_vector**2)
	score 			= lasso.score(feature_matrix, target_vector, sample_weight=1/etarget_vector**2)
	n, p 			= len(target_vector), feature_matrix.shape[1]
	adjust_R2 		= 1-(1-score)*(n-1)/(n-p-1)
	score 			= adjust_R2

	if lag=='False':
		k_feature2 	= feature_matrix.shape[1]
	else:
		k_feature2 	= int(feature_matrix.shape[1]/(day+1))
		k_feature 	= int(k_feature2/2)

	w_std = np.zeros(k_feature2)
	for i in range(len(w_std)):
		if lag=='True':
			_, w_std[i] = weighted_avg_and_std(feature_matrix2[:,i], 1/etarget_vector**2)
		else:
			_, w_std[i] = weighted_avg_and_std(feature_matrix[:,i], 1/etarget_vector**2)
	print('Weighted rms is reduced from {:.2f} to {:.2f} (-{:.0f}%); \n\
		Modelled RV weigthed rms = {:.2f};\n\
		Adjusted R squared = {:.3f}.'
			.format(w_std_all, res_wrms, (1-res_wrms/w_std_all)*100, model_wrms, score))

	if lag=='False':

		import pandas as pd 
		df = pd.DataFrame({	'coefficients'		: lasso.coef_, 
							'std'				: w_std,
							'variance'			: (lasso.coef_*w_std)**2,
							'variance percentage': (lasso.coef_*w_std)**2 / sum((lasso.coef_*w_std)**2) * 100
							})	
		print(df.round({'coefficients'	: 2, 
						'std'		: 2,
						'variance'		: 2,
						'variance percentage': 1
						}))
		
		return y_hat, w_std_all, res_wrms, score, df

	else: 

		coeff_matrix = np.zeros((day*2+1, k_feature2))
		for i in range(k_feature):
			coeff_matrix[:,i] = lasso.coef_[i:k_feature*(2*day+1):k_feature]
		coeff_matrix[day,k_feature:]=lasso.coef_[k_feature*(2*day+1):]

		variance_matrix = np.zeros(coeff_matrix.shape)
		for i in range(day*2+1):
			variance_matrix[i,:] = coeff_matrix[i,:]*w_std
		variance_matrix = variance_matrix**2
		variance_matrix = variance_matrix / variance_matrix.sum() * 100

		return y_hat, w_std_all, res_wrms, score, variance_matrix
    
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    
    return (average, np.sqrt(variance))

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def imshow_matrix(coeff_array, score, res_wrms, alpha, k_max, file_name):

	from matplotlib import colors

	'''
		file_name = fwhm_bis_coef
		file_name = fwhm_bis_multi_coef
		file_name = pca_coef
	'''

	day = int((coeff_array.shape[0]-1)/2)
	x = np.arange(day * 2 + 1) - day
	y = np.arange(coeff_array.shape[1]) + 1

	from mpl_toolkits.axes_grid1 import make_axes_locatable
	fig = plt.figure(figsize=(len(x)/1.5+2, len(y)/1.5+1), frameon=False)
	title = r'$\bar{R}$' + r'$^2$'
	plt.title(title + ' = {:.3f}'.format(score)
			  + '   Residual WRMS = {:.2f} m/s'.format(res_wrms))
	plt.xlabel('Lag [days]')
	ax = plt.gca()

	if file_name == 'lasso_coef':
		im2 = plt.imshow(np.transpose(coeff_array),  vmin=-5, vmax=5, cmap=plt.cm.bwr, alpha=.9)
	else:
		im2 = plt.imshow(np.transpose(coeff_array),  cmap=plt.cm.get_cmap('Reds'), alpha=.9, norm=colors.LogNorm(vmin=0.1, vmax=100))

	# Loop over data dimensions and create text annotations.
	for i in range(coeff_array.shape[1]):
		for j in range(day*2+1):
			if coeff_array[j, i] != 0:
				text = ax.text(j, i, '{:.1f}'.format(coeff_array[j, i]),
							   ha="center", va="center", color="k")

	ax.set_xticks(np.arange(len(x)))
	ax.set_yticks(np.arange(len(y)))
	ax.set_xticklabels(x)
	if file_name == 'fwhm_bis_coef':
		ax.set_yticklabels(['FWHM', 'BIS'])
	if file_name == 'Figure/fwhm_bis_multi_coef':
		ax.set_yticklabels(['S-FWHM', 'S-BIS', 'L-FWHM', 'L-BIS'])
	if file_name == 'pca_coef':	
		ax.set_yticklabels([r'PC$_1$', r'PC$_2$', r'PC$_3$'])	
	if file_name == 'Figure/fiesta_multi_coef':
		ax.set_yticklabels([r'S-PC$_1$', r'S-PC$_2$', r'S-PC$_3$', r'L-PC$_1$', r'L-PC$_2$', r'L-PC$_3$'])

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	plt.colorbar(im2, cax=cax, label='Variance percentage') 
	plt.savefig(file_name + '_{:.2f}_{:d}_{:d}'.format(alpha, day, k_max) +'.pdf')
	plt.show()
	plt.close()
    
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def scatter_hist(x, y, xerr, yerr, ax, ax_histx, ax_histy):

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.errorbar(x, y, xerr, yerr, c='black', marker='o', ls='none', alpha=0.2)
    ax.set_xlabel('Model 3 residual [m/s]')
    ax.set_ylabel('Model 6 residual [m/s]')

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='black', alpha=0.5)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color='black', alpha=0.5)
    
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

def plot_soap_config(FEATURE):
    
    N_sample  = 8
    lat       = np.linspace(10, 80, num=N_sample)
    RV_gauss_all = np.zeros((N_sample, 100))
    ΔA_k_all = np.zeros((N_sample, 5, 100))
    Δv_k_all = np.zeros((N_sample, 5, 100))

    for i in range(N_sample):
        print('\nlatitude = ' + str(lat[i]))
        folder_name = 'CCF_PROT=25.05_i=90.00_lon=(180.0,0.0,0.0,0.0)_lat=(' + '%.1f' % lat[i] +  ',0.0,0.0,0.0)_size=(0.1000,0.0000,0.0000,0.0000)'
        DIR     = 'Data/' + FEATURE + '_lat/' + folder_name
        FILE    = sorted(glob.glob(DIR + '/fits/*fits'))
        N       = len(FILE)

        for n in range(N):
            hdulist     = fits.open(FILE[n])
            CCF[:, n]   = hdulist[0].data[v_idx]

        v_k, A_k, RV_gauss = FIESTA(V_grid_SOAP, CCF, eCCF=CCF*0, template=CCF[:,0], k_max=5)
        v_k *= 1000
        RV_gauss = (RV_gauss - RV_gauss[0]) * 1000
        RV_gauss_all[i,:] = RV_gauss

        ΔA_k_all[i,:,:] = np.array([A_k[:,kk] - A_k[:,0] for kk in range(100)]).T

        for k in range(v_k.shape[0]):
            v_k[k,:] -= RV_gauss
        Δv_k_all[i,:,:] = v_k  

    # 
    colors = cm.seismic(np.linspace(0, 1, RV_gauss_all.shape[0]))

    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(5,2,figsize=(8, 6)) 
    plt.subplots_adjust(left = 0.15, hspace=0.1, wspace=0.35, top = 0.95, right = 0.95) 

    phase = np.arange(100)/100
    idx = (phase<=0.8) & (phase>=0.2)

    for i in range(5):

        for j in range(N_sample):
            axes[i,0].plot(phase[idx], ΔA_k_all[j,i,idx], alpha=0.8, color=colors[7-j], label=r'$%d\degree$' %lat[j])
            axes[i,1].plot(phase[idx], Δv_k_all[j,i,idx], alpha=0.8, color=colors[7-j])
        axes[i,0].set_ylabel(r'$k$=%d' %(i+1))
        axes[i,1].set_ylabel('[m/s]')
        axes[0,0].set_title(FEATURE + r' $\Delta A_k$')
        axes[0,1].set_title(FEATURE + r' $\Delta RV_k$')

        if i == 1:            
            axes[i,0].legend(fontsize=6, loc=3)

        for l in range(2):
            if i!=4:
                axes[i,l].set_xticks([])
            else:
                axes[i,l].set_xlabel('Rotation phase') 

    fig.align_ylabels(axes[:,0])
    fig.align_ylabels(axes[:,1])
    plt.savefig('Figure/FIESTA-' + FEATURE + '_latitude.pdf')
    plt.show()

    RV_gauss = RV_gauss.reshape(-1, 1)
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(figsize=(20, 4))
    fig.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.16, right=0.98, top=0.9, wspace=0.4, hspace=0.4)

    for j in range(5):  
        ax = plt.subplot(1, 5, j+1)
        for i in range(RV_gauss_all.shape[0]):
            ax.plot(RV_gauss_all[i,:], Δv_k_all[i,j,:], '.-', alpha=0.8, color=colors[7-i], label=r'$%d\degree$' %lat[i])
            ax.set_xlabel(r'$RV_{Gaussian}$ [m/s]')
            ax.set_ylabel(r'$\Delta RV_{}$ [m/s]'.format(j+1))
            if FEATURE == 'spot':
                if j==0:
                    ax.legend(prop={'size': 10})
            if j==2:
                ax.set_title('Solar '+ FEATURE)
    plt.savefig('Figure/FIESTA-' + FEATURE + '_latitude_correlation.pdf')
    plt.show()    
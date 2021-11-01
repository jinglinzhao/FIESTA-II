import numpy as np
from scipy.optimize import curve_fit
import copy
import matplotlib.pyplot as plt

# ------------------------------------------
# Gaussian function
# ------------------------------------------
def gaussian(x, amp, mu, sig, c):
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c

# ------------------------------------------
# Discrete Fourier transform (DFT)
# ------------------------------------------
# https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft
# freq returns only the positive frequencies 
def FT(signal, spacing):
	n 			= signal.size
	fourier 	= np.fft.rfft(signal, n)
	ξ 		= np.fft.rfftfreq(n, d=spacing)
	A 			= np.abs(fourier)
	ϕ 		= np.angle(fourier)
	return [A, ϕ, ξ]

# ------------------------------------------
# Wrap the phase
# ------------------------------------------
def wrap(array):
'''	
	An individual phase ranges within (-np.pi, np.pi)
	The difference of two phases ranges within (-2*np.pi, 2*np.pi)
	Adding a phase of multiples of 2*np.pi is effectively the same phase
	This function wraps the phase difference such that it lies within (-np.pi, np.pi)
'''	
	for i in np.arange(len(array)):
		array[i] = array[i] - int(array[i]/np.pi) * 2 * np.pi
	return array

#==============================================================================
# 
# FourIEr phase SpecTrum Analysis (FIESTA)
# 
#==============================================================================

def FIESTA(V_grid, CCF, eCCF, template=[], SNR=2, k_max=None):
	'''
		V_grid: 	1D velocity grid array (N_v,)
		CCF: 		CCF array containing all the files (N_v, N_file)
		eCCF:		error associated with the CCF array for all the files (N_v, N_file)
		template: 	CCF template (N_v,)
		SNR:		SNR of individual A, ϕ and their time-series
		k_max: 		number of modes for the output; if not specified, the modes up to the noise limit will be calculated
	'''

	N_file 	= CCF.shape[1]
	spacing = np.diff(V_grid)[0]

	# ------------------------------------------
	# Construct a template
	# ------------------------------------------
	# If template is given, use the template.
	if template!=[]:
		tpl_CCF = template
	# If template is not given but eCCF is given, calculate the weighted average. - usually the case
	elif ~np.all(eCCF == 0):
		tpl_CCF = np.average(CCF, axis=1, weights=1/eCCF**2)
	# If no template is given and eCCF = 0, use the first file as the template.
	else:
		tpl_CCF = CCF[:,0]

	# ------------------------------------------
	# Define the range of the CCF
	# ------------------------------------------
	'''
		Choose the "interesting" part of V_grid for analysis.
		The following chooses the range up to 5-sigma.
	'''
	popt, pcov 		= curve_fit(gaussian, V_grid, tpl_CCF, p0=[0.5, (max(V_grid)+min(V_grid))/2, 1, 0])
	sigma 			= popt[2]
	V_centre 		= popt[1]
	V_min, V_max 	= V_centre - 5*sigma, V_centre + 5*sigma

	# reshape all input spectra 
	idx 	= (V_grid>V_min) & (V_grid<V_max)
	tpl_CCF = tpl_CCF[idx]
	V_grid 	= V_grid[idx]
	CCF 	= CCF[idx,:]
	eCCF 	= eCCF[idx,:]

	print('\nVelocity grid used [{:.2f}, '.format(min(V_grid)) + '{:.2f}]\n'.format(max(V_grid)))

	# Information of the template CCF 
	A_tpl, ϕ_tpl, ξ = FT(tpl_CCF, spacing)
	popt, pcov 		= curve_fit(gaussian, V_grid, tpl_CCF, p0=[0.5, (max(V_grid)+min(V_grid))/2, 1, 0])
	RV_gauss_tpl 	= popt[1]

	# ------------------------------------------
	# Amplitude and phase uncertainties 
	# ------------------------------------------
	
	A_spectrum 		= np.zeros((ξ.size, N_file))
	ϕ_spectrum 		= np.zeros((ξ.size, N_file))
	v_spectrum 		= np.zeros((ξ.size-1, N_file))
	σ_A 			= np.zeros(N_file)
	σ_ϕ 			= np.zeros((ξ.size, N_file))
	RV_gauss 		= np.zeros(N_file)
	ξ_normal_array 	= []

	for n in range(N_file):
		
		# DFT
		A, ϕ, ξ 		= FT(CCF[:,n], spacing)
		A_spectrum[:,n] = A
		ϕ_spectrum[:,n]	= ϕ

		Δϕ 				= wrap(ϕ - ϕ_tpl)
		v_spectrum[:,n] = -Δϕ[1:] / (2 *np.pi*ξ[1:])

		## RV measured as centroid of a Gaussian fit.			
		popt, pcov 	= curve_fit(gaussian, V_grid, CCF[:,n], p0=[0.5, (max(V_grid)+min(V_grid))/2, 1, 0])
		RV_gauss[n] = popt[1] - RV_gauss_tpl

		# calculate the uncertainties 
		# The 1-sigma uncertainty for the amplitude is the same for all k's
		σ 		= (sum(eCCF[:,n]**2)/2)**0.5
		σ_A[n]	= σ
		σ_ϕ[:,n]= σ/A

		# ξ_normal
		ξ_normal_array.append(max(ξ[0.2*A > σ]))

	# ------------------------------------------
	# proceed the following only if noise is present 	
	# ------------------------------------------
	if ~np.all(eCCF == 0):

		# --------------------
		# ξ_normal
		# --------------------
		'''
			Determine the frequency 
			beyond which noise in the Fourier domain 
			deviate from normal distribution.
		'''	
		ξ_normal = np.median(ξ_normal_array)
		print('ξ_normal = {:.2f}\n'.format(ξ_normal))


		# --------------------
		# ξ_individual
		# --------------------
		'''
			Make sure SNR of individual amplitudes and phases 
			are larger than the given SNR (usually 2)
		'''
		individual_SNR 	= np.zeros(ξ.size)
		
		for i in range(ξ.size):
			individual_SNR[i] 	= np.median(A_spectrum[i,:] / σ_A)

		ξ_individual = max(ξ[individual_SNR>SNR])


		# --------------------
		# ξ_time-series
		# --------------------
		'''
			Signal: the variation of the time-series.
			Noise: 	the median uncertainty of individual measurement.
			The standard deviation of a time-series is at least twice 
			as large as the median uncertainty of individual measurement.
		'''
		ts_SNR_A 	= np.std(A_spectrum, axis=1) / np.median(σ_A)
		ξ_A 		= ξ[ts_SNR_A > SNR]

		ts_SNR_ϕ 	= np.std(ϕ_spectrum, axis=1) / np.median(σ_ϕ,axis=1)
		ξ_ϕ			= ξ[ts_SNR_ϕ > SNR]
		ξ_ts		= min(max(ξ_A), max(ξ_ϕ))


		# --------------------
		# residuals
		# --------------------
		res_rms 	= np.zeros(ξ.size)
		ft 			= np.fft.rfft(tpl_CCF, tpl_CCF.size)
		for i in range(ft.size):
			pseudo_ft = copy.copy(ft)
			if i < (ft.size-1):
				pseudo_ft[(i+1):] = 0
			pseudo_ift = np.fft.irfft(pseudo_ft, len(tpl_CCF))
			res_rms[i] = np.std(tpl_CCF - pseudo_ift)


		# --------------------
		# ξ_modelling_noise
		# --------------------
		print('\nThe median SNR of all CCFs is {:.0f}'.format(np.median((1-CCF)/eCCF)))
		ξ_modelling_noise = min(ξ[1/res_rms > np.median((1-CCF)/eCCF)])
		print('ξ_modelling_noise = {:.2f}\n'.format(ξ_modelling_noise))


		# --------------------
		# ξ_FIESTA
		# --------------------
		ξ_FIESTA 	= min(ξ_normal, ξ_individual, ξ_ts, ξ_modelling_noise)
		if k_max == None:
			k_max 		= len(ξ[ξ<=ξ_FIESTA])-1
		else:
			k_max = k_max

		print('\nBased on the user-defined SNR = {:.1f}:'.format(SNR))
		print('ξ_individual = {:.2f}\nξ_timeseries = {:.2f}\n'
				.format(ξ_individual, ξ_ts))
		print('In summary, the cut-off frequency for FIESTA is recommended to be {:.2f} ({:d} frequency modes)\n'
				.format(ξ_FIESTA, len(ξ[ξ<=ξ_FIESTA])-1))
		

		import pandas as pd 
		df = pd.DataFrame({	'ξ'					: ξ, 
							'individual_SNR'	: individual_SNR,
							'ts_SNR_A'			: ts_SNR_A, 
							'ts_SNR_ϕ'			: ts_SNR_ϕ,
							'modelling noise'	: res_rms,
							'recoverable_CCF_SNR': [round(np.median(1-tpl_CCF)/res_rms) for res_rms in res_rms]
							})

		print(df.round({	'ξ'					: 3, 
							'individual_SNR'	: 1,
							'ts_SNR_A'			: 1, 
							'ts_SNR_ϕ'			: 1,
							'modelling noise'	: 5,
							'recoverable_CCF_SNR': 0
							}))


		σ_v_spectrum 	= (σ_ϕ[1:,:].T / (2*np.pi*ξ[1:].T)).T 

	'''
		If noise is present in the input spectra (as is normally the case),
		return the FIESTA outputs with error estimates.
		Otherwise (e.g. for quicker results), 
		return the FIESTA outputs without error estimates.
	'''
	if ~np.all(eCCF == 0):
		return df, v_spectrum[:k_max,:], σ_v_spectrum[:k_max,:], A_spectrum[1:k_max+1,:], np.vstack([σ_A]*k_max), RV_gauss
	else: 
		return v_spectrum[:k_max,:], A_spectrum[1:k_max+1,:], RV_gauss
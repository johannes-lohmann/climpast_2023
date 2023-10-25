import matplotlib.pyplot as pl
import numpy as np
import matplotlib as mpl
from random import randrange
from sklearn.neighbors import KernelDensity
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
#from pylab import *
from pylab import polyval, linspace, percentile, poly1d, plot
from scipy.signal import detrend
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from panel_labels import letter_subplots

pl.rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size'   : 14})

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

mpl.rcParams['axes.xmargin'] = 0.03
mpl.rcParams['axes.ymargin'] = 0.03

mpl.rcParams['axes.unicode_minus'] = False

def MAIN():

	volcs = np.loadtxt('bipolar_volcanos_published.txt')
	N=len(volcs)
	mask_nh = np.loadtxt('mask_nh.txt')
	### mask where 1 means eruption happens in interstadial
	mask_gi = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
	
	### Load eruption ages
	# GICC05 times of all volcanic eruptions in individual records taking into account the actual depths of the depositions.
	# In case no depth was available, the NGRIP depth and correponsing GICC05 age is given (filtered out later)
	volcs_neem_all = np.loadtxt('bipolar_volcanos_neem.txt')
	volcs_grip_all = np.loadtxt('bipolar_volcanos_grip.txt')
	volcs_gisp2_all = np.loadtxt('bipolar_volcanos_gisp2.txt')
	
	volcs_edc_all = np.loadtxt('bipolar_volcanos_edc.txt')
	volcs_edml_all = np.loadtxt('bipolar_volcanos_edml.txt')
	### alternative ages.
	#volcs_edml_all = np.loadtxt('bipolar_volcanos_edml_jiamei.txt')
	volcs_wais_all = np.loadtxt('bipolar_volcanos_wais.txt')
	### alternative ages on WD2014 age scale.
	#volcs_wais_all = np.loadtxt('bipolar_volcanos_wais_wd2014.txt')
	N=len(volcs)
	
	
	### Mask eruptions that have no identified depth in the individual cores.
	mask_neem = np.loadtxt('neem_volc_mask.txt')
	mask_grip = np.loadtxt('grip_volc_mask.txt')
	mask_gisp2 = np.loadtxt('gisp2_volc_mask.txt')
	mask_edc = np.loadtxt('edc_volc_mask.txt')
	mask_edml = 82*[1]
	mask_edml[54] = 0###eruption that falls in data gap 43.7-45ka
	### eruptions not in LIN22 unipolar list for EDML
	#mask_edml[0] = 0
	#mask_edml[4] = 0
	#mask_edml[14] = 0
	#mask_edml[15] = 0
	#mask_edml[28] = 0
	#mask_edml[47] = 0
	#mask_edml[58] = 0
	#mask_edml[62] = 0
	#mask_edml[66] = 0
	#mask_edml[74] = 0
	mask_ngrip = 82*[1]

	### 0 means eruptions that do not have an identified depth in the respective core.
	volcs_grip=[]; volcs_grip0=[]
	volcs_gisp2=[]; volcs_gisp20=[]
	volcs_neem=[]; volcs_neem0=[]

	for i in range(N):
		if mask_neem[i]==1:
		        volcs_neem.append(volcs[i])
		else:
		        volcs_neem0.append(volcs[i])

		if mask_gisp2[i]==1:
		        volcs_gisp2.append(volcs[i])
		else:
		        volcs_gisp20.append(volcs[i])

		if mask_grip[i]==1:
		        volcs_grip.append(volcs[i])
		else:
		        volcs_grip0.append(volcs[i])
	
	print('Eruptions with depth in GRIP: ', len(volcs_grip))
	print('Eruptions with depth in GISP2: ', len(volcs_gisp2))
	print('Eruptions with depth in NEEM: ', len(volcs_neem))
	
	
	evts1=0; evts2=0; evts3=0; evts4=0
	idcs_all = []
	for i in range(N):
		if (mask_gisp2[i] + mask_grip[i] +mask_neem[i])==0:
			evts1+=1
		elif (mask_gisp2[i] + mask_grip[i] +mask_neem[i])==1:
			evts2+=1
		elif (mask_gisp2[i] + mask_grip[i] +mask_neem[i])==2:
			evts3+=1
		elif (mask_gisp2[i] + mask_grip[i] +mask_neem[i])==3:
			evts4+=1
			idcs_all.append(i)
			
	print('Eruptions in all 4 Greenland cores: ', evts4)
	print('Eruptions in 3 Greenland cores: ', evts3)
	print('Eruptions in 2 Greenland cores: ', evts2)
	print('Eruptions in NGRIP only: ', evts1)


	### Load and cut the oxygen isotope records.
	
	#t_ngrip2, ngrip2 = np.loadtxt('ngrip_d18o_1y_test.txt')
	t_ngrip, ngrip = np.loadtxt('ngrip_d18o_1y_11k.txt')

	#t_grip2, grip2 = np.loadtxt('grip_d18o_1y_bottom.txt')
	t_grip, grip = np.loadtxt('grip_d18o_1y_11k.txt')

	t_gisp2, gisp2 = np.loadtxt('gisp2_d18o_1y.txt')
	t_neem, neem = np.loadtxt('neem_d18o_1y_bottom_corr.txt')
	t_neem = t_neem[11660:-350]; neem = neem[11660:-350]#[14761:-350]
	t_grip = t_grip[:-350]; grip = grip[:-350]
	t_gisp2 = t_gisp2[55:]; gisp2 = gisp2[55:]
	t_ngrip = t_ngrip[:91950]; ngrip=ngrip[:91950]
	t_neem = t_neem[:91950]; neem=neem[:91950]

	t_wais, wais = np.loadtxt('wais_d18o_highres_1y_volc.txt')
	### alternative WAIS record on WD2014 age scale
	#t_wais, wais = np.loadtxt('wais_d18o_highres_1y_wd2014.txt')
	t_wais=t_wais[11677:]; wais=wais[11677:]
	
	t_edml, edml = np.loadtxt('edml_d18o_1y_volc.txt')
	t_edml=t_edml[11668:]; edml=edml[11668:]

	t_edc, edc = np.loadtxt('edc_d18o_1y_volc.txt')
	t_edc=t_edc[11600:]; edc=edc[11600:]

	t, stack = np.loadtxt('greenland_d18o_stack_11k.txt')


	### Load the DO onset and termination times.
	stack_terminations = np.loadtxt('stack_downtimings.txt')
	stack_terminations = stack_terminations[8:-1]; stack_terminations = stack_terminations[::-1]
	stack_terminations=np.append(12826.,stack_terminations)#12896.
	stack_terminations=np.append(stack_terminations,60000.)
	stack_onsets = np.loadtxt('stack_onsets.txt')[::-1]
	stack_onsets=np.append(stack_onsets,61000.)
	stack_onsets=np.append(11670.,stack_onsets)
	
	### mask to filter out events shortly before DO onsets
	idcs_DO=[9, 22, 24, 40, 44, 64, 71]
	mask_do = np.asarray(len(volcs)*[1.0])
	mask_do[idcs_DO] = 0.0
	
	cores = ['NGRIP', 'GRIP', 'GISP2', 'NEEM', 'Stack', 'WAIS', 'EDML', 'EDC']
	records = {'NGRIP': ngrip, 'GRIP': grip, 'GISP2': gisp2, 'NEEM': neem, 'Stack': stack, 'WAIS':wais, 'EDML': edml, 'EDC': edc}
	volcano_times = {'NGRIP': volcs, 'GRIP': volcs_grip_all, 'GISP2': volcs_gisp2_all, 'NEEM': volcs_neem_all, 'Stack': volcs, 'WAIS': volcs_wais_all, 'EDML': volcs_edml_all, 'EDC': volcs_edc_all}
	masks = {'NGRIP': mask_ngrip, 'GRIP': mask_grip, 'GISP2': mask_gisp2, 'NEEM': mask_neem, 'Stack': mask_ngrip, 'WAIS': mask_ngrip, 'EDML': mask_edml, 'EDC': mask_edc}
	### shift of eruption time based on analysis of sulfate spikes, to align 
	### to (average) true eruption start
	shifts = {'NGRIP': -.5-1.5, 'GRIP': -1.5, 'GISP2': -.5-1.5, 'NEEM': -2.-1.5, 'Stack': -1.5, 'WAIS': -0.7-1.5, 'EDML': -1.5, 'EDC': 1.-1.5}


	### Calculate average response curves, and response curves for GI and GS subsets,
	### as well as response curves for younger and older halfs.
	### For ALL cores.
	
	lead = 10 ### time before eruption where averaging is stopped
	M = 50 ### length of detrended segment (times two)
	### a and b are used to define the segment around the eruption for average isotope resp.
	### not used for the response curves.
	### NEEDS TO BE ADJUSTED FOR THE DIFFERENT CORES DEPENDING ON RESPONSE CURVE
	### k is for the lead time (before b)
	#a = 3; b = 3
	a = 5; b = 1 ### for Stack
	#a = 7; b = 7 ### for EDC
	#a = 2; b = 2 ### for WAIS: year of eruption, one before, two after.
	k = 50

	responses = {}; responses_gi = {}; responses_gs = {}; means = {}; resp_curves = {}; p14 = {}; p86 = {}
	resp_curves_gi = {}; resp_curves_gs = {}; resp_curves_old = {}; resp_curves_young = {}
	for name in cores:
		resp, means0 = calc_response(volcano_times[name], t_ngrip, stack_terminations, stack_onsets, records[name], N, M, a, b, lead, k, shifts[name])
		responses[name] = resp ; means[name] = means0
		### for average response curve and percentiles, filter out eruptions with no depth in respective cores.
		resp_mask = []; resp_gi = []; resp_gs = []
		for i in range(len(resp)):
			if masks[name][i]==1:
				resp_mask.append(resp[i])
				if mask_gi[i]==1:
					resp_gi.append(resp[i])
				else:
					resp_gs.append(resp[i])
					
		responses_gs[name] = resp_gs; responses_gi[name] = resp_gi
		print(len(resp_mask), len(resp_mask[0]), len(resp_gi), len(resp_gs))
		
		resp_young = resp_mask[:int(len(resp_mask)/2)]; resp_old = resp_mask[int(len(resp_mask)/2):]
		
		resp_curve, perc14, perc86 = mean_confidence_band(np.asarray(resp_mask))
		resp_curves[name] = np.asarray(resp_curve); p14[name] = perc14; p86[name] = perc86
		resp_curves_gi[name] = mean_curve(np.asarray(resp_gi))
		resp_curves_gs[name] = mean_curve(np.asarray(resp_gs))
		resp_curves_old[name] = mean_curve(np.asarray(resp_old))
		resp_curves_young[name] = mean_curve(np.asarray(resp_young))

	
	
	### response curve of Greenland STACK is RECALCULATED, so that for each eruption 
	### only the cores count that actually have an identified depth.
	
	responses_stacked = [(responses['NGRIP'][i] + mask_grip[i]*responses['GRIP'][i] + mask_gisp2[i]*responses['GISP2'][i] + mask_neem[i]*responses['NEEM'][i])/(1.+ mask_neem[i] + mask_grip[i] + mask_gisp2[i]) for i in range(len(volcs))]
	
	resp_stack_gi = []; resp_stack_gs = []
	resp0 = 0; resp1 = 0
	resp_test1 = 0.
	for i in range(len(volcs)):
		resp_curv0 = (responses['NGRIP'][i] + mask_grip[i]*responses['GRIP'][i] + mask_gisp2[i]*responses['GISP2'][i] + mask_neem[i]*responses['NEEM'][i])/(1.+ mask_neem[i] + mask_grip[i] + mask_gisp2[i])
		print(responses['NGRIP'][i][M], responses['GRIP'][i][M], resp_curv0[M], mask_grip[i], mask_gisp2[i], mask_neem[i])
		resp0+=resp_curv0[M]; resp1+=resp_curv0[M-1]
		
		resp_test1 += mask_neem[i]*responses['NEEM'][i][M-1]
		
		if mask_gi[i]==1:
			resp_stack_gi.append(responses_stacked[i])
		else:
			resp_stack_gs.append(responses_stacked[i])
	
	
	print(resp0/82., resp1/82., resp_test1/61.)
	
	means_stacked = [(means['NGRIP'][i] + mask_grip[i]*means['GRIP'][i] + mask_gisp2[i]*means['GISP2'][i] + mask_neem[i]*means['NEEM'][i])/(1.+mask_neem[i]+ mask_grip[i] +mask_gisp2[i]) for i in range(len(volcs))]
	
	resp_curve_stacked, p14_stacked, p86_stacked = mean_confidence_band(np.asarray(responses_stacked))
	
	resp_curve_stacked_alt = mean_curve(np.asarray([resp_curves['NGRIP'], resp_curves['GRIP'], resp_curves['GISP2'], resp_curves['NEEM']]))
	resp_curve_stacked_gi_alt = mean_curve(np.asarray([resp_curves_gi['NGRIP'], resp_curves_gi['GRIP'], resp_curves_gi['GISP2'], resp_curves_gi['NEEM']]))
	resp_curve_stacked_gs_alt = mean_curve(np.asarray([resp_curves_gs['NGRIP'], resp_curves_gs['GRIP'], resp_curves_gs['GISP2'], resp_curves_gs['NEEM']]))
	
	resp_curve_stacked_young_alt = mean_curve(np.asarray([resp_curves_young['NGRIP'], resp_curves_young['GRIP'], resp_curves_young['GISP2'], resp_curves_young['NEEM']]))
	resp_curve_stacked_old_alt = mean_curve(np.asarray([resp_curves_old['NGRIP'], resp_curves_old['GRIP'], resp_curves_old['GISP2'], resp_curves_old['NEEM']]))
	
	resp_curves_gi['Stack'] = resp_curve_stacked_gi_alt#mean_curve(np.asarray(resp_stack_gi))
	resp_curves_gs['Stack'] = resp_curve_stacked_gs_alt#mean_curve(np.asarray(resp_stack_gs))
	
	resp_curves_old['Stack'] = resp_curve_stacked_old_alt#mean_curve(np.asarray(responses_stacked[int(len(responses_stacked)/2):]))
	resp_curves_young['Stack'] = resp_curve_stacked_young_alt#mean_curve(np.asarray(responses_stacked[:int(len(responses_stacked)/2)]))
	
	means['Stack'] = means_stacked
	p14['Stack'] = p14_stacked
	p86['Stack'] = p86_stacked
	
	resp_curves['Stack'] = resp_curve_stacked_alt#resp_curve_stacked
	
	print(resp_curves['NGRIP'][M-1], resp_curves['NEEM'][M-1], resp_curves['GRIP'][M-1], resp_curves['GISP2'][M-1])
	
	resp_total = 82.*resp_curves['NGRIP'][M-1] + 61.*resp_curves['NEEM'][M-1] + 59.*resp_curves['GRIP'][M-1] + 65.* resp_curves['GISP2'][M-1]
	print(resp_total/(82.+61.+59.+65.))
	
	print('---------- Average bipolar response curves ---------------')
	for name in cores:
		print(name)
		print(resp_curves_old[name][k-5:k+5])#resp_curves
	
	print('Average isotopic anomaly Stack: ', np.mean(means['Stack']))
	print('Average isotopic anomaly NGRIP: ', np.mean(means['NGRIP']))
	print('Average isotopic anomaly WAIS: ', np.mean(means['WAIS']))
	print('Average isotopic anomaly EDC: ', np.mean(means['EDC']))
	
	fig, axes = pl.subplots(3,3, figsize=(12.,9))
	letter_subplots(letters='a', yoffset=1.05)
	pl.subplots_adjust(left=0.1, bottom=0.07, right=0.97, top=0.96, wspace=0.24, hspace=0.27)
	i=0
	for name in cores:
		print(int(i/3)%3, i%3)
		axes[int(i/3)%3][i%3].fill_between(range(-k,M), p14[name], p86[name] ,color='gray',alpha=0.25)
		axes[int(i/3)%3][i%3].bar(range(-k,M), resp_curves[name], color='royalblue', width=1., label=name)
		axes[int(i/3)%3][i%3].grid(axis='x', which='both'); axes[int(i/3)%3][i%3].grid(axis='y', which='both')
		
		axes[int(i/3)%3][i%3].plot(range(-M,M), resp_curves_gi[name], color='darkorange', linewidth=0.8, label='GI')
		axes[int(i/3)%3][i%3].plot(range(-M,M), resp_curves_gs[name], color='lightseagreen', linewidth=0.8, label='GS')
		
		#axes[int(i/3)%3][i%3].plot(range(-M,M), resp_curves_old[name], color='gold', linewidth=0.8, label='Older')
		#axes[int(i/3)%3][i%3].plot(range(-M,M), resp_curves_young[name], color='orangered', linewidth=0.8, label='Younger')
		axes[int(i/3)%3][i%3].set_ylabel('$\delta^{18}$O anomaly')
		axes[int(i/3)%3][i%3].set_xlabel('Time before eruption (years)')
		axes[int(i/3)%3][i%3].set_xlim(-25,40)
		#axes[int(i/3)%3][i%3].set_ylim(-1.5,1.2)
		if i==0:
			axes[int(i/3)%3][i%3].legend(loc='upper right', fontsize=11)
		i+=1


	
	
	
	
	### GI vs GS response for Bipolar eruptions in NGRIP and Stack.
	### including resampling such that the sulfate deposition magnitude
	### distributions of GS and GI samples match.
	
	### match times from Svensson 2020 and Lin 2022
	### NGRIP
	volcs_nh, mag_nh = np.loadtxt('magnitude_age_nh_9k_publ.txt', unpack=True)
	volcs_bi_nh_ages = np.loadtxt('bipolar_volcanos_published.txt')
	idcs=np.argwhere(np.isin(volcs_bi_nh_ages,[12961., 33328., 43327., 59180.])) 
	## event at 43327 is much earlier in NH records. Too lazy to do manually...
	volcs_bi_nh_ages = np.delete(volcs_bi_nh_ages, idcs)
	volcs_bi_nh, mag_bi_nh, volcs_uni_nh, mag_uni_nh, mask_gi_lin22 = extract_bipolar(volcs_bi_nh_ages, volcs_nh, mag_nh, mask_gi)
	
	### EDC
	#volcs_sh, mag_sh = np.loadtxt('magnitude_age_sh_9k_publ.txt', unpack=True)
	#volcs_bi_sh_ages = np.loadtxt('bipolar_volcanos_edc.txt', unpack=True)
	#volcs_bi_sh, mag_bi_sh, volcs_uni_sh, mag_uni_sh, mask_gi_lin22 = extract_bipolar(volcs_bi_sh_ages, volcs_sh, mag_sh, mask_gi)
	
	volcs_bi_sh, mag_bi_sh = np.loadtxt('bipolar_volcanos_edc_avg.txt', unpack=True)
	mask_gi_lin22 = []
	for i in range(len(volcs_edc_all)):
		idx = find_nearest(volcs_bi_sh, volcs_edc_all[i])
		print(i, volcs_bi_sh[idx], volcs_edc_all[i], mask_gi[i])
		if np.abs(volcs_bi_sh[idx]-volcs_edc_all[i])<0.01:
			mask_gi_lin22.append(mask_gi[i])
			
	print('------ Bipolar eruption ages -----------')
	print(volcs_bi_sh)
	
	print(mask_gi)
	print(mask_gi_lin22)
	
	print(len(mask_gi), len(mask_gi_lin22))
	print(len(volcs_bi_sh), len(mask_gi_lin22))
	
	volcs_gi = []; volcs_gs = []; mag_gi = []; mag_gs = []
	for i in range(len(volcs_bi_sh)):#volcs_bi_nh
		if mask_gi_lin22[i]==1:
			volcs_gi.append(volcs_bi_sh[i])#volcs_bi_nh
			mag_gi.append(mag_bi_sh[i])#mag_bi_nh
		else:
			volcs_gs.append(volcs_bi_sh[i])
			mag_gs.append(mag_bi_sh[i])
		
	
	print(volcs_gi)
	print(volcs_gs)
	
	### Get resample response curves, from resampling so that magnitude
	### distributions match for GS and GI.
	#resp_curve_resampled = response_mag_match(volcs_gi, volcs_gs, mag_gi, mag_gs, M, a, b, k, t_ngrip, records['NGRIP'], stack_terminations, stack_onsets, shifts['NGRIP'])
	#resp_curve_resampled = response_mag_match(volcs_gi, volcs_gs, mag_gi, mag_gs, M, a, b, k, t_ngrip, records['EDC'], stack_terminations, stack_onsets, shifts['EDC'])
	
	fig=pl.figure(figsize=(5.,3.5))
	#letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	#pl.plot(range(-M,M), resp_curves_gi['NGRIP'], color='black', label='GI')
	#pl.plot(range(-M,M), resp_curves_gs['NGRIP'], color='tomato', label='GS')
	
	pl.plot(range(-M,M), resp_curves_gi['EDC'], color='black', label='GI')
	pl.plot(range(-M,M), resp_curves_gs['EDC'], color='tomato', label='GS')
	#pl.plot(range(-M,M), resp_curve_resampled, '--', color='royalblue', label='Resampled')
	pl.xlim(-30,40); pl.legend(loc='best'); 
	pl.ylabel('$\delta^{18}$O anomaly')
	pl.xlabel('Time before eruption (years)')
	
	
	### Plot distributions of the average volcanic anomalies. 
	#hypothesis_test(volcs, means['Stack'], stack, t_ngrip, 'Stack', a, b, lead, mask_gi, mask_nh)
	#hypothesis_test(volcs, means['NGRIP'], ngrip, t_ngrip, 'NGRIP', a, b, lead, mask_gi, mask_nh)
	#hypothesis_test(volcs, means['WAIS'], wais, t_ngrip, 'WAIS', a, b, lead, mask_gi, mask_nh)
	hypothesis_test(volcs, means['EDC'], edc, t_ngrip, 'EDC', a, b, lead, mask_gi, mask_nh)
	
	
	### Evaluate CORRELATION of response and MAGNITUDE.
	
	volcs2, so4, _, _, mag_green, mag_antarc = np.loadtxt('bipolar_loading_publ.txt', unpack=True)
	
	print('baaaaarrrrrrrr')
	print(len(so4))
	
	correlation_sulfur_response(volcs, volcs2, means['Stack'], np.log(so4), responses_stacked, resp_curves['Stack'])#np.log(mag_green)
	
	correlation_sulfur_response(volcs, volcs2, means['EDC'], np.log(mag_antarc), responses['EDC'], resp_curves['EDC'])#np.log(so4)
	
	#correlation_sulfur_response(volcs, volcs2, means['WAIS'], np.log(mag_green), responses['WAIS'], resp_curves['WAIS'])#np.log(so4)
	
	responses_latitude(volcs, responses_stacked, mask_nh, M, resp_curve_stacked_alt)
	#responses_latitude(volcs, responses['EDC'], mask_nh, M, resp_curve_stacked_alt)
	#responses_latitude(volcs, responses['NGRIP'], mask_nh, M)
	#responses_latitude(volcs, responses['WAIS'], mask_nh, M)
	
	
	### CORRELATION of average anomaly of bipolar eruptions for different CORES.
	
	#_, means_wais = calc_response(volcano_times['WAIS'], t_ngrip, stack_terminations, stack_onsets, records['WAIS'], N, M, 2, 2, lead, k, shifts['WAIS'])
	#_, means_edc = calc_response(volcano_times['EDC'], t_ngrip, stack_terminations, stack_onsets, records['EDC'], N, M, 7, 7, lead, k, shifts['EDC'])
	
	#correlation_greenland_antarctica(means['Stack'], means_wais, means_edc)
	
	#correlation_cores(means, ['NGRIP', 'GRIP', 'NEEM', 'GISP2'], masks)
	
	#pca_cores(volcs, idcs_all, means, mask_nh)
	
	#diffusion(volcs, responses, M)
	
	#scan_transitions(volcs, responses['Stack'], N, M, stack_onsets, stack_terminations)'
	
	'''
	fig=pl.figure(figsize=(6,4.5))
	pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98)
	pl.fill_between(range(-M,M), p14['Stack'], p86['Stack'] ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_curves['Stack'], color='black');pl.grid(axis='both')
	pl.plot(range(-M,M), resp_curves_old['Stack'], color='gold', label='Older')
	pl.plot(range(-M,M), resp_curves_young['Stack'], color='orangered', label='Younger')

	pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-30,50); pl.legend(loc='best')
	pl.xlabel('Time before eruption (years)')

	

	idcs_DO=[9, 22, 24, 40, 44, 64, 71]
	resp_stackDO, stack_perc14DO, stack_perc86DO = mean_confidence_band(resp_stack_all[idcs_DO])

	fig=pl.figure()
	pl.suptitle('Volcanos before DO events')
	#for idx in idcs_DO:
	#	pl.plot(range(-M,M), resp_stack_all[idx]);pl.grid(axis='both')
	pl.fill_between(range(-M,M),stack_perc14DO,stack_perc86DO ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_stackDO);pl.grid(axis='both')
	#pl.text(0.95,0.95,'Stack',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)');pl.ylabel('$\delta^{18}$O anomaly')
	'''
	
	#fig=pl.figure()
	#pl.fill_between(range(-M,M), p14['Stack'], p86['Stack'] ,color='gray',alpha=0.25)
	#pl.bar(range(-M,M), resp_curves['Stack']);pl.grid(axis='both')
	#pl.ylabel('Stack $\delta^{18}$O anomaly'); pl.xlabel('Time before eruption (years)')
	#pl.xlim(-30,50)


	pl.show()


def correlation_greenland_antarctica(stack, wais, edc):
	### compare isotopic response of Greenland stack to WAIS and EDC
	### and also compare WAIS and EDC.
	### Use different averaging times/segments. WAIS a=b=2; EDC a=b=7; stack a=5, b=1
	
	### FILTER OUT the ones in EDC with no depth? Only 4...
	
	print(len(stack), len(wais), len(edc))
	
	fig=pl.figure()
	ax=pl.subplot(131)
	xi, y_fit, lower, upper = lin_regr_uncertainty(stack, wais)
	rp = spearmanr(stack, wais)[0]
	pl.plot(stack, wais, 'o', color='black')
	pl.plot(xi, y_fit, color='tomato')
	pl.plot(xi, lower, '--')
	pl.plot(xi, upper, '--', color='cornflowerblue')
	pl.text(0.35,0.97, '$r_s$ = %s'%round(rp,2) ,horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Stack'); pl.ylabel('WAIS')
	
	ax=pl.subplot(132)
	xi, y_fit, lower, upper = lin_regr_uncertainty(stack, edc)
	rp = spearmanr(stack, edc)[0]
	pl.plot(stack, edc, 'o', color='black')
	pl.plot(xi, y_fit, color='tomato')
	pl.plot(xi, lower, '--')
	pl.plot(xi, upper, '--', color='cornflowerblue')
	pl.text(0.35,0.97, '$r_s$ = %s'%round(rp,2) ,horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Stack'); pl.ylabel('EDC')
	
	ax=pl.subplot(133)
	xi, y_fit, lower, upper = lin_regr_uncertainty(wais, edc)
	rp = spearmanr(wais, edc)[0]
	pl.plot(wais, edc, 'o', color='black')
	pl.plot(xi, y_fit, color='tomato')
	pl.plot(xi, lower, '--')
	pl.plot(xi, upper, '--', color='cornflowerblue')
	pl.text(0.35,0.97, '$r_s$ = %s'%round(rp,2) ,horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('WAIS'); pl.ylabel('EDC')
	
	


def scan_transitions(volcs, resp, N, M, stack_onsets, stack_terminations):

	fig=pl.figure()
	#for i in range(0,16):#N
	j=0
	#for i in range(16,32):
	#for i in range(32,48):
	#for i in range(48,64):
	#for i in range(64, 80):
	for i in range(80, N):
		idx_onset = find_nearest(stack_onsets, volcs[i])
		idx_term = find_nearest(stack_terminations, volcs[i])
		#print(volcs[i], stack_onsets[idx_onset], stack_terminations[idx_term])
		print(volcs[i], np.abs(volcs[i]-stack_onsets[idx_onset]), np.abs(volcs[i]-stack_terminations[idx_term]))
		pl.subplot(4,4,j+1)
		pl.plot(range(-M+int(volcs[i]),M+int(volcs[i])), resp[i])#range(-M,M)
		pl.plot(100*[volcs[i]], np.linspace(min(resp[i]), max(resp[i]), 100), '--')
		if np.abs(volcs[i]-stack_onsets[idx_onset])<50.:
			print('ONSET')
		if np.abs(volcs[i]-stack_terminations[idx_term])<50.:
			print('TERMINATION')
		#pl.plot(100*[volcs[i]], np.linspace(min(resp[i]), max(resp[i]), 100), '--')
		j+=1
	
def diffusion(volcs, responses, M):
	resp = responses['Stack']
	resp_young = resp[:27]; resp_mid = resp[27:54]; resp_old = resp[54:]
	resp_curve_young = mean_curve(np.asarray(resp_young))
	resp_curve_mid = mean_curve(np.asarray(resp_mid))
	resp_curve_old = mean_curve(np.asarray(resp_old))
	
	def integrated_anomaly(x):
		i = 0; j = 1
		anomaly_sum = 0.

		while x[M+i]<0:
			anomaly_sum += x[M+i]
			#print(i, x[M+i])
			i+=1
			
		while x[M-j]<0:
			anomaly_sum += x[M-j]
			#print(j, x[M-j])
			j+=1
		print(j-1, i+j-1)
		#print(anomaly_sum/(i+j-1))
		#print(i,j)
		return anomaly_sum, i+j-1
	
	sum_young, dur_young = integrated_anomaly(resp_curve_young)
	sum_mid, dur_mid = integrated_anomaly(resp_curve_mid)
	sum_old, dur_old = integrated_anomaly(resp_curve_old)
	print(sum_young, sum_mid, sum_old)
	print(dur_young, dur_mid, dur_old)
	
	### bootstrap uncertainties by sampling eruptions with replacement.
	def bootstrap(resp0):
		N_boot = 1000
		sample_areas = np.empty(N_boot)
		for i in range(N_boot):
			boots = []
			for n in range(len(resp0)):
				pick = np.random.multinomial(1, [1./len(resp0)]*len(resp0))
				#print(pick)
				#print(np.where(pick==1)[0][0])
				boots.append(resp0[np.where(pick==1)[0][0]])
			resp_curve0 = mean_curve(np.asarray(boots))
			blab = integrated_anomaly(resp_curve0)[0]
			sample_areas[i] = blab#integrated_anomaly(resp_curve0)[0]
			print(blab)
		#print(sample_areas)
		return np.percentile(sample_areas, 5.), np.mean(sample_areas), np.percentile(sample_areas, 95.)
		
	p5_young, mean_young, p95_young = bootstrap(resp_young)
	p5_mid, mean_mid, p95_mid = bootstrap(resp_mid)
	p5_old, mean_old, p95_old = bootstrap(resp_old)
	print(p5_mid, mean_mid, p95_mid)
	
	fig=pl.figure()
	pl.plot(1., mean_young, 'x')
	pl.plot(1., sum_young, 'o')

	pl.plot(2., mean_mid, 'x')
	pl.plot(2., sum_mid, 'o')
	
	pl.plot(3., mean_old, 'x')
	pl.plot(3., sum_old, 'o')
		
	
	fig=pl.figure()
	pl.bar(range(-M,M), resp_curve_young)
	

def pca_cores(volcs, idcs_all, anomalies, mask_nh):
	
	print(len(anomalies['NGRIP']), len(anomalies['GRIP']), len(anomalies['NEEM']), len(anomalies['GISP2']), len(anomalies['WAIS']))
	
	idcs_DO_all=[9, 22, 24, 40, 44, 64, 71]
		
	idcs_nh_all = []; idcs_sh_all = []
	for i in range(len(volcs)):
		if mask_nh[i] == 1:
			idcs_nh_all.append(i)
		else:
			idcs_sh_all.append(i)
	
	idcs_DO=[]; idcs_nh = []; idcs_sh = []
	ngrip = []; grip = []; neem = []; gisp2 = []; wais = []
	j=0
	for i in range(len(volcs)):
		if i in idcs_all:
			ngrip.append(anomalies['NGRIP'][i])
			grip.append(anomalies['GRIP'][i])
			neem.append(anomalies['NEEM'][i])
			gisp2.append(anomalies['GISP2'][i])
			wais.append(anomalies['WAIS'][i])
			if i in idcs_DO_all:
				idcs_DO.append(j)
			if i in idcs_nh_all:
				idcs_nh.append(j)
			else:
				idcs_sh.append(j)
			j+=1
			
	print(len(idcs_nh), len(idcs_sh), len(idcs_DO), len(idcs_all))
	
			
	def biplot(score,coeff,pcax,pcay,labels=None):
		pca1=pcax-1
		pca2=pcay-1
		xs = score[:,pca1]
		ys = score[:,pca2]
		#n=len(labels)
		n=5#2
		scalex = 1.0/(xs.max()- xs.min())
		scaley = 1.0/(ys.max()- ys.min())
		fig=pl.figure(figsize=(6,4.5))
		pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98)
		
		for i in idcs_DO:
			pl.plot(xs[i]*scalex,ys[i]*scaley, 'o', color='tomato', markersize=10)
		#pl.scatter(xs*scalex,ys*scaley, color='royalblue')
		
		for i in idcs_sh:
			pl.plot(xs[i]*scalex,ys[i]*scaley, 'o', color='springgreen')
		pl.plot(xs[idcs_sh[0]]*scalex,ys[idcs_sh[0]]*scaley, 'o', color='springgreen', label='SH/LL')
		for i in idcs_nh:
			pl.plot(xs[i]*scalex,ys[i]*scaley, 'o', color='royalblue')
		pl.plot(xs[idcs_nh[0]]*scalex,ys[idcs_nh[0]]*scaley, 'o', color='royalblue', label='NH')

		#for k, txt in enumerate(evt_label):
		#        pl.annotate(txt, (xs[k]*scalex,ys[k]*scaley))
		for i in range(n):
		        pl.arrow(0, 0, coeff[pca1,i], coeff[pca2,i],color='r',alpha=0.5) #coeff[i,pca1], coeff[i,pca2]
		        if labels is None:
		                pl.text(coeff[pca1,i]* 1.15, coeff[pca2,i] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
		        else:
		                pl.text(coeff[pca1,i]* 1.15, coeff[pca2,i] * 1.15, labels[i], color='g', ha='center', va='center')
		pl.xlim(-1,1)
		pl.ylim(-1,1)
		pl.xlabel("PC{}".format(pcax))
		pl.ylabel("PC{}".format(pcay))
		pl.grid(); pl.legend(loc='lower left')
        
	Xp = np.asarray([ngrip, grip, neem, gisp2, wais])
	#Xp = np.asarray([ngrip, grip, neem, gisp2])
	Xp = Xp.transpose()
	Xp = StandardScaler().fit_transform(Xp)
	pca = PCA(n_components=2)
	pca.fit(Xp)
	expl_var = pca.explained_variance_ratio_
	score =  pca.fit_transform(Xp)
	coeff = pca.components_
	
	print(pca.n_components_, pca.n_features_, pca.n_samples_)

	print('Expl variance: ', pca.explained_variance_ratio_)
	print('Components: ', pca.components_)
	biplot(score,coeff,1,2, labels=['NGRIP', 'GRIP', 'NEEM', 'GISP2', 'WAIS'])

	
	
def correlation_cores(means, cores, masks):
	fig, axes = pl.subplots(2,3, figsize=(11,7))
	letter_subplots(letters='a', yoffset=1.05)
	pl.subplots_adjust(left=0.08, bottom=0.1, right=0.97, top=0.96, wspace=0.25, hspace=0.27)
	j=0
	cores2 = cores.copy()
	for name1 in cores:
		for name2 in cores2:
			if name1!=name2:
				idcs = []
				for i in range(len(means[name1])):
					if (masks[name1][i]==1 and masks[name2][i]==1):
						idcs.append(i)
				print(name1, name2, len(idcs))
				print(idcs)

				means1 = np.asarray(means[name1])[idcs]
				means2 = np.asarray(means[name2])[idcs]
				
				#means10 = sm.add_constant(means1)
				#model = sm.OLS(means2,means10)
				#results = model.fit()
				#a, b = results.params
				#xvals = np.linspace(min(means1), max(means1), 100)
				
				xi, y_fit, lower, upper = lin_regr_uncertainty(means1, means2)
				rp = spearmanr(means1, means2)[0]
				
				#ax=pl.subplot(2,3,j+1)
				#g = sns.jointplot(x=means1, y=means2, kind="reg")
				axes[int(j/3)%2][j%3].plot(means1, means2, 'o', color='black')
				#pl.plot(xvals, [a + b*x for x in xvals], color='tomato')
				axes[int(j/3)%2][j%3].plot(xi, y_fit, color='tomato')
				axes[int(j/3)%2][j%3].plot(xi, lower, '--')
				axes[int(j/3)%2][j%3].plot(xi, upper, '--', color='cornflowerblue')
				axes[int(j/3)%2][j%3].text(0.5,0.97, '$r_s$ = %s'%round(rp,2) ,horizontalalignment='right',verticalalignment='top',transform = axes[int(j/3)%2][j%3].transAxes,fontsize=15)
				axes[int(j/3)%2][j%3].set_xlabel(name1); axes[int(j/3)%2][j%3].set_ylabel(name2)
				
				j+=1
		cores2.remove(name1)
		
def responses_latitude(volcs, responses, mask_nh, M, resp_curve_stacked_alt):

	responses_nh = []; responses_sh = []
	responses_all = responses
	resp_curve_all, p14_all, p86_all = mean_confidence_band(np.asarray(responses))
	for i in range(len(responses)):
		if mask_nh[i] == 1:
			responses_nh.append(responses[i])
		else:
			responses_sh.append(responses[i])
	resp_curve_nh, p14_nh, p86_nh = mean_confidence_band(np.asarray(responses_nh))
	resp_curve_sh, p14_sh, p86_sh = mean_confidence_band(np.asarray(responses_sh))
	
	print(len(responses_nh), len(responses_sh))
	
	fig=pl.figure(figsize=(6,4.5))
	pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98)
	pl.fill_between(range(-M,M), p14_all, p86_all ,color='gray',alpha=0.25)
	#pl.bar(range(-M,M), resp_curve_stacked_alt, color='black');pl.grid(axis='both')
	pl.bar(range(-M,M), resp_curve_all, color='black');pl.grid(axis='both')
	pl.plot(range(-M,M), resp_curve_nh, color='royalblue', label='NH')
	pl.plot(range(-M,M), resp_curve_sh, color='springgreen', label='LL or SH')

	pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-30,40); pl.legend(loc='best')
	pl.xlabel('Time before eruption (years)')

def calc_response(volcs, t_ngrip, stack_terminations, stack_onsets, record, N, M, a, b, lead, k, shift):
	volc_gi_count = 0
	resp_all = np.empty((N,2*M))#; resp_gi = []; resp_gs = []; 
	anomalies = []
	#gi_mask = np.empty(N)
	print(M, k, a, b)
	### filter out part with DO onsets -> replace time series by NANs
	t_crit = np.asarray([14689., 27783., 28908., 38223., 40139., 49306., 55001.])
	for i in range(N):
		cut=0
		idx = find_nearest(t_ngrip, volcs[i]-shift)
		if min(np.abs(volcs[i]-t_crit))<M:
			#print('barrrrp',  min(np.abs(volcs[i]-t_crit)))
			cut = int(min(np.abs(volcs[i]-t_crit)))
		'''
		idx1 = find_nearest(stack_terminations, volcs[i])
		if volcs[i]>stack_terminations[idx1]:
		        idx1+=1

		idx2 = find_nearest(stack_onsets, volcs[i])
		if volcs[i]>stack_onsets[idx2]:
		        idx2+=1

		if stack_terminations[idx1]<stack_onsets[idx2]:
		        #print('stadial', volcs[i], stack_terminations[idx1], stack_onsets[idx2])
		        resp_gs.append(process_segment(record[idx-M:idx+M])) #- np.mean(ngrip[idx-30:idx+30])
		        #gi_mask[i]=0

		else:
		        #print 'interstadial', volcs[i], stack_terminations[idx1], stack_onsets[idx2]
		        volc_gi_count +=1
		        resp_gi.append(process_segment(record[idx-M:idx+M]))
		        #gi_mask[i]=1
	        '''

		anomalies.append(np.mean(record[idx-a:idx+b]) - np.mean(record[idx+b:idx+b+k]))
		
		#anomalies.append(np.mean(record[idx-a:idx+b] - np.mean(record[idx+b:idx+b+k])))
		#anomalies.append(np.mean(record[idx-a:idx+b]) - np.mean(record[idx+lead:idx+M]))
		#resp_all[i,:] = process_segment(record[idx-M:idx+M], k, b) #- np.mean(ngrip[idx-30:idx+30])
		
		#segment = record[idx-M:idx+M]
		
		segment = detrend(record[cut+idx-M:idx+M])
		segment = segment - np.mean(segment[M+lead-cut:])
		if cut>0:
			segment = np.concatenate((cut*[np.nan], segment))
		
		#anomalies.append(np.mean(segment[M-a:M+b]))
		#resp_all[i,:] = record[idx-M:idx+M] - np.mean(record[idx+b:idx+b+k])
		resp_all[i,:] = segment
		
	#print(gi_mask)
	#return resp_all, np.asarray(resp_gi), np.asarray(resp_gs), anomalies
	return resp_all, anomalies
	

def correlation_sulfur_response(volcs, volcs2, means, so4, resp_curves, resp_mean):
	
	#41887 missing in sulfur data.
	loading=[]; ages=[]; response=[]; curves=[]
	for i in range(len(volcs2)):
		#if min(abs(volcs2[i]-volcs))!=0:
		#	idx = np.argmin(abs(volcs2[i]-volcs))
		#	print(volcs2[i], volcs[idx])
		if min(abs(volcs2[i]-volcs))==0:
			idx = np.argmin(abs(volcs2[i]-volcs))
			loading.append(so4[i])
			ages.append(volcs2[i])
			response.append(means[idx])
			curves.append(resp_curves[idx])
	
	rp = spearmanr(loading, response)[0]
	print('Correlation', rp)
	
	### Response curves for smaller vs larger magnitude.
	curves_small=[]; curves_large=[]
	i0=0
	for i in range(len(loading)):
		if loading[i]<=np.median(loading):
			curves_small.append(curves[i])
		else:
			curves_large.append(curves[i])
			
	resp_curve_small = mean_curve(np.asarray(curves_small))
	resp_curve_large = mean_curve(np.asarray(curves_large))
	
	fig=pl.figure(figsize=(6,4.5))
	pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98)
	#pl.fill_between(range(-M,M), p14_all, p86_all ,color='gray',alpha=0.25)
	#pl.bar(range(-M,M), resp_curve_stacked_alt, color='black');pl.grid(axis='both')
	pl.bar(range(-50,50), resp_mean, color='black');pl.grid(axis='both')
	pl.plot(range(-50,50), resp_curve_small, color='royalblue', label='Small')
	pl.plot(range(-50,50), resp_curve_large, color='springgreen', label='Large')

	pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-30,40); pl.legend(loc='best')
	pl.xlabel('Time before eruption (years)')
	
	
	### scatterplot of mean anomalies vs magnitude.
	g = sns.jointplot(x=loading, y=response, kind="reg")#bw#marginal_kws=dict(bins=50, fill=False)
	#g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
	pl.xlabel('Stratospheric aerosol loading (Tg)'); pl.ylabel('Isotopic response')
	idcs = [9, 22, 24, 40, 44, 63, 70]
	print([ages[i] for i in idcs])
	print(len(ages))
	
	labels = ['1', '3', '4', '8', '9', '13', '15.1']
	for i in range(len(idcs)):
		#print(ages[idcs[i]])
		pl.plot(loading[idcs[i]], response[idcs[i]], 'o', markersize=8., color='tomato')
		pl.text(loading[idcs[i]], response[idcs[i]], labels[i])
	pl.plot(loading[19], response[19], 'o', markersize=8., color='gold')
	
	'''
	### correlation with average NH deposition
	deposition_all, age_all = np.loadtxt('magnitude_age_nh_9k_prepr.txt', unpack=True)
	#age_jiamei = np.loadtxt('age_bipolar_jiamei_9k_prepr.txt') #these are now as in SVE20
	age_jiamei = np.loadtxt('age_bipolar_jiamei_nh_9k_resub.txt') #ages adjusted to the NH table
	
	### remove manually from NH bipolar list: First two; 33335; 44744; 44763
	age_jiamei = age_jiamei[2:]
	#print(age_jiamei)
	idcs=np.argwhere(np.isin(age_jiamei,[33335., 44744., 44763.]))
	age_jiamei = np.delete(age_jiamei, idcs)
	print(age_jiamei)
	
	### BIPOLAR EVENTS MISSING in unipolar lists...
	#59180 -> Missing in NH data
	#43327 -> 43296 in NH list!!
	#33328 -> Missing in NH data
	#25759 -> 25750 in NH data; 
	#12961 -> Missing in NH data
	ages_nh = []; response = []
	for i in range(len(age_jiamei)):
		#if min(abs(age_jiamei[i]-volcs))<10:
		#	idx = np.argmin(abs(age_jiamei[i]-volcs))
		#	ages_nh.append(age_jiamei[i])
		#	response.append(means[idx])
		idx = np.argmin(abs(age_jiamei[i]-volcs))
		print(age_jiamei[i], volcs[idx], min(abs(age_jiamei[i]-volcs)))
		response.append(means[idx])
		ages_nh.append(age_jiamei[i])
		
	print(len(ages_nh))
	
	deposition = []; rm_idcs = []
	k=0
	for i in range(len(ages_nh)):
		if min(abs(ages_nh[i]-age_all))<.1:
			idx = np.argmin(abs(ages_nh[i]-age_all))
			k+=1
			#print(k)
			#ages.append(age_all[i])
			deposition.append(deposition_all[idx])
		else:
			rm_idcs.append(i)
			print(ages_nh[i])
	print(len(deposition))
	print(rm_idcs)
	for index in sorted(rm_idcs, reverse=True):
		del response[index]
		
	print(len(response))
	
	g = sns.jointplot(x=deposition, y=response, kind="reg")
	'''
	
#def process_segment(data, k, b):
	#return detrend(data)
	### OLD
	#return data - np.mean(data)
	### RATHER: anomaly with respect to preceding 50 years!
#	return data - np.mean(data)

def hypothesis_test(volcs, means, record, t, label, a, b, lead, mask_gi, mask_nh):

	### cut record so that bootstrap does not include stuff outside eruption interval.
	cut_a = find_nearest(t, volcs[0]-100)
	cut_b = find_nearest(t, volcs[-1]+100)
	t = t[cut_a:cut_b]
	record = record[cut_a:cut_b]
	
	means=np.asarray(means)
	#for i in range(len(volcs)):
	#	print(volcs[i],': ', means[i])
	means_boot = bootstrap_means(record, a, b, lead)
	'''
	statistics_sort = np.sort(means_boot)
	p_index = np.argmin(np.abs(statistics_sort - means[4]))
	pvalue_precise = float(p_index)/(len(means_boot))
	print('p-value Kurile Lake (%s): %s'%(label,pvalue_precise))
	
	p_index = np.argmin(np.abs(statistics_sort - means[2]))
	pvalue_precise = float(p_index)/(len(means_boot))
	print('p-value Khangar (%s): %s'%(label,pvalue_precise))
	'''
	
	fig=pl.figure(figsize=(5.,3.5))
	#letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	xgrid = np.linspace(min(means_boot)-0.5*np.std(means_boot), max(means_boot)+0.5*np.std(means_boot),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(means_boot)/12.).fit(means_boot[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='gray', alpha=0.5, label='Bootstrap')

	xgrid = np.linspace(min(means)-0.5*np.std(means), max(means)+0.5*np.std(means),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(means)/4.).fit(means[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='royalblue', alpha=0.4, label='Volcanic events')
	
	for i in range(len(volcs)):
		#if i==0:
		if mask_gi[i]==0:
			pl.plot(100*[means[i]], np.linspace(0,0.07,100), '--', color='lightseagreen')
		else:
			pl.plot(100*[means[i]], np.linspace(0,0.07,100), '--', color='darkorange')
		
	### seprate into NH-SH as well as GI-GS
	
	gi_nh = []; gi_sh = []; gs_nh = []; gs_sh = []
	for i in range(len(volcs)):
		if mask_gi[i]==1:
			if mask_nh[i]==1:
				print('GI + NH', means[i])
				gi_nh.append(means[i])
			else:
				print('GI + LLSH', means[i])
				gi_sh.append(means[i])
		else:
			if mask_nh[i]==1:
				print('GS + NH', means[i])
				gs_nh.append(means[i])
			else:
				print('GS + LLSH', means[i])
				gs_sh.append(means[i])
				
	print('GI + NH: ', len(gi_nh), np.mean(gi_nh))
	print('GI + SH: ', len(gi_sh), np.mean(gi_sh))
	print('GS + NH: ', len(gs_nh), np.mean(gs_nh))
	print('GS + SH: ', len(gs_sh), np.mean(gs_sh))
		
	idcs_DO=[9, 22, 24, 40, 44, 64, 71]
	for i in idcs_DO:
		print(volcs[i], means[i])
		pl.plot(100*[means[i]], np.linspace(0,0.15,100), color='tomato', linewidth=1.5)
	#pl.plot(100*[means[19]], np.linspace(0,0.1,100), color='gold', linewidth=1.5)
	
	pl.plot(100*[np.mean(means)], np.linspace(0., max(np.exp(log_dens)), 100), color='royalblue', linewidth=2.5)
	pl.plot(100*[np.percentile(means_boot, 16.)], np.linspace(0., max(np.exp(log_dens)), 100), '--', color='black')

	pl.ylabel('PDF'); pl.xlabel('6-year isotopic anomaly'); pl.legend(loc='best', fontsize=11)
	
	print('Signal-to-noise Ratio: ', np.mean(means)/np.percentile(means_boot, 16.))
	
	volcs_sorted = [x for _,x in sorted(zip(means,volcs))]
	means_sorted = np.sort(means)
	#for i in range(len(volcs)):
	#	print(volcs_sorted[i],': ', means_sorted[i])

def response_mag_match(volcs_gi0, volcs_gs0, mag_gi0, mag_gs0, M, a, b, k0, t, data, stack_terminations, stack_onsets, shift):

	### NGRIP
	#mag_gs = mag_gs0; mag_gi = mag_gi0
	#volcs_gs = volcs_gs0; volcs_gi = volcs_gi0
	
	### NEEM -> Reverse roles of GI and GS? Not necessarily, but it makes a difference.
	### EDC
	mag_gs = mag_gi0; mag_gi = mag_gs0
	volcs_gs = volcs_gi0; volcs_gi = volcs_gs0

	### remove 2 largest GI events for better matching of proposal and target.
	### also for NGRIP bipolar.
	#mag_gi.remove(max(mag_gi))
	### 1 for EDC unipolar and also Bipolar? (largest event is GS, but it is very close.)
	#mag_gi.remove(max(mag_gi))
	
	N = len(mag_gs)
	
	def sample_proposal():
		x_ind = np.random.multinomial(1, [1./N]*N, size=1)[0]
		x = np.where(x_ind==1)[0]
		x=x[0]
		return mag_gs[x], volcs_gs[x]

	#perc = [0., 2., 4., 6., 9., 12., 16., 21., 25., 30., 40., 50., 60., 70., 85., 100.]
	#perc = [0., 2., 3., 5., 7., 9., 13., 16., 18., 22., 30., 38., 45., 54., 60., 70., 85., 100.]
	#perc = [0., 3., 6., 12., 15., 20., 25., 30., 40., 50., 60., 70., 80., 90.]
	#perc = [0., 5., 10., 20., 30., 40., 55., 65., 80., 90., 100.] ### NGRIP
	#perc = [0., 30., 60., 65., 75., 80., 90., 100.] ### NEEM GS
	#perc = [0., 5., 10., 20., 30., 40., 60., 80., 100.] ### NEEM
	#perc = [0., 10, 20., 30., 60., 65., 75., 80., 90., 100.] ### EDC GS
	#perc = [0., 10, 20., 30., 60., 75., 85., 100.] ### EDC GI
	
	#perc = [0., 20., 40., 60., 75., 80., 90., 100.] ### NGRIP bipolar
	#perc = [0., 18., 35., 42., 65., 75., 78., 83., 95., 100.] ### NGRIP bipolar
	perc = [0., 20., 30., 45., 65., 75., 80., 90., 100.] ### EDC Bipolar (sample from GI)
	
	print('-------------------- Quantile Resampling ---------------------')
	print('# GI samples: ', len(mag_gi))
	print(np.sort(mag_gi))
	print('# GS samples: ', len(mag_gs))
	print(np.sort(mag_gs))
	
	print('Largest event in (truncated) target sample: ', max(mag_gi))
	quantiles = [np.percentile(mag_gi, x) for x in perc]
	print(quantiles)
	counts_target = [((quantiles[i-1] <= mag_gi) & (mag_gi < quantiles[i])).sum() for i in range(1,len(perc))]
	densities_target = [counts_target[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
	print(densities_target)
	print(len(mag_gi)/100.)
	
	counts = [((quantiles[i-1] < mag_gs) & (mag_gs < quantiles[i])).sum() for i in range(1,len(perc))]
	idcs = [np.where(np.logical_and(mag_gs>=quantiles[i-1], mag_gs<=quantiles[i])) for i in range(1,len(perc))]
	samples_quant = [np.asarray(mag_gs)[idcs[i]] for i in range(len(idcs))]
	print(perc[1:])
	print(counts)
	### density is counts per 1%-quantile
	densities = [counts[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
	print(densities)
	factors = [1.-min(densities)/densities[i] for i in range(len(densities))]
	print(factors)
	
	reps = 1#000#00
	resamples_all = []; accept_no = []
	volcs_resamples_all = []
	for k in range(reps):
		print(k)
		resamples = []; volcs_resamples = []
		#mag_samples = [sample_proposal() for i in range(N)]
		mag_samples = []
		volc_samples = []
		N0 = 50000
		for i in range(N0):
			mag_samples0, volc_samples0 = sample_proposal()
			mag_samples.append(mag_samples0); volc_samples.append(volc_samples0)
		#print(mag_samples)
		### Note in the following one of the limits should be equal or greater/smaller
		counts = [((quantiles[i-1] <= mag_samples) & (mag_samples < quantiles[i])).sum() for i in range(1,len(perc))]
		#print(counts)
		### Same should be done here, more importantly
		idcs = [np.where(np.logical_and(mag_samples>=quantiles[i-1], mag_samples<quantiles[i])) for i in range(1,len(perc))]
		samples_quant = [np.asarray(mag_samples)[idcs[i]] for i in range(len(idcs))]
		volcs_quant = [np.asarray(volc_samples)[idcs[i]] for i in range(len(idcs))]
		#print(samples_quant)
		densities = [counts[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
		if (np.asarray(densities).all() == 0.):
			print('problem')
			continue
		factors = [1.-min(densities)/densities[i] for i in range(len(densities))]
		#print(factors)
		
		for i in range(len(counts)):
			resamples_quant = []
			volcs_resamples_quant = []
			samples0 = samples_quant[i]
			volcs0 = volcs_quant[i]
			for j in range(len(samples0)):
				if np.random.uniform()>factors[i]:
					resamples_quant.append(samples0[j])
					volcs_resamples_quant.append(volcs0[j])
			#print(len(resamples_quant)/(perc[i+1]-perc[i]))
			resamples.append(resamples_quant)
			volcs_resamples.append(volcs_resamples_quant)
		resamples = [item for sublist in resamples for item in sublist]
		volcs_resamples = [item for sublist in volcs_resamples for item in sublist]
		#print(len(resamples))
		accept_no.append(len(resamples))
		resamples_all.append(resamples)
		volcs_resamples_all.append(volcs_resamples)
		
	volcs_resamples_all = [item for sublist in volcs_resamples_all for item in sublist]
	
	accept_all = [item for sublist in resamples_all for item in sublist]
	#print(accept_all)
	#print('Mean # samples', np.mean(accept_no))
	#print('10-, 90-percentile # samples', np.percentile(accept_no, 10.), np.percentile(accept_no, 90.))
	
	grid_pts_sampled = np.linspace(2.,max(np.log(accept_all))+.5, 1000)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(accept_all))/6.).fit(np.asarray(np.log(accept_all))[:, np.newaxis])
	log_dens = kde.score_samples(grid_pts_sampled[:, np.newaxis])
	pdf_sampled = np.exp(log_dens)
	
	grid_cdf_sample, cdf_target_sample = calc_cumu_density(np.log(accept_all))
	
	grid_cdf, cdf_target = calc_cumu_density(np.log(mag_gi))
	grid_cdf_prop, cdf_proposal = calc_cumu_density(np.log(mag_gs))

	### Standard KDE.
	grid_pdf_gi = np.linspace(2., max(np.log(mag_gi))+.5, 1000.)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/6.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(grid_pdf_gi[:, np.newaxis])
	dens_gi = np.exp(log_dens)
	
	grid_pdf_gs = np.linspace(2., max(np.log(mag_gs))+.5, 1000.)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gs))/6.).fit(np.asarray(np.log(mag_gs))[:, np.newaxis])
	log_dens = kde.score_samples(grid_pdf_gs[:, np.newaxis])
	dens_gs = np.exp(log_dens)
	

	fig, axes=pl.subplots(1,2, figsize=(10.,3.5))
	letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	#pl.suptitle('NH eruptions not identified as bipolar according to magnitude (Quantile matching)')
	#pl.suptitle('NH eruptions not identified as bipolar according to magnitude (Prob. ratio test)')

	axes[0].plot(grid_pdf_gs, dens_gs, color='tomato', label='GS')#fact*
	axes[0].plot(grid_pdf_gi, dens_gi, color='black', label='GI')
	axes[0].plot(grid_pts_sampled, pdf_sampled, '--', color='royalblue', label='Resampled')
	axes[0].set_xlabel('Magnitude (kg/km$^2$)'); axes[0].set_ylabel('PDF'); axes[0].legend(loc='best')
	
	axes[1].plot(grid_cdf, cdf_target, color='black', label='GI')
	axes[1].plot(grid_cdf_prop, cdf_proposal, color='tomato', label='GS')
	axes[1].plot(grid_cdf_sample, cdf_target_sample, '--', color='royalblue', label='Resampled')
	axes[1].legend(loc='best'); axes[1].set_xlabel('Magnitude (kg/sqkm)'); axes[1].set_ylabel('CDF')
	

	lead = 5
	
	resp, means0 = calc_response(volcs_gi, t, stack_terminations, stack_onsets, data, len(volcs_gi), M, a, b, lead, k0, shift)
	resp_curve_gi, perc14_gi, perc86_gi = mean_confidence_band(np.asarray(resp))
	resp, means0 = calc_response(volcs_gs, t, stack_terminations, stack_onsets, data, len(volcs_gs), M, a, b, lead, k0, shift)
	resp_curve_gs, perc14_gs, perc86_gs = mean_confidence_band(np.asarray(resp))
	
	print(len(volcs_resamples_all))
	#print(volcs_resamples_all)
	
	resp, means0 = calc_response(volcs_resamples_all, t, stack_terminations, stack_onsets, data, len(volcs_resamples_all), M, a, b, lead, k0, shift)
	resp_curve_sample, perc14_sample, perc86_sample = mean_confidence_band(np.asarray(resp))


	fig=pl.figure()
	#fig, axes=pl.subplots(1,2, figsize=(10.,3.5))
	#letter_subplots(axes, letters='a', yoffset=1.05) 
	#pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	#pl.subplot(121)
	#axes[0].plot(range(-k,M), resp_curve_gi, color='black', label='GI')
	#pl.plot(range(-k,M), perc14_gi, '--', color='black')
	#pl.plot(range(-k,M), perc86_gi, '--', color='black')
	#axes[0].plot(range(-k,M), resp_curve_gs, color='tomato', label='GS')
	pl.plot(range(-k0,M), resp_curve_gi, color='black', label='GI')
	pl.plot(range(-k0,M), resp_curve_gs, color='tomato', label='GS')
	pl.plot(range(-k0,M), resp_curve_sample, color='green', label='Sampled')
	
	return resp_curve_sample

def extract_bipolar(volcs_bi_ages, volcs, mag, mask_gi):

	### Can choose in between keeping the SVE20 ages
	### or using the LIN22 ages for the bipolar eruptions.

	k0=0
	idcs_rm = []; volcs_bi = []; mag_bi = []
	mask_gi_new = []
	for i in range(len(volcs_bi_ages)):
		if min(abs(volcs_bi_ages[i]- volcs))<5:#5
			idx = np.argmin(abs(volcs_bi_ages[i]-volcs))
			k0+=1
			print(k0, volcs_bi_ages[i], volcs[idx], mag[idx])
			idcs_rm.append(idx)
			#volcs_bi.append(volcs[idx]) ### LIN22 ages
			volcs_bi.append(volcs_bi_ages[i]) ### SVE20 ages
			mask_gi_new.append(mask_gi[i])
			mag_bi.append(mag[idx])
			
			
	volcs_uni = list(volcs)
	mag_uni = list(mag)
	for index in sorted(idcs_rm, reverse=True):
		del volcs_uni[index]
		del mag_uni[index]
	mag_uni = np.asarray(mag_uni)
	mag_bi = np.asarray(mag_bi)
	print(len(volcs_uni), len(volcs))
	
	print(len(mag_uni), len(mag))
	return volcs_bi, mag_bi, volcs_uni, mag_uni, mask_gi_new


def bootstrap_means(data, a, b, lead):
	N = 50000; K = len(data); M = 50#; L = 2
	means = np.empty(N)
	for i in range(N):
		idx = randrange(M,K-M-b)#randrange(a,K-M-b)
		means[i] = np.mean(data[idx-a:idx+b])-np.mean(data[idx+b:idx+M+b])
		#means[i] = np.mean(data[idx-a:idx+b])-np.mean(data[idx+lead:idx+M])
		#np.mean(record[idx-a:idx+b]) - np.mean(record[idx+b:idx+b+k])
		#segment = detrend(data[idx-M:idx+M])
		#segment = segment - np.mean(segment[M+lead:])
		#means[i] = np.mean(segment[M-a:M+b])
	return means

def confidence_band(data):
        return [np.percentile(data[:,j], 14.) for j in range(len(data[0,:]))], [np.percentile(data[:,j], 86.) for j in range(len(data[0,:]))]

def mean_confidence_band(data):
        return [np.nanmean(data[:,j]) for j in range(len(data[0,:]))], [np.nanpercentile(data[:,j], 14.) for j in range(len(data[0,:]))], [np.nanpercentile(data[:,j], 86.) for j in range(len(data[0,:]))]
        
def calc_cumu_density(data):
    	N = len(data)
    	grid = np.sort(data)
    	density = np.array(range(N))/float(N)
    	return [grid, density]

def mean_curve(data):
	return np.asarray([np.nanmean(data[:,j]) for j in range(len(data[0,:]))])

def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
     
def lin_regr_uncertainty(x, y):
	# fit
	f = lambda x, *p: polyval(p, x)
	p, cov = curve_fit(f, x, y, [1, 1])
	# simulated draws from the probability density function of the regression
	xi = linspace(np.min(x), np.max(x), 100)
	ps = np.random.multivariate_normal(p, cov, 10000)
	ysample = np.asarray([f(xi, *pi) for pi in ps])
	lower = percentile(ysample, 2.5, axis=0)
	upper = percentile(ysample, 97.5, axis=0)

	# regression estimate line
	y_fit = poly1d(p)(xi)

	# plot
	#plot(x, y, 'bo')
	#plot(xi, y_fit, 'r-')
	#plot(xi, lower, 'b--')
	#plot(xi, upper, 'b--')
	
	return xi, y_fit, lower, upper
     
     
'''
	k = 0 ### count DO events
	volc_gi_count = 0
	for i in range(N):

		idx = find_nearest(t_ngrip, volcs[i])

		idx1 = find_nearest(stack_terminations, volcs[i])
		if volcs[i]>stack_terminations[idx1]:
		        idx1+=1

		idx2 = find_nearest(stack_onsets, volcs[i])
		if volcs[i]>stack_onsets[idx2]:
		        idx2+=1

		if stack_terminations[idx1]<stack_onsets[idx2]:
		        #print('stadial', volcs[i], stack_terminations[idx1], stack_onsets[idx2])

		        resp_ngrip_all_gs.append(process_segment(ngrip[idx-M:idx+M])) #- np.mean(ngrip[idx-30:idx+30])
		        resp_grip_all_gs.append(process_segment(grip[idx-M:idx+M]))
		        resp_gisp2_all_gs.append(process_segment(gisp2[idx-M:idx+M]))
		        resp_neem_all_gs.append(process_segment(neem[idx-M:idx+M]))
		        resp_stack_all_gs.append(process_segment(stack[idx-M:idx+M]))
		        resp_wais_all_gs.append(process_segment(wais[idx-M:idx+M]))
		        resp_edml_all_gs.append(process_segment(edml[idx-M:idx+M]))
		        resp_edc_all_gs.append(process_segment(edc[idx-M:idx+M]))

		else:
		        #print 'interstadial', volcs[i], stack_terminations[idx1], stack_onsets[idx2]
		        volc_gi_count +=1
		        #resp_ngrip_all_gi[i,:] = ngrip[idx-30:idx+30]- np.mean(ngrip[idx-30:idx+30])
		        #resp_grip_all_gi[i,:] = grip[idx-30:idx+30] - np.mean(grip[idx-30:idx+30])
		        #resp_gisp2_all_gi[i,:] = gisp2[idx-30:idx+30] - np.mean(gisp2[idx-30:idx+30])
		        #resp_neem_all_gi[i,:] = neem[idx-30:idx+30] - np.mean(neem[idx-30:idx+30])
		        #resp_stack_all_gi[i,:] = stack[idx-30:idx+30] - np.mean(stack[idx-30:idx+30])
		        resp_ngrip_all_gi.append(process_segment(ngrip[idx-M:idx+M]))
		        resp_grip_all_gi.append(process_segment(grip[idx-M:idx+M]))
		        resp_gisp2_all_gi.append(process_segment(gisp2[idx-M:idx+M]))
		        resp_neem_all_gi.append(process_segment(neem[idx-M:idx+M]))
		        resp_stack_all_gi.append(process_segment(stack[idx-M:idx+M]))
		        resp_wais_all_gi.append(process_segment(wais[idx-M:idx+M]))
		        resp_edml_all_gi.append(process_segment(edml[idx-M:idx+M]))
		        resp_edc_all_gi.append(process_segment(edc[idx-M:idx+M]))

		#means_ngrip.append(np.mean(ngrip[idx-5:idx+2]) - np.mean(ngrip[idx+2:idx+52]))
		means_stack.append(np.mean(stack[idx-5:idx+5]) - np.mean(stack[idx+5:idx+55]))

		resp_ngrip_all[i,:] = process_segment(ngrip[idx-M:idx+M]) #- np.mean(ngrip[idx-30:idx+30])
		resp_grip_all[i,:] = process_segment(grip[idx-M:idx+M])
		resp_gisp2_all[i,:] = process_segment(gisp2[idx-M:idx+M])
		resp_neem_all[i,:] = process_segment(neem[idx-M:idx+M])
		resp_stack_all[i,:] = process_segment(stack[idx-M:idx+M])
		resp_wais_all[i,:] = process_segment(wais[idx-M:idx+M])
		resp_edml_all[i,:] = process_segment(edml[idx-M:idx+M])
		resp_edc_all[i,:] = process_segment(edc[idx-M:idx+M])
'''

'''
	ax=pl.subplot(241)
	pl.fill_between(range(-M,M),ngrip_perc14,ngrip_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_ngrip);pl.grid(axis='both')
	pl.text(0.95,0.95,'NGRIP',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(242)
	pl.fill_between(range(-M,M),grip_perc14,grip_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_grip);pl.grid(axis='both')
	pl.text(0.95,0.95,'GRIP',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(243)
	pl.fill_between(range(-M,M),gisp2_perc14,gisp2_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_gisp2);pl.grid(axis='both')
	pl.text(0.95,0.95,'GISP2',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(244)
	pl.fill_between(range(-M,M),neem_perc14,neem_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_neem);pl.grid(axis='both')
	pl.text(0.95,0.95,'NEEM',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(245)
	pl.fill_between(range(-M,M),stack_perc14,stack_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_stack);pl.grid(axis='both')
	pl.text(0.95,0.95,'Stack',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)');pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(246)
	pl.fill_between(range(-M,M),wais_perc14,wais_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_wais);pl.grid(axis='both')
	pl.text(0.95,0.95,'WAIS',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
	ax=pl.subplot(247)
	pl.fill_between(range(-M,M),edml_perc14,edml_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_edml);pl.grid(axis='both')
	pl.text(0.95,0.95,'EDML',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
	ax=pl.subplot(248)
	pl.fill_between(range(-M,M),edc_perc14,edc_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_edc);pl.grid(axis='both')
	pl.text(0.95,0.95,'EDC',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')

	fig=pl.figure(figsize=(15,5))
	pl.subplots_adjust(left=0.1, bottom=0.12, right=0.99, top=0.9, wspace=0.2, hspace=0.27)
	pl.suptitle('Interstadial volcanos')
	ax=pl.subplot(241)
	pl.fill_between(range(-M,M),ngrip_perc14_gi,ngrip_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_ngrip_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'NGRIP',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(242)
	pl.fill_between(range(-M,M),grip_perc14_gi,grip_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_grip_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'GRIP',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(243)
	pl.fill_between(range(-M,M),gisp2_perc14_gi,gisp2_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_gisp2_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'GISP2',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(244)
	pl.fill_between(range(-M,M),neem_perc14_gi,neem_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_neem_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'NEEM',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(245)
	pl.fill_between(range(-M,M),stack_perc14_gi,stack_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_stack_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'Stack',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)');pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(246)
	pl.fill_between(range(-M,M),wais_perc14_gi,wais_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_wais_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'WAIS',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
	ax=pl.subplot(247)
	pl.fill_between(range(-M,M),edml_perc14_gi,edml_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_edml_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'EDML',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
	ax=pl.subplot(248)
	pl.fill_between(range(-M,M),edc_perc14_gi,edc_perc86_gi ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_edc_gi);pl.grid(axis='both')
	pl.text(0.95,0.95,'EDC',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')

	fig=pl.figure(figsize=(15,5))
	pl.subplots_adjust(left=0.1, bottom=0.12, right=0.99, top=0.9, wspace=0.2, hspace=0.27)
	pl.suptitle('Stadial volcanos')
	ax=pl.subplot(241)
	pl.fill_between(range(-M,M),ngrip_perc14_gs,ngrip_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_ngrip_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'NGRIP',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(242)
	pl.fill_between(range(-M,M),grip_perc14_gs,grip_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_grip_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'GRIP',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(243)
	pl.fill_between(range(-M,M),gisp2_perc14_gs,gisp2_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_gisp2_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'GISP2',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(244)
	pl.fill_between(range(-M,M),neem_perc14_gs,neem_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_neem_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'NEEM',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	ax=pl.subplot(245)
	pl.fill_between(range(-M,M),stack_perc14_gs,stack_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_stack_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'Stack',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)');pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(246)
	pl.fill_between(range(-M,M),wais_perc14_gs,wais_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_wais_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'WAIS',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
	ax=pl.subplot(247)
	pl.fill_between(range(-M,M),edml_perc14_gs,edml_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_edml_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'EDML',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
	ax=pl.subplot(248)
	pl.fill_between(range(-M,M),edc_perc14_gs,edc_perc86_gs ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), resp_edc_gs);pl.grid(axis='both')
	pl.text(0.95,0.95,'EDC',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
'''


'''
	for i in range(len(volcs_grip)):
		idx = find_nearest(t_ngrip, volcs_grip[i])
		resp_grip_match_all[i,:] = detrend(grip[idx-M:idx+M]) 

	for i in range(len(volcs_grip0)):
		idx = find_nearest(t_ngrip, volcs_grip0[i])
		resp_grip_0_all[i,:] = detrend(grip[idx-M:idx+M]) 

	for i in range(len(volcs_gisp2)):
		idx = find_nearest(t_ngrip, volcs_gisp2[i])
		resp_gisp2_match_all[i,:] = detrend(gisp2[idx-M:idx+M]) 

	for i in range(len(volcs_gisp20)):
		idx = find_nearest(t_ngrip, volcs_gisp20[i])
		resp_gisp2_0_all[i,:] = detrend(gisp2[idx-M:idx+M]) 

	for i in range(len(volcs_neem)):
		idx = find_nearest(t_ngrip, volcs_neem[i])
		resp_neem_match_all[i,:] = detrend(neem[idx-M:idx+M]) 

	for i in range(len(volcs_neem0)):
		idx = find_nearest(t_ngrip, volcs_neem0[i])
		resp_neem_0_all[i,:] = detrend(neem[idx-M:idx+M]) 

	grip_match_mean, grip_match_perc14, grip_match_perc86 = mean_confidence_band(resp_grip_match_all)
	grip_0_mean, grip_0_perc14, grip_0_perc86 = mean_confidence_band(resp_grip_0_all)
	gisp2_match_mean, gisp2_match_perc14, gisp2_match_perc86 = mean_confidence_band(resp_gisp2_match_all)
	gisp2_0_mean, gisp2_0_perc14, gisp2_0_perc86 = mean_confidence_band(resp_gisp2_0_all)
	neem_match_mean , neem_match_perc14, neem_match_perc86 = mean_confidence_band(resp_neem_match_all)
	neem_0_mean, neem_0_perc14, neem_0_perc86 = mean_confidence_band(resp_neem_0_all)


	#fig=pl.figure()
	#for i in range(len(resp_ngrip_all_gs)):
	#        pl.plot(range(-30,30), resp_ngrip_all_gs[i], color='gray')

	fig=pl.figure(figsize=(10,11))
	pl.subplots_adjust(left=0.12, bottom=0.08, right=0.99, top=0.98, wspace=0.2, hspace=0.27)
	ax=pl.subplot(321)
	pl.fill_between(range(-M,M),grip_match_perc14,grip_match_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), grip_match_mean);pl.grid(axis='both')
	pl.text(0.95,0.95,'GRIP matched',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(322)
	pl.fill_between(range(-M,M),grip_0_perc14,grip_0_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), grip_0_mean);pl.grid(axis='both')
	pl.text(0.95,0.95,'GRIP others',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)

	ax=pl.subplot(323)
	pl.fill_between(range(-M,M),gisp2_match_perc14,gisp2_match_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), gisp2_match_mean);pl.grid(axis='both')
	pl.text(0.95,0.95,'GISP2 matched',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.ylabel('$\delta^{18}$O anomaly')
	ax=pl.subplot(324)
	pl.fill_between(range(-M,M),gisp2_0_perc14,gisp2_0_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), gisp2_0_mean);pl.grid(axis='both')
	pl.text(0.95,0.95,'GISP2 others',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)

	ax=pl.subplot(325)
	pl.fill_between(range(-M,M),neem_match_perc14,neem_match_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), neem_match_mean);pl.grid(axis='both')
	pl.text(0.95,0.95,'NEEM matched',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.ylabel('$\delta^{18}$O anomaly');pl.xlabel('Time (years)')
	ax=pl.subplot(326)
	pl.fill_between(range(-M,M),neem_0_perc14,neem_0_perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-M,M), neem_0_mean);pl.grid(axis='both')
	pl.text(0.95,0.95,'NEEM other',horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=15)
	pl.xlabel('Time (years)')
'''


'''
	p_ngrip, p_grip, p_gisp2, p_t = np.loadtxt('matchpoints_gicc05.txt', skiprows=2, unpack=True)
	p_ngrip2, p_neem, match_neem = np.loadtxt('matchpoints_gicc05_neem.txt', skiprows=2, unpack=True)

	### remove all matchpoints in between GRIP and GISP2 only
	idx= ~np.isnan(p_ngrip)
	p_ngrip = p_ngrip[~np.isnan(p_ngrip)]

	### matchpoints NGRIP->GRIP
	match_grip = p_t[idx]
	p_grip = p_grip[idx]
	idx0= ~np.isnan(p_grip)
	match_grip = match_grip[idx0]
	### matchpoints NGRIP->GISP2
	match_gisp2 = p_t[idx]
	p_gisp2 = p_gisp2[idx]
	idx0= ~np.isnan(p_gisp2)
	match_gisp2 = match_gisp2[idx0]
	
	for i in range(N):
		idx = np.abs(match_neem-volcs[i]).argmin()
		if np.abs(match_neem[idx]-volcs[i])<10.:
		        volcs_neem.append(volcs[i])
		else:
		        volcs_neem0.append(volcs[i])

		idx = np.abs(match_gisp2-volcs[i]).argmin()
		if np.abs(match_gisp2[idx]-volcs[i])<10.:
		        volcs_gisp2.append(volcs[i])
		else:
		        volcs_gisp20.append(volcs[i])

		idx = np.abs(match_grip-volcs[i]).argmin()
		if np.abs(match_grip[idx]-volcs[i])<10.:
		        volcs_grip.append(volcs[i])
		else:
		        volcs_grip0.append(volcs[i])
		        
	#resp_grip_match_all = np.empty((len(volcs_grip),2*M)); resp_grip_0_all = np.empty((len(volcs_grip0),2*M))
	#resp_gisp2_match_all = np.empty((len(volcs_gisp2),2*M)); resp_gisp2_0_all = np.empty((len(volcs_gisp20),2*M))
	#resp_neem_match_all = np.empty((len(volcs_neem),2*M)); resp_neem_0_all = np.empty((len(volcs_neem0),2*M))
'''

'''
### CUSTOM response curves for events before DO events (no detrending)
idcs_DO=[9, 22, 24, 40, 44, 64, 71]
i=0
sum_curve_all = np.empty(2*M)
fig=pl.figure()
for idx in idcs_DO:
	pl.subplot(4,2,i+1)
	
	t_idx = find_nearest(t_ngrip, volcano_times['NGRIP'][idx]-shifts['NGRIP'])
	segment = records['NGRIP'][t_idx-M:t_idx+M]
	segment = segment - np.mean(segment[M+b:])#M+lead:
	
	j=0
	for core in ['GRIP', 'GISP2', 'NEEM']:
		t_idx = find_nearest(t_ngrip, volcano_times[core][idx]-shifts[core])
		seg0 = records[core][t_idx-M:t_idx+M]
		seg0 = seg0 - np.mean(seg0[M+b:])
		if (core=='GRIP' and mask_grip[idx]==1.):
			#print('In GRIP')
			j+=1
			segment += seg0
		if (core=='GISP2' and mask_gisp2[idx]==1.):
			#print('In GISP2')
			j+=1
			segment += seg0
		if (core=='NEEM' and mask_neem[idx]==1.):
			#print('In NEEM')
			j+=1
			segment += seg0
	segment = segment/(j+1.)
	
	sum_curve_all+=segment
	pl.plot(range(-M,M), segment)
	print(volcs[idx], np.mean(segment[M-a:M+b]), means_stacked[idx])#, np.mean(segment[M+b:]))
	i+=1
pl.subplot(4,2,8)
pl.plot(range(-M,M), sum_curve_all/8.)
'''

if __name__ == '__main__':
        MAIN()

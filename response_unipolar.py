import matplotlib.pyplot as pl
import numpy as np
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import KernelDensity
from scipy.signal import detrend
from scipy.ndimage import filters
from scipy.signal import gaussian
from math import factorial as fact
import math
from panel_labels import letter_subplots
from pylab import polyval, linspace, percentile, poly1d
from scipy.optimize import curve_fit
from random import randrange

#from awkde import GaussianKDE
from ssvkernel import ssvkernel

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

	M = 50 ### time before eruption to include in response
	k = 50 ### time after eruption to include in response
	### a,b years before and after to calculate mean anomaly from response curve.
	#a = 3; b = 7 #wais WHY?? b=7...???
	a = 10; b = 20 #edc
	#a = 7; b = 3 #  # #ngrip a = 4; b = 0 #
	#a = 10; b = 10
	#a = 5; b = 15 #neem and gisp2
	
	#depth, mag, volcs = np.loadtxt('volcanos_wais_newages.txt', unpack=True)
	#depth, _, volcs, mag = np.loadtxt('volcanos_wais_newages_avg.txt', unpack=True)
	
	#depth, mag, volcs = np.loadtxt('volcanos_wais_wd2014.txt', unpack=True)
	#depth, mag, volcs = np.loadtxt('volcanos_edml_newages.txt', unpack=True)
	#depth, mag, volcs = np.loadtxt('volcanos_edc99.txt', unpack=True) ### OLD
	#depth, mag, volcs = np.loadtxt('volcanos_edc_old.txt', unpack=True) ### OLD
	#depth, mag, volcs = np.loadtxt('volcanos_edc96.txt', unpack=True) ### OLD
	#depth, mag, volcs = np.loadtxt('volcanos_edc_splice.txt', unpack=True)
	
	depth, _, volcs, mag = np.loadtxt('volcanos_edc_splice_avg.txt', unpack=True)

	#depth, mag, volcs = np.loadtxt('volcanos_ngrip_newages.txt', unpack=True)
	## as magnitudes use the average deposition of all cores, instead of just NGRIP
	#depth, _, volcs, mag = np.loadtxt('volcanos_ngrip_newages_avg.txt', unpack=True)
	
	#depth, mag, volcs = np.loadtxt('volcanos_neem_newages.txt', unpack=True)
	#depth, _, volcs, mag = np.loadtxt('volcanos_neem_newages_avg.txt', unpack=True)
	#depth, mag, volcs = np.loadtxt('volcanos_gisp2_newages.txt', unpack=True)
	
	t_wais, wais = np.loadtxt('wais_d18o_highres_1y_volc.txt')	
	#t_wais, wais = np.loadtxt('wais_d18o_highres_1y_wd2014.txt')	
	t_edc, edc = np.loadtxt('edc_d18o_1y_volc.txt')
	#t_edml, edml = np.loadtxt('edml_d18o_1y_volc.txt')
	#t_wais=t_wais[11677:]; wais=wais[11677:]
	t_neem, neem = np.loadtxt('neem_d18o_1y_bottom_corr.txt')
	#t_gisp2, gisp2 = np.loadtxt('gisp2_d18o_1y_6k.txt')
	
	### NGRIP: merge Holocene (NGRIP1) and Glacial (NGRIP2) on high resolution.
	t_ngrip, ngrip = np.loadtxt('ngrip_d18o_1y_9_5k.txt')
	ngrip_hol, t_ngrip_hol = np.loadtxt('ngrip_d18O_holocene_highres_uneven.txt', unpack=True)
	### remove jumps in NGRIP early holocene record (<2ka)
	ngrip_new = []; t_ngrip_new = []; i=0
	while i<14232:
		t_ngrip_new.append(t_ngrip_hol[i])
		if t_ngrip_hol[i+1]==t_ngrip_hol[i]:
			ngrip_new.append((ngrip_hol[i+1]+ngrip_hol[i])/2.)
			i+=2
		else:
			ngrip_new.append(ngrip_hol[i])
			i+=1

	ngrip_hol = np.concatenate((ngrip_new, ngrip_hol[14232:]))
	t_ngrip_hol = np.concatenate((t_ngrip_new, t_ngrip_hol[14232:]))
	def process_record(t, data):
		spl = UnivariateSpline(t, data, s=0, k=1)
		timegrid = np.arange(min(t_ngrip_hol), max(t_ngrip_hol), 1./12.)
		interpol = spl(timegrid)
		timegrid = timegrid[9:]; interpol = interpol[9:]
		interpol_1y = [np.mean(interpol[12*j:12*(j+1)]) for j in range(int(len(interpol)/12.))]
		timegrid_1y = [int(np.mean(timegrid[12*j:12*(j+1)])) for j in range(int(len(interpol)/12.))]
		return timegrid_1y, interpol_1y
	
	timegrid_ngrip, ngrip_interpol = process_record(t_ngrip_hol, ngrip_hol)
	ngrip = ngrip[291:]; t_ngrip = t_ngrip[291:]
	#fig=pl.figure()
	#pl.plot(t_ngrip, ngrip, 'x-')
	#pl.plot(t_ngrip_hol, ngrip_hol, 'x-')
	#pl.plot(timegrid_ngrip, ngrip_interpol, 'x-')
	ngrip = np.concatenate((ngrip_interpol, ngrip))
	t_ngrip = np.concatenate((timegrid_ngrip, t_ngrip))
	#pl.plot(t_ngrip, ngrip, '--')
	
	#isotope_response(depth, mag, volcs, t_edml, edml, a, b, k, M)
	#isotope_response(depth, mag, volcs+1.5+1.5, t_wais, wais, a, b, k, M)
	isotope_response(depth, mag, volcs+1.5+1.5, t_edc, edc, a, b, k, M)
	#isotope_response(depth, mag, volcs+1.+1.5, t_ngrip, ngrip, a, b, k, M)
	#isotope_response(depth, mag, volcs + 1.5, t_neem, neem, a, b, k, M)
	#isotope_response(depth, mag, volcs, t_gisp2, gisp2, a, b, k, M)
	
	### Compare distribution of GS vs GI magnitudes in datasets averaged over cores.
	#magnitude_gs_gi()
	
	### For DO study.
	#mag_bipolar, mag_nonbipolar, mag_do, age_bipolar = simple_upper_limit()
	#print(np.sort(mag_nonbipolar))
	#print(np.sort(mag_bipolar))
	#dist_matching_limit(mag_bipolar, mag_nonbipolar, mag_do, age_bipolar)
	

	pl.show()


def response_mag_match(volcs_gi0, volcs_gs0, mag_gi0, mag_gs0, M, a, b, k0, t, data):

	### NGRIP
	#mag_gs = mag_gs0; mag_gi = mag_gi0
	#volcs_gs = volcs_gs0; volcs_gi = volcs_gi0
	
	### NEEM -> Reverse roles of GI and GS? Not necessarily, but it makes a difference.
	### EDC
	mag_gs = mag_gi0; mag_gi = mag_gs0
	volcs_gs = volcs_gi0; volcs_gi = volcs_gs0

	### remove 2 largest GI events for better matching of proposal and target.
	#mag_gi.remove(max(mag_gi))
	### 1 for EDC
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
	perc = [0., 10, 20., 30., 60., 75., 85., 100.] ### EDC GI
	
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
		N0 = 20000
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
	grid_pdf_gi = np.linspace(2., max(np.log(mag_gi))+.5, 1000)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/6.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(grid_pdf_gi[:, np.newaxis])
	dens_gi = np.exp(log_dens)
	
	grid_pdf_gs = np.linspace(2., max(np.log(mag_gs))+.5, 1000)
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
	axes[0].plot(grid_pts_sampled, pdf_sampled, '--', color='royalblue', label='Sampled')
	axes[0].set_xlabel('Magnitude (kg/km$^2$)'); axes[0].set_ylabel('PDF'); axes[0].legend(loc='best')
	
	axes[1].plot(grid_cdf, cdf_target, color='black', label='GI')
	axes[1].plot(grid_cdf_prop, cdf_proposal, color='tomato', label='GS')
	axes[1].plot(grid_cdf_sample, cdf_target_sample, '--', color='royalblue', label='sampled')
	axes[1].legend(loc='best'); axes[1].set_xlabel('Magnitude (kg/sqkm)'); axes[1].set_ylabel('CDF')
	

	
	#pl.subplot(122)
	#pl.hist(accept_no, int(max(accept_no)-min(accept_no)), density=True, color='gray')
	#pl.ylabel('PDF'); pl.xlabel('Number of eruptions')
	
	resp, means0, base0 = calc_response(volcs_gi, t, data, len(volcs_gi), M, a, b, k0)
	resp_curve_gi, perc14_gi, perc86_gi = mean_confidence_band(np.asarray(resp))
	resp, means0, base0 = calc_response(volcs_gs, t, data, len(volcs_gs), M, a, b, k0)
	resp_curve_gs, perc14_gs, perc86_gs = mean_confidence_band(np.asarray(resp))
	
	print(len(volcs_resamples_all))
	#print(volcs_resamples_all)
	
	resp, means0, base0 = calc_response(volcs_resamples_all, t, data, len(volcs_resamples_all), M, a, b, k0)
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

def eruptions_gs_gi(volcanos, mag):

	### NOTE GI-2.1 is not included... add manually.
	### visually onset of GI-2.2 at 23,400; termination at 23,220
	### GI-2.1 at 23,000, termination at 22,880

	stack_onsets = np.loadtxt('stack_onsets.txt')[::-1]
	stack_terminations = np.loadtxt('stack_downtimings.txt')
	stack_terminations = stack_terminations[8:-1]; stack_terminations = stack_terminations[::-1]
	stack_terminations=np.append([12826., 22880.], stack_terminations)
	stack_terminations=np.append(stack_terminations,60000.)
	stack_onsets=np.append([11700., stack_onsets[0], 23000.], stack_onsets[1:])
	
	#print(stack_onsets)
	#print(stack_terminations)
	
	gi = []; gi_dur = []; gs_dur = []#; gi_probs = []; gs_probs = []
	gs = [[11700., stack_terminations[0]]]
	for i in range(len(stack_onsets)-1):
		gi.append([stack_terminations[i], stack_onsets[i+1]])
		gs.append([stack_onsets[i+1], stack_terminations[i+1]])
		gi_dur.append(stack_onsets[i+1]-stack_terminations[i])
	
	for i in range(len(stack_onsets)):
		gs_dur.append(stack_terminations[i]-stack_onsets[i])
	
	count_gi=0; dur_tot = 0; waiting_gi = []
	volcs_gi = []; mag_gi = []
	for i in range(len(gi)):
		indcs = np.where(np.logical_and(volcanos>=gi[i][0], volcanos<=gi[i][1]))
		if len(indcs[0])>1:
			waiting_gi.append(volcanos[indcs[0][1:]]-volcanos[indcs[0][:-1]])
		#print(gi[i])
		#print(gi_dur[i])
		#print(volcanos[indcs[0]])
		volcs_gi.append(volcanos[indcs[0]])
		mag_gi.append(mag[indcs[0]])
		dur_tot+= gi_dur[i]
		#print(dur_tot)
		#print('GI %s duration: %s; Volcanos: %s; Prob.: %s'%(evt_label[i], round(gi_dur[i],1), len(indcs[0]),  round(poisson_cdf(gi_dur[i]/1000., len(indcs[0])),4)))
		count_gi+=len(indcs[0])
		#gi_probs.append(poisson_cdf(gi_dur[i]/1000., len(indcs[0])))
		
		
	count_gs=0; dur_tot = 0; waiting_gs = []
	volcs_gs = []; mag_gs = []
	for i in range(len(gs)):
		indcs = np.where(np.logical_and(volcanos>=gs[i][0], volcanos<=gs[i][1]))
		if len(indcs[0])>1:
			waiting_gs.append(volcanos[indcs[0][1:]]-volcanos[indcs[0][:-1]])
		#print(gs[i])
		#print(gs_dur[i])
		#print(volcanos[indcs[0]])
		volcs_gs.append(volcanos[indcs[0]])
		mag_gs.append(mag[indcs[0]])
		dur_tot+= gs_dur[i]
		#print(dur_tot)
		#print('GS %s duration: %s; Volcanos: %s; Prob.: %s'%(evt_label_gs[i], round(gs_dur[i],1), len(indcs[0]),  round(poisson_cdf(gs_dur[i]/1000., len(indcs[0])),4)))
		count_gs+=len(indcs[0])
		#gs_probs.append(poisson_cdf(gs_dur[i]/1000., len(indcs[0])))
	
	print('Volcanos in GI: ', count_gi)
	print('Volcanos in GS: ', len(volcanos)-count_gi)
	print('GI duration: ', round(sum(gi_dur),1))
	print('GS duration: ', round(sum(gs_dur),1))
	print('GI volcanos per kyr: ', 1000.*count_gi/sum(gi_dur))
	print('GS volcanos per kyr: ', 1000.*(len(volcanos)-count_gi)/sum(gs_dur))
	
	return [item for sublist in volcs_gs for item in sublist], [item for sublist in volcs_gi for item in sublist], [item for sublist in mag_gs for item in sublist], [item for sublist in mag_gi for item in sublist]


def isotope_response(depth, mag, volcs, t, data, a, b, k, M):

	'''
	### For EDC: compare response before 43800 and after for the different depth scales.
	idc= find_nearest(43800., volcs)
	volcs1 = volcs[:idc]; volcs2 = volcs[idc:]
	print(len(volcs1), len(volcs2))
	
	fig=pl.figure()
	resp, means0, base0 = calc_response(volcs1, t, data, len(volcs1), M, a, b, k)
	resp_curve, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	pl.plot(range(-k,M), resp_curve)
	resp, means0, base0 = calc_response(volcs2, t, data, len(volcs2), M, a, b, k)
	resp_curve, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	pl.plot(range(-k,M), resp_curve)
	N = len(volcs)
	
	'''
	### Discard Holocene:
	idcx_holo= find_nearest(11700., volcs)
	volcs_holo = volcs[:idcx_holo+1]
	volcs = volcs[idcx_holo+1:] #32 in NGRIP
	mag = mag[idcx_holo+1:]
	N = len(volcs)
	#print(volcs)
	resp, means0, base0 = calc_response(volcs_holo, t, data, len(volcs_holo), M, a, b, k)
	resp_curve_holo, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	
	print('# eruptions after 11.7ka', len(volcs), volcs[0])
	
	'''
	### for NGRIP: remove GI eruptions with deposition <20 kg/km2
	### NOTE THIS IS A DIFFERENT SET WHEN USING THE DEPOSITION AVERAGED OVER CORES!!
	idcs=[]
	for i in range(len(volcs)):
		if mag[i]<20:
			idcs.append(i)
	volcs = list(volcs)
	mag = list(mag)
	print('NGRIP eruptions with <20 deposition', [volcs[idx0] for idx0 in idcs])
	for index in sorted(idcs, reverse=True):
		del volcs[index]
		del mag[index]
	N = len(volcs)
	volcs=np.asarray(volcs)
	mag=np.asarray(mag)
	print(N)
	'''
	
	
	'''
	### for EDML: extract bipolar eruptions to confirm absence of signal.
	edml_bipolar = np.loadtxt('bipolar_volcanos_edml_jiamei.txt')
	volcs_edml_bipolar = []
	k0=0
	for i in range(len(edml_bipolar)):
		if min(abs(edml_bipolar[i]- volcs))<1:#5
			idx = np.argmin(abs(edml_bipolar[i]-volcs))
			k0+=1
			print(k0)
			volcs_edml_bipolar.append(volcs[idx])
			
	resp, means0, base0 = calc_response(volcs_edml_bipolar, t, data, len(volcs_edml_bipolar), M, a, b, k)
	resp_curve_edml, perc14_edml, perc86_edml = mean_confidence_band(np.asarray(resp))
	'''
	
	response_vs_time(volcs, t, data, M, a, b, k, resp_curve_holo, N, mag)
	
	### compare isotopic response of bipolar and unipolar eruptions
	#comp_bipolar(volcs, mag, data, t, M, a, b, k)
	
	### split up in stadial and interstadial subsets (discarding Holocene part)
	volcs_gs, volcs_gi, mag_gs, mag_gi = eruptions_gs_gi(volcs, mag)
	
	### analyze the deposition magnitudes in GS vs GI
	#analyze_magnitudes(volcs_gs, volcs_gi, mag_gs, mag_gi)
	
	### compare diffusion of signal over time in GS vs GI
	#diffusion_gi_gs(volcs_gs, volcs_gi, volcs, mag_gs, mag_gi, t, data, M, a, b, k)
	
	resp, means0, base0 = calc_response(volcs_gi, t, data, len(volcs_gi), M, a, b, k)
	resp_curve_gi, perc14_gi, perc86_gi = mean_confidence_band(np.asarray(resp))
	
	#hypothesis_test(volcs_gi, means0, data, t, a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gs, t, data, len(volcs_gs), M, a, b, k)
	resp_curve_gs, perc14_gs, perc86_gs = mean_confidence_band(np.asarray(resp))
	
	#hypothesis_test(volcs_gs, means0, data, t, a, b, M)
	
	fig, axes=pl.subplots(1,2, figsize=(10.,3.5))
	letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	#pl.subplot(121)
	axes[0].plot(range(-k,M), resp_curve_gi, color='black', label='GI')
	axes[0].plot(range(-k,M), perc14_gi, '--', color='black')
	axes[0].plot(range(-k,M), perc86_gi, '--', color='black')
	axes[0].plot(range(-k,M), resp_curve_gs, color='tomato', label='GS')
	axes[0].plot(range(-k,M), perc14_gs, '--', color='tomato')
	axes[0].plot(range(-k,M), perc86_gs, '--', color='tomato')
	axes[0].legend(loc='best')
	axes[0].set_ylabel('$\delta^{18}$O anomaly');axes[0].set_xlim(-30,50); axes[0].set_xlabel('Time before eruption (years)')
	
	#representative sample where there is equally much stadial and interstadial.
	
	#ta = 31960. # first interstadial
	#tb = 38200. # last point in interstadial
	#ta = 39930. # first interstadial
	#tb = 46880. # last point in interstadial
	ta = 32039.
	tb = 46880.
	
	idxa = find_nearest(ta, np.asarray(volcs_gi))
	idxb = find_nearest(tb, np.asarray(volcs_gi))
	#print(volcs_gi[idxa], volcs_gi[idxb])
	volcs_gi = volcs_gi[idxa:idxb+1]
	mag_gi = mag_gi[idxa:idxb+1]
	
	#ta = 32500. # first stadial
	#tb = 38802. # last point in stadial
	#ta = 40130. # first stadial
	#tb = 48530. # last point in stadial
	ta = 32511.
	tb = 47503.
	idxa = find_nearest(ta, np.asarray(volcs_gs))
	idxb = find_nearest(tb, np.asarray(volcs_gs))
	#print(volcs_gs[idxa], volcs_gs[idxb])
	volcs_gs = volcs_gs[idxa:idxb+1]
	mag_gs = mag_gs[idxa:idxb+1]
	
	print('Stadial eruptions: ', len(volcs_gs))
	print(volcs_gs)
	print('Interstadial eruptions: ', len(volcs_gi))
	print(volcs_gi)
	
	print('Stadial depositions: ', mag_gs)
	print('Interstadial depositions: ', mag_gi)
	#print(volcs_gs[339:413])
	#print(volcs_gi[45:92])
	
	### Test response in GS vs GI when using the same magnitude distributions.
	resp_curve_sample = response_mag_match(volcs_gi, volcs_gs, mag_gi, mag_gs, M, a, b, k, t, data)

	resp, means0, base0 = calc_response(volcs_gi, t, data, len(volcs_gi), M, a, b, k)
	resp_curve_gi, perc14_gi, perc86_gi = mean_confidence_band(np.asarray(resp))
	
	#hypothesis_test(volcs_gi, means0, data, t, a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gs, t, data, len(volcs_gs), M, a, b, k)
	resp_curve_gs, perc14_gs, perc86_gs = mean_confidence_band(np.asarray(resp))
	
	#hypothesis_test(volcs_gs, means0, data, t, a, b, M)
	
	#pl.subplot(122)
	axes[1].plot(range(-k,M), resp_curve_gi, color='black', label='GI')
	#pl.plot(range(-k,M), perc14_gi, '--', color='black')
	#pl.plot(range(-k,M), perc86_gi, '--', color='black')
	axes[1].plot(range(-k,M), resp_curve_gs, color='tomato', label='GS')
	axes[1].plot(range(-k,M), resp_curve_sample, '--', color='green', label='Resample')
	#pl.plot(range(-k,M), perc14_gs, '--', color='tomato')
	#pl.plot(range(-k,M), perc86_gs, '--', color='tomato')
	axes[1].legend(loc='best')
	axes[1].set_ylabel('$\delta^{18}$O anomaly');axes[1].set_xlim(-30,50); axes[1].set_xlabel('Time before eruption (years)')

	### all eruptions.
	resp, means0, base0 = calc_response(volcs, t, data, N, M, a, b, k)
	resp_curve, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	
	### divide based on baseline d18O quantiles.
	quantiles = [15., 30., 45., 60., 75., 90.]
	#quantiles = [14., 28., 42., 56., 70., 84.]
	#quantiles = [12., 24., 36., 48., 60., 72., 84.]
	#quantiles = [13., 26., 39., 52., 65., 78., 91.]
	#quantiles = [10., 20., 30., 40., 50., 60., 70., 80., 90.]
	perc = [np.percentile(base0, x) for x in quantiles]
	perc = np.concatenate(([min(base0)], perc, [max(base0)]))
	print(perc)
	
	#fig=pl.figure()
	
	mean_responses = []
	medians = []; amplitudes = []; durations = []; areas = []
	minima = []
	curves_percentiles = []
	for j in range(1,len(perc)):
		resp_mask=[]; anom_mask=[]; base_mask=[]
		for i in range(len(resp)):
			if perc[j]>=base0[i]>perc[j-1]:
				resp_mask.append(resp[i])
				anom_mask.append(means0[i])
				base_mask.append(base0[i])
		mean_responses.append(np.mean(anom_mask))
		resp_curveb, perc14b, perc86b = mean_confidence_band(np.asarray(resp_mask))
		area, dur, ampl = calc_area(resp_curveb, k)
		print('baselinnneee', len(base_mask))
		print(perc[j-1], perc[j], np.median(base_mask), area, dur, ampl)
		medians.append(np.median(base_mask))
		areas.append(area)
		durations.append(dur)
		amplitudes.append(ampl)
		minima.append(min(resp_curveb))
		curves_percentiles.append(resp_curveb)
		#pl.plot(range(-k,M), resp_curve, linewidth=0.8+0.2*j, label='%i - %i'%(perc[j-1],perc[j]), color='black', alpha=0.2+0.1*j)
		pl.legend(loc='upper right', fontsize=11); pl.xlabel('Time before eruption (years)')
		pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-20,40)
	
	fig=pl.figure(figsize=(5.5,4.))
	pl.subplots_adjust(left=0.15, bottom=0.18, right=0.87, top=0.93, wspace=0.25, hspace=0.3)
	ax=pl.subplot(111)
	#pl.subplot(221)
	#pl.plot(base0, means0, 'o')
	#pl.plot(medians, mean_responses, 'o'); pl.ylabel('Avg.')
	#pl.subplot(222)
	ax.plot(medians, minima, 'o-', color='black')#; pl.ylabel('Min.')
	ax.grid()
	ax.set_ylabel('Minimum $\delta^{18}$O anomaly')
	ax2=pl.twinx(ax)
	#pl.subplot(223)
	##pl.plot(medians, amplitudes, 'o'); pl.ylabel('Ampl.')
	#pl.subplot(224)
	ax2.plot(medians, areas, 's--', markerfacecolor='None', color='green')#; pl.ylabel('Int.')
	ax2.set_ylabel('Integrated anomaly')
	ax.set_xlabel('Median baseline $\delta^{18}$O')
	
	#hypothesis_test(volcs, means0, data, t, a, b, M)
	print('Uncertainty response curve (all eruptions): ', np.std(resp_curve[55:]))
	
	#to_file = np.asarray([[volcs[i], means0[i]] for i in range(len(volcs))])
	#np.savetxt('ngrip_volc_isotopes.txt', to_file)
	
	fig=pl.figure(figsize=(13,9))
	pl.subplot(221)
	pl.fill_between(range(-k,M), perc14, perc86 ,color='gray',alpha=0.25)
	pl.bar(range(-k,M), resp_curve, color='black', width=1.);pl.grid(axis='x', which='major'); pl.grid(axis='y', which='both')
	#pl.plot(range(-k,M), resp_curve_edml)
	#pl.plot(range(-k,M), perc14_edml)
	#pl.plot(range(-k,M), perc86_edml)
	#pl.plot(range(-k,M), resp_curves_gi[name], color='darkorange', linewidth=0.8)
	#pl.plot(range(-k,M), resp_curves_gs[name], color='lightseagreen', linewidth=0.8)
	#pl.plot(range(-k,M), resp_curves_old[name], color='gold', linewidth=0.8)
	#pl.plot(range(-k,M), resp_curves_young[name], color='orangered', linewidth=0.8)
	pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-30,50); pl.xlabel('Time before eruption (years)')
	
	#mag_ranges = [10., 15., 25., 40., 60., 80., 100.]
	
	resp_curve_all = np.copy(resp_curve)
	perc14_all = np.copy(perc14)
	perc86_all = np.copy(perc86)
	
	
	### divide based on magnitude quantiles.
	quantiles = [15., 30., 45., 60., 75., 90.]
	perc = [int(np.percentile(mag, x)) for x in quantiles]
	print(perc)

	#fig=pl.figure(figsize=(9,6))
	pl.subplot(222)
	for j in range(len(perc)):
		resp_mask=[]
		for i in range(len(resp)):
			if mag[i]>perc[j]:
				resp_mask.append(resp[i])
		#print(len(resp_mask))	
		resp_curve, perc14, perc86 = mean_confidence_band(np.asarray(resp_mask))
		pl.plot(range(-k,M), resp_curve, color='black', alpha=0.2+0.1*j, label='>%s'%str(perc[j]))
		pl.grid(axis='x', which='major'); pl.grid(axis='y', which='both'); pl.legend(loc='best', fontsize=11); pl.xlabel('Time before eruption (years)')
		pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-20,40)
		
	resp_curve_largest = resp_curve
	#mag_ranges = [0., 20., 30., 40., 60., 80., 100., 250.]
	

	perc = np.concatenate(([0.], perc, [max(mag)]))
	print(perc)
	
	mean_responses = []
	medians = []; amplitudes = []; durations = []; areas = []
	minima = []
	curves_percentiles = []
	#fig=pl.figure(figsize=(9,6))
	pl.subplot(223)
	for j in range(1,len(perc)):
		resp_mask=[]; anom_mask=[]; mag_mask=[]
		for i in range(len(resp)):
			if perc[j]>=mag[i]>perc[j-1]:
				resp_mask.append(resp[i])
				anom_mask.append(means0[i])
				mag_mask.append(mag[i])
		#print(len(resp_mask))	
		mean_responses.append(np.mean(anom_mask))
		resp_curve, perc14, perc86 = mean_confidence_band(np.asarray(resp_mask))
		area, dur, ampl = calc_area(resp_curve, k)
		print(perc[j-1], perc[j], np.median(mag_mask), area, dur, ampl)
		print('Noise level: ', np.mean(perc14[55:]))
		medians.append(np.median(mag_mask))
		areas.append(area)
		durations.append(dur)
		amplitudes.append(ampl)
		minima.append(min(resp_curve))
		curves_percentiles.append(resp_curve)
		pl.plot(range(-k,M), resp_curve, linewidth=0.8+0.2*j, label='%i - %i'%(perc[j-1],perc[j]), color='black', alpha=0.2+0.1*j)
		pl.legend(loc='upper right', fontsize=11); pl.xlabel('Time before eruption (years)')
		pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-20,40)
		
	#fig=pl.figure()
	pl.subplot(224)
	pl.plot(mag, means0, 'o', color='royalblue')
	for i in range(1,len(perc)):
		pl.plot(np.linspace(perc[i-1], perc[i], 100), 100*[mean_responses[i-1]], color='tomato')
	pl.xlabel('Magnitude'); pl.ylabel('20-year isotopic anomaly')
	
	#print(resp_curve)
	#print(resp_curve[k])
	
	fig=pl.figure(figsize=(7,5))
	pl.subplots_adjust(left=0.15, bottom=0.14, right=0.97, top=0.98, wspace=.2, hspace=0.3)
	pl.subplot(221)
	pl.plot(medians, amplitudes, 'o'); pl.xlabel('Magnitude'); pl.ylabel('Amplitude')
	pl.subplot(222)
	pl.plot(medians, areas, 'o'); pl.xlabel('Magnitude'); pl.ylabel('Area')
	pl.subplot(223)
	pl.plot(medians, durations, 'o'); pl.xlabel('Magnitude'); pl.ylabel('Duration')
	pl.subplot(224)
	pl.plot(medians, minima, 'o'); pl.xlabel('Magnitude'); pl.ylabel('Cooling minimum')
	
	fig,axes=pl.subplots(1,2, figsize=(10,4))
	letter_subplots(axes, letters='a', yoffset=1.05)
	pl.subplots_adjust(left=0.08, bottom=0.18, right=0.99, top=0.9, wspace=0.3, hspace=0.17)
	#for j in range(1,len(perc)):
	#	axes[0].plot(range(-k,M), curves_percentiles[j-1], linewidth=0.8+0.2*j, label='%i - %i'%(perc[j-1],perc[j]), color='black', alpha=0.2+0.1*j)
	#axes[0].legend(loc='upper right', fontsize=11); axes[0].set_xlabel('Time before eruption (years)')
	#axes[0].set_ylabel('$\delta^{18}$O anomaly');axes[0].set_xlim(-20,40)
	
	axes[0].fill_between(range(-k,M), perc14_all, perc86_all ,color='gray',alpha=0.25)
	axes[0].bar(range(-k,M), resp_curve_all, color='black', width=1.);axes[0].grid(axis='x', which='major'); axes[0].grid(axis='y', which='both')
	axes[0].plot(range(-k,M), resp_curve_largest, color='crimson')
	axes[0].set_ylabel('$\delta^{18}$O anomaly');axes[0].set_xlim(-30,50); axes[0].set_xlabel('Time before eruption (years)')
	
	xi, y_fit, lower, upper = lin_regr_uncertainty(medians, areas)
	#rp = spearmanr(medians, areas)[0]
	axes[1].plot(xi, y_fit, color='tomato')
	axes[1].plot(xi, lower, '--')
	axes[1].plot(xi, upper, '--', color='cornflowerblue')
	
	axes[1].plot(medians, areas, 'o', color='black')
	axes[1].set_xlabel('Median magnitude')
	axes[1].set_ylabel('Integrated anomaly')
	axes[1].grid(axis='both')
	axes[1].set_xlim(0,155)
	
	fig,axes=pl.subplots(2,2, figsize=(9,6))
	letter_subplots(letters='a', yoffset=1.08)
	pl.subplots_adjust(left=0.1, bottom=0.15, right=0.97, top=0.93, wspace=0.25, hspace=0.3)
	#axes[0][0].plot(range(-k,M), curves_percentiles[0], label='%i - %i'%(min(mag),perc[1]), color='black')# NGRIP
	axes[0][0].plot(range(-k,M), curves_percentiles[1], label='%i - %i'%(perc[1],perc[2]), color='black')# EDC
	axes[0][0].legend(loc='best')
	axes[0][0].set_ylabel('$\delta^{18}$O anomaly')
	axes[0][0].plot(range(-k,M), len(range(-k,M))*[0.], color='gray', linewidth =0.7)
	#axes[0][0].set_xlim(-20,40)
	#axes[0][0].set_ylim(-1.35,0.5)
	#axes[0][0].set_ylim(-0.65,0.3)
	axes[0][1].plot(range(-k,M), curves_percentiles[3], label='%i - %i'%(perc[3],perc[4]), color='black')
	axes[0][1].legend(loc='best')
	axes[0][1].plot(range(-k,M), len(range(-k,M))*[0.], color='gray', linewidth =0.7)
	#axes[0][1].set_xlim(-20,40)
	#axes[0][1].set_ylim(-1.35,0.5)
	#axes[0][1].set_ylim(-0.65,0.3)
	axes[1][0].plot(range(-k,M), curves_percentiles[5], label='%i - %i'%(perc[5],perc[6]), color='black') #NGRIP
	#axes[1][0].plot(range(-k,M), curves_percentiles[4], label='%i - %i'%(perc[4],perc[5]), color='black') #EDC
	axes[1][0].legend(loc='best')
	axes[1][0].set_ylabel('$\delta^{18}$O anomaly')
	axes[1][0].set_xlabel('Time before eruption (years)')
	axes[1][0].plot(range(-k,M), len(range(-k,M))*[0.], color='gray', linewidth =0.7)
	#axes[1][0].set_xlim(-20,40)
	#axes[1][0].set_ylim(-1.35,0.5)
	#axes[1][0].set_ylim(-0.65,0.3)
	axes[1][1].plot(range(-k,M), curves_percentiles[6], label='%i - %i'%(perc[6],perc[7]), color='black')
	axes[1][1].legend(loc='best')
	axes[1][1].plot(range(-k,M), len(range(-k,M))*[0.], color='gray', linewidth =0.7)
	axes[1][1].set_xlabel('Time before eruption (years)')
	#axes[1][1].set_xlim(-20,40)
	#axes[1][1].set_ylim(-1.35,0.5)
	#axes[1][1].set_ylim(-0.65,0.3)

def response_vs_time(volcs, t, data, M, a, b, k, resp_curve_holo, N, mag):


	resp, means0, base0 = calc_response(volcs, t, data, len(volcs), M, a, b, k)
	resp_curve_all, perc14_all, perc86_all = mean_confidence_band(np.asarray(resp))

	### split up in fourths according to age.
	N1 = int(N/4)
	N2 = int(2*N/4)
	N3 = int(3*N/4)
	
	### COUNT HOW MANY IN GI VS GS FOR EACH SEGMENT.
	### -> Potentially sample/weigh accordingly
	
	resp, means0, base0 = calc_response(volcs[:N1], t, data, len(volcs[:N1]), M, a, b, k)
	resp_curve1, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	print(len(volcs[:N1]))
	
	print('Uncertainty response curve (youngest quarter): ', np.std(resp_curve1[55:]))
	
	area, dur, ampl = calc_area(resp_curve1, k)
	print(area, dur, ampl)
	cut_a = find_nearest(t, volcs[0]-100)
	cut_b = find_nearest(t, volcs[N1]+100)
	#hypothesis_test(volcs[:N1], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M, 'All')
	
	resp, means0, base0 = calc_response(volcs[N1:N2], t, data, len(volcs[N1:N2]), M, a, b, k)
	resp_curve2, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	print(len(volcs[N1:N2]))
	area, dur, ampl = calc_area(resp_curve2, k)
	print(area, dur, ampl)
	cut_a = find_nearest(t, volcs[N1]-100)
	cut_b = find_nearest(t, volcs[N2]+100)
	#hypothesis_test(volcs[N1:N2], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M, 'All')
	
	resp, means0, base0 = calc_response(volcs[N2:N3], t, data, len(volcs[N2:N3]), M, a, b, k)
	resp_curve3, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	print(len(volcs[N2:N3]))
	area, dur, ampl = calc_area(resp_curve3, k)
	print(area, dur, ampl)
	cut_a = find_nearest(t, volcs[N2]-100)
	cut_b = find_nearest(t, volcs[N3]+100)
	#hypothesis_test(volcs[N2:N3], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M, 'All')
	
	resp, means0, base0 = calc_response(volcs[N3:], t, data, len(volcs[N3:]), M, a, b, k)
	resp_curve4, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	print(len(volcs[N3:]))
	area, dur, ampl = calc_area(resp_curve4, k)
	print(area, dur, ampl)
	cut_a = find_nearest(t, volcs[N3]-100)
	cut_b = find_nearest(t, volcs[-1]+100)
	#hypothesis_test(volcs[N3:], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M, 'All')
	

	
	### finer segmentation -> if around 80 eruptions are left per bin this should be fine.
	Nseg = 9#10
	N0 = int(N/Nseg); area_all = []; t_mean = []
	fig=pl.figure()
	for i in range(Nseg):
		resp, means0, base0 = calc_response(volcs[i*N0:(i+1)*N0], t, data, len(volcs[i*N0:(i+1)*N0]), M, a, b, k)
		resp_curveX, perc14, perc86 = mean_confidence_band(np.asarray(resp))
		print(volcs[i*N0], volcs[(i+1)*N0])
		print(len(volcs[i*N0:(i+1)*N0]), area, dur, ampl)
		area, dur, ampl = calc_area(resp_curveX, k)
		area_all.append(area); t_mean.append(np.mean(volcs[i*N0:(i+1)*N0]))
		pl.plot(range(-k,M), resp_curveX)#, label='%s-%s'%(str(round(volcs[0]/1000., 1)), str(round(volcs[N1]/1000., 1))))
	
	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.plot(t_mean, area_all, 'o', color='black')
	pl.plot(t_mean[0], area_all[0], 's', markersize=10, color='crimson')
	xi, y_fit, lower, upper = lin_regr_uncertainty(t_mean[1:], area_all[1:])
	pl.plot(xi, y_fit, color='tomato')
	pl.plot(xi, lower, '--')
	pl.plot(xi, upper, '--', color='cornflowerblue')
	
	pl.xlabel('Center of time window (years b2k)'); pl.ylabel('Integrated anomaly (permil)')
	
	### do the same in running window of 20 eruptions.
	### or rather non-overlapping windows?
	
	'''
	t_layer, layer = np.loadtxt('ngrip_layerthickness_9_5k_uneven.txt')
	spl = UnivariateSpline(t_layer, layer, s=0, k=1)
	t_int = np.arange(11700., 60000., 1.)
	layer_interpol = spl(t_int)
	
	t_res, resolution = np.loadtxt('ngrip_d18o_time_res.txt')
	
	t_stack, stack = np.loadtxt('greenland_d18o_stack_11k.txt')
	t_wais, wais = np.loadtxt('wais_d18o_highres_1y_volc.txt')	
	
	t_obl, ecc, obl, peri, insol, _ = np.loadtxt('milankovitch.txt', skiprows=8, unpack=True)
	t_obl=-t_obl*1000
	
	running_ampl = []; running_time = []; running_mag = []; running_acc = []; running_layer = []; running_res = []
	running_d18o = []
	window = 15
	t_acc, acc = np.loadtxt('kindler14_accum_ngrip.txt', unpack=True)
	for i in range(window,len(volcs)):
		resp, means0, base0 = calc_response(volcs[i-window:i], t, data, len(volcs[i-window:i]), M, a, b, k)
		resp_curve0, perc14, perc86 = mean_confidence_band(np.asarray(resp))
		area, dur, ampl = calc_area(resp_curve0, k)
		running_ampl.append(ampl); running_time.append(np.mean(volcs[i-window:i]))
		running_mag.append(np.median(np.log(mag[i-window:i])))
		idc_acc1 = find_nearest(t_acc, volcs[i-window])
		idc_acc2 = find_nearest(t_acc, volcs[i])
		running_acc.append(np.mean(acc[idc_acc1:idc_acc2]))
		idc_lt1 = find_nearest(t_int, volcs[i-window])
		idc_lt2 = find_nearest(t_int, volcs[i])
		running_layer.append(np.mean(layer[idc_lt1:idc_lt2]))
		idc_res1 = find_nearest(t_res, volcs[i-window])
		idc_res2 = find_nearest(t_res, volcs[i])
		running_res.append(np.mean(resolution[idc_res1:idc_res2]))
		idc_iso1 = find_nearest(t_stack, volcs[i-window])
		idc_iso2 = find_nearest(t_stack, volcs[i])
		running_d18o.append(np.mean(stack[idc_iso1:idc_iso2]))
		#print(volcs[i]-volcs[i-window])
	
	### alternative: use fixed 500-year bins. (to resolve stadials and interstadials)
	### this gives only 5-10 eruptions per data point.
	# need at least 600 year bins, so that there is always at least 2 eruptions in window...
	dt_bin = 600#500
	bins = np.arange(12000., 60600., dt_bin)
	#print(bins)
	binned_ampl = np.empty(len(bins)-1)
	t_bin = bins[:-1] + dt_bin/2
	#print(t_bin)
	#print(volcs)
	for i in range(1,len(bins)):
		if i<len(bins)-1:
			v0 = next(x[0] for x in enumerate(volcs) if x[1] > bins[i-1])
			v1 = next(x[0] for x in enumerate(volcs) if x[1] > bins[i])
			#print(bins[i-1], bins[i])
			#print(volcs[v0:v1])
			resp, means0, base0 = calc_response(volcs[v0:v1], t, data, len(volcs[v0:v1]), M, a, b, k)
			resp_curve0, perc14, perc86 = mean_confidence_band(np.asarray(resp))
			area, dur, ampl = calc_area(resp_curve0, k)
			binned_ampl[i-1] = ampl
			
		else:
			#print(bins[i-1], bins[i])
			#print(volcs[v1+1:])
			resp, means0, base0 = calc_response(volcs[v1+1:], t, data, len(volcs[v1+1:]), M, a, b, k)
			resp_curve0, perc14, perc86 = mean_confidence_band(np.asarray(resp))
			area, dur, ampl = calc_area(resp_curve0, k)
			binned_ampl[i-1] = ampl
		

	def smoothing_filter_gauss(x,stdev=50, N=200):
        	wind = gaussian(stdev,N)# stdev and window size
        	return filters.convolve(x, wind/wind.sum(), mode='nearest')
	stack_low = smoothing_filter_gauss(stack)
	wais_low = smoothing_filter_gauss(wais)
	layer_low = smoothing_filter_gauss(layer_interpol)
	fig=pl.figure(figsize=(12.,10.))
	pl.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.98, wspace=0.3, hspace=0.17)
	pl.subplot(411)
	#pl.plot(t_stack, stack)
	pl.xlim(11000., 60000.); pl.ylabel('$\delta^{18}$O stack')
	pl.plot(t_stack, stack_low, color='black')
	pl.subplot(412)
	pl.plot(t_bin, binned_ampl, 'o')
	pl.plot(running_time, running_ampl, color='black'); pl.xlim(11000., 60000.)
	pl.ylabel('$\delta^{18}$O cooling amplitude')
	pl.subplot(413)
	#pl.plot(t_wais, wais)
	pl.xlim(11000., 60000.)
	#pl.ylabel('$\delta^{18}$O WAIS')
	pl.ylabel('$\delta^{18}$O resolution')
	#pl.plot(t_wais, wais_low, color='black')
	#pl.plot(running_time, running_acc)
	#pl.plot(t_layer, layer)
	#pl.plot(t_int, layer_interpol)
	#pl.plot(t_int, layer_low)
	pl.plot(t_res, resolution)
	#pl.plot(t_obl, obl)
	#pl.plot(t_obl, insol)
	#pl.plot(t_obl, np.sin(peri))
	pl.subplot(414)
	pl.plot(running_time, running_mag, color='tomato'); pl.xlim(11000., 60000.)
	pl.ylabel('Median log NGRIP deposition'); pl.xlabel('Time (years BP)')
	
	fig=pl.figure()
	pl.subplot(221)
	pl.plot(running_acc, running_ampl, 'o')
	pl.subplot(222)
	pl.plot(running_layer, running_ampl, 'o')
	pl.subplot(223)
	pl.plot(running_res, running_ampl, 'o')
	pl.subplot(224)
	pl.plot(running_d18o, running_ampl, 'o')
	'''
	
	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.fill_between(range(-k,M), perc14_all, perc86_all ,color='gray',alpha=0.25)
	pl.bar(range(-k,M), resp_curve_all, color='black', width=1.);pl.grid(axis='x', which='major'); pl.grid(axis='y', which='major')
	
	pl.plot(range(-k,M), 2*M*[0.], color='gray')
	pl.plot(range(-k,M), resp_curve_holo, '--', color='pink', label='Holoc.')
	pl.plot(range(-k,M), resp_curve1, label='%s-%s'%(str(round(volcs[0]/1000., 1)), str(round(volcs[N1]/1000., 1))))
	pl.plot(range(-k,M), resp_curve2, label='%s-%s'%(str(round(volcs[N1]/1000., 1)), str(round(volcs[N2]/1000., 1))))
	pl.plot(range(-k,M), resp_curve3, label='%s-%s'%(str(round(volcs[N2]/1000., 1)), str(round(volcs[N3]/1000., 1))))
	pl.plot(range(-k,M), resp_curve4, label='%s-%s'%(str(round(volcs[N3]/1000., 1)), str(round(volcs[-1]/1000., 1))))
	
	pl.legend(loc='best')
	pl.ylabel('$\delta^{18}$O anomaly');pl.xlim(-30,50); pl.xlabel('Time before eruption (years)')
	#pl.xlim(-20, 40)
	
def comp_bipolar(volcs, mag, data, t, M, a, b, k):

	### Filter out bipolar eruptions
	#volcs_bi = np.loadtxt('bipolar_volcanos_published.txt') #NGRIP
	#idcs=np.argwhere(np.isin(volcs_bi,[12961., 33328., 43327., 59180.])) ### NGRIP
	### event at 43327 is much earlier in NH records. Too lazy to do manually...
	#volcs_bi = np.delete(volcs_bi, idcs)
	
	#volcs_bi = np.loadtxt('bipolar_volcanos_edc.txt')
	_, volcs_bi = np.loadtxt('bipolar_volcanos_edc_jiamei.txt', unpack=True)
	
	### NEED ALSO TO CHECK FOR ERUPTIONS WITH <10 deposition in EDC.
	### -> Just one eruption (Taupo)
	
	k0=0
	idcs_rm = []; volcs_bi_ngrip = []; mag_bi_ngrip = []
	for i in range(len(volcs_bi)):
		if min(abs(volcs_bi[i]- volcs))<5:#5
			idx = np.argmin(abs(volcs_bi[i]-volcs))
			k0+=1
			print(k0, volcs_bi[i], volcs[idx], mag[idx])
			idcs_rm.append(idx)
			volcs_bi_ngrip.append(volcs[idx])
			mag_bi_ngrip.append(mag[idx])
			
	
	### delete some events to see sensitivity of response curve
	#del volcs_bi_ngrip[8]
	#del mag_bi_ngrip[8]
	#del volcs_bi_ngrip[7]
	#del mag_bi_ngrip[7]
	#del volcs_bi_ngrip[6]
	#del mag_bi_ngrip[6]
	#del volcs_bi_ngrip[5]
	#del mag_bi_ngrip[5]
	#del volcs_bi_ngrip[2]
	#del mag_bi_ngrip[2]
	
	volcs_uni = list(volcs)
	mag_uni = list(mag)
	#print(idcs_rm)
	for index in sorted(idcs_rm, reverse=True):
		del volcs_uni[index]
		del mag_uni[index]
	mag_uni = np.asarray(mag_uni)
	mag_bi_ngrip = np.asarray(mag_bi_ngrip)
	print(len(volcs_uni), len(volcs))
	
	print('Blab', len(mag_uni), len(mag))
	
	#for i in range(20):
	#	idx = find_nearest(t, volcs_bi_ngrip[i])
	#	fig=pl.figure()
	#	segment = detrend(data[idx-k:idx+M])
	#	segment - np.mean(segment[k+b:])
	#	#pl.plot(range(-k,M), data[idx-k:idx+M])
	#	pl.plot(range(-k,M), segment)
	

	
	
	
	resp, means0, base0 = calc_response(volcs, t, data, len(volcs), M, a, b, k)
	resp_curve, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	resp, means_uni, base0 = calc_response(volcs_uni, t, data, len(volcs_uni), M, a, b, k)
	resp_curve_uni, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	resp, means_bi, base0 = calc_response(volcs_bi_ngrip, t, data, len(volcs_bi_ngrip), M, a, b, k)
	resp_curve_bi, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	
	fig=pl.figure(figsize=(8.5,3.5))
	pl.subplots_adjust(left=0.13, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	
	pl.subplot(121)
	xgrid = np.linspace(min(means_uni)-0.5*np.std(means_uni), max(means_uni)+0.5*np.std(means_uni),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(means_uni)/4.).fit(np.asarray(means_uni)[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='royalblue', alpha=0.4, label='Unipolar')
	
	xgrid = np.linspace(min(means_bi)-0.5*np.std(means_bi), max(means_bi)+0.5*np.std(means_bi),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(means_bi)/4.).fit(np.asarray(means_bi)[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='Bipolar')
	pl.legend(loc='best')
	pl.xlabel('10-year Isotopic anomaly (permil)'); pl.ylabel('PDF')
	
	
	### get representative sample from unipolar list with same 
	# deposition distribution as the bipolar list,
	# discarding the bipolar data gap.
	# Probabilistic procedure, so many different samples.
	# simply use flat list of all samples
	volcs_sample, mag_sample = quantile_sample(volcs_uni, mag_uni, volcs_bi_ngrip, mag_bi_ngrip)
	
	volcs_sample = [item for sublist in volcs_sample for item in sublist]
	mag_sample = [item for sublist in mag_sample for item in sublist]
	#print(volcs_sample)
	#print(mag_sample)
	
	pl.subplot(122)
	xgrid = np.linspace(min(np.log(mag_uni))-1.*np.std(np.log(mag_uni)), max(np.log(mag_uni))+1.*np.std(np.log(mag_uni)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_uni))/4.).fit(np.asarray(np.log(mag_uni))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='royalblue', alpha=0.4, label='Unipolar')
	
	xgrid = np.linspace(min(np.log(mag_bi_ngrip))-1.*np.std(np.log(mag_bi_ngrip)), max(np.log(mag_bi_ngrip))+1.*np.std(np.log(mag_bi_ngrip)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_bi_ngrip))/4.).fit(np.asarray(np.log(mag_bi_ngrip))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='Bipolar')
	
	xgrid = np.linspace(min(np.log(mag_sample))-1.*np.std(np.log(mag_sample)), max(np.log(mag_sample))+1.*np.std(np.log(mag_sample)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_sample))/4.).fit(np.asarray(np.log(mag_sample))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.plot(xgrid, np.exp(log_dens), label='Sampled')
	
	pl.legend(loc='best')
	pl.xlabel('NGRIP log deposition')#; pl.ylabel('PDF')
	
	
	resp, means, base0 = calc_response(volcs_sample, t, data, len(volcs_sample), M, a, b, k)
	resp_curve_sample, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	
	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	#pl.plot(range(-k,M), resp_curve, label='All')
	pl.plot(range(-k,M), resp_curve_uni, label='Unipolar')
	pl.plot(range(-k,M), resp_curve_bi, label='Bipolar')
	pl.plot(range(-k,M), resp_curve_sample, label='Uni sampled')
	pl.ylabel('$\delta^{18}$O anomaly')#;pl.xlim(-30,50)
	pl.xlabel('Time before eruption (years)')
	pl.legend(loc='best')


def diffusion_gi_gs(volcs_gs, volcs_gi, volcs, mag_gs, mag_gi, t, data, M, a, b, k):

	### estimate SNR in fixed time slices.
	
	cut_divide = 4.#7.
	cut_incr = 48000./cut_divide
	cut_times = np.asarray([12000.+(i*cut_incr) for i in range(int(cut_divide)+1)])
	print(cut_times)
	'''
	snr_all = np.empty(int(cut_divide))
	snr_gs = np.empty(int(cut_divide))
	snr_gi = np.empty(int(cut_divide))
	for i in range(1, len(cut_times)):
		cut_t_a = find_nearest(cut_times[i-1], t)
		cut_t_b = find_nearest(cut_times[i], t)
		cut_a = find_nearest(cut_times[i-1], volcs_gs)
		cut_b = find_nearest(cut_times[i], volcs_gs)
		gs_count = cut_b-cut_a
		resp, means0, base0 = calc_response(volcs_gs[cut_a:cut_b], t, data, len(volcs_gs[cut_a:cut_b]), M, a, b, k)
		snr_gs[i-1] = hypothesis_test(volcs_gs[cut_a:cut_b], means0, data[cut_t_a-100:cut_t_b+100], t[cut_t_a-100:cut_t_b+100], a, b, M, 'GS', plot='False')
		
		cut_a = find_nearest(cut_times[i-1], volcs_gi)
		cut_b = find_nearest(cut_times[i], volcs_gi)
		volcs_gi_sample = volcs_gi[cut_a:cut_b]
		gi_count = cut_b-cut_a
		resp, means0, base0 = calc_response(volcs_gi[cut_a:cut_b], t, data, len(volcs_gi[cut_a:cut_b]), M, a, b, k)
		snr_gi[i-1] = hypothesis_test(volcs_gi[cut_a:cut_b], means0, data[cut_t_a-100:cut_t_b+100], t[cut_t_a-100:cut_t_b+100], a, b, M, 'GI', plot='False')
		
		cut_a = find_nearest(cut_times[i-1], volcs)
		cut_b = find_nearest(cut_times[i], volcs)
		volcs_sample = volcs[cut_a:cut_b]
		all_count = cut_b-cut_a
		resp, means0, base0 = calc_response(volcs[cut_a:cut_b], t, data, len(volcs[cut_a:cut_b]), M, a, b, k)
		snr_all[i-1] = hypothesis_test(volcs[cut_a:cut_b], means0, data[cut_t_a-100:cut_t_b+100], t[cut_t_a-100:cut_t_b+100], a, b, M, 'All', plot='False')
		print(all_count, gi_count, gs_count)
		

	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.plot(cut_times[:-1]+cut_incr/2., snr_all, 'o', color='royalblue')	
	pl.plot(cut_times[:-1]+cut_incr/2., snr_gs, 'o', color='tomato', label='GS')	
	pl.plot(cut_times[:-1]+cut_incr/2., snr_gi, 'o', color='black', label='GI')
	pl.xlim(12000., 60000.)
	pl.xlabel('Time (kyr b2k)'); pl.ylabel('SNR')
	
	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.bar(cut_times[:-1]+250, snr_all, width=cut_incr-500, color='black', alpha=0.5, align='edge')
	#pl.xlim(12000., 60000.)
	pl.xlabel('Time (kyr b2k)'); pl.ylabel('SNR')
	#pl.xticks(cut_times[:-1])
	pl.grid(axis='y')
	'''
	
	
	### do the same in moving window.
	dt = 100.
	cut_divide = 4.#7.
	cut_incr = 48000./cut_divide
	t_wind = np.arange(12000., 60000.+dt-cut_incr, dt)
	print(t_wind)
	snr_all = np.empty(len(t_wind))
	snr_gs = np.empty(len(t_wind))
	snr_gi = np.empty(len(t_wind))
	var_all = np.empty(len(t_wind))
	var_gs = np.empty(len(t_wind))
	var_gi = np.empty(len(t_wind))
	
	gs_number = np.empty(len(t_wind))
	gi_number = np.empty(len(t_wind))
	for i in range(len(t_wind)):

		cut_t_a = find_nearest(t_wind[i], t)
		cut_t_b = find_nearest(t_wind[i]+cut_incr, t)
		
		cut_a = find_nearest(t_wind[i], volcs_gs)
		cut_b = find_nearest(t_wind[i]+cut_incr, volcs_gs)
		gs_count = cut_b-cut_a
		gs_number[i] = gs_count
		resp, means0, base0 = calc_response(volcs_gs[cut_a:cut_b], t, data, len(volcs_gs[cut_a:cut_b]), M, a, b, k)
		snr_gs[i], var_gs[i] = hypothesis_test(volcs_gs[cut_a:cut_b], means0, data[cut_t_a-100:cut_t_b+100], t[cut_t_a-100:cut_t_b+100], a, b, M, 'GS', plot='False')
		
		cut_a = find_nearest(t_wind[i], volcs_gi)
		cut_b = find_nearest(t_wind[i]+cut_incr, volcs_gi)
		gi_count = cut_b-cut_a
		gi_number[i] = gi_count
		resp, means0, base0 = calc_response(volcs_gi[cut_a:cut_b], t, data, len(volcs_gi[cut_a:cut_b]), M, a, b, k)
		snr_gi[i], var_gi[i] = hypothesis_test(volcs_gi[cut_a:cut_b], means0, data[cut_t_a-100:cut_t_b+100], t[cut_t_a-100:cut_t_b+100], a, b, M, 'GI', plot='False')
		
		cut_a = find_nearest(t_wind[i], volcs)
		cut_b = find_nearest(t_wind[i]+cut_incr, volcs)
		all_count = cut_b-cut_a
		resp, means0, base0 = calc_response(volcs[cut_a:cut_b], t, data, len(volcs[cut_a:cut_b]), M, a, b, k)
		snr_all[i], var_all[i] = hypothesis_test(volcs[cut_a:cut_b], means0, data[cut_t_a-100:cut_t_b+100], t[cut_t_a-100:cut_t_b+100], a, b, M, 'All', plot='False')
		print(i, all_count, gi_count, gs_count)
		
	fig=pl.figure(figsize=(5.,3.5))
	#letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	pl.plot(t_wind+cut_incr/2., snr_gi, color='black', label='GI')
	pl.plot(t_wind+cut_incr/2., snr_gs, color='tomato', label='GS')
	pl.plot(t_wind+cut_incr/2., snr_all, color='royalblue', linewidth=2.5)
	pl.xlabel('Time (kyr b2k)'); pl.ylabel('SNR'); pl.legend(loc='best', fontsize=11)
	pl.xlim(12000., 60000.)
	pl.grid(axis='y')
	
	fig=pl.figure(figsize=(5.,3.5))
	#letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	pl.plot(t_wind+cut_incr/2., var_gi, color='black', label='GI')
	pl.plot(t_wind+cut_incr/2., var_gs, color='tomato', label='GS')
	pl.plot(t_wind+cut_incr/2., var_all, color='royalblue', linewidth=2.5)
	pl.xlabel('Time (kyr b2k)'); pl.ylabel('Variability'); pl.legend(loc='best', fontsize=11)
	pl.xlim(12000., 60000.)
	pl.grid(axis='y')
	
	fig=pl.figure(figsize=(5.,3.5))
	#letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	pl.plot(t_wind+cut_incr/2., gi_number, color='black', label='GI')
	pl.plot(t_wind+cut_incr/2., gs_number, color='tomato', label='GS')
	pl.xlabel('Time (kyr b2k)'); pl.ylabel('Number of eruptions')
	pl.xlim(12000., 60000.)
	pl.grid(axis='y')
	
	### split up in fourths according to age.	
	fig, axes=pl.subplots(1,2, figsize=(10.,3.5))
	letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.92, wspace=0.3, hspace=0.17)
	
	N1 = int(len(volcs_gs)/4)
	N2 = int(2*len(volcs_gs)/4)
	N3 = int(3*len(volcs_gs)/4)
	
	print('--- Stadial eruptions youngest to oldest: d18O amplitude vs. mean deposition ---')
	print('Number of eruptions per bin: ', N1)
	resp, means0, base0 = calc_response(volcs_gs[:N1], t, data, len(volcs_gs[:N1]), M, a, b, k)
	resp_curve1, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve1, k)
	print(ampl, np.mean(mag_gs[:N1]))
	cut_a = find_nearest(t, volcs_gs[0]-100)
	cut_b = find_nearest(t, volcs_gs[N1]+100)
	#hypothesis_test(volcs_gs[:N1], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gs[N1:N2], t, data, len(volcs_gs[N1:N2]), M, a, b, k)
	resp_curve2, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve2, k)
	print(ampl, np.mean(mag_gs[N1:N2]))
	cut_a = find_nearest(t, volcs_gs[N1]-100)
	cut_b = find_nearest(t, volcs_gs[N2]+100)
	#hypothesis_test(volcs_gs[N1:N2], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gs[N2:N3], t, data, len(volcs_gs[N2:N3]), M, a, b, k)
	resp_curve3, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve3, k)
	print(ampl, np.mean(mag_gs[N2:N3]))
	cut_a = find_nearest(t, volcs_gs[N2]-100)
	cut_b = find_nearest(t, volcs_gs[N3]+100)
	#hypothesis_test(volcs_gs[N2:N3], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gs[N3:], t, data, len(volcs_gs[N3:]), M, a, b, k)
	resp_curve4, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve4, k)
	print(ampl, np.mean(mag_gs[N3:]))
	cut_a = find_nearest(t, volcs_gs[N3]-100)
	cut_b = find_nearest(t, volcs_gs[-1]+100)
	#hypothesis_test(volcs_gs[N3:], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	axes[0].plot(range(-k,M), 2*M*[0.], color='gray')
	axes[0].plot(range(-k,M), resp_curve1, label='%s-%s'%(str(round(volcs_gs[0]/1000., 1)), str(round(volcs_gs[N1]/1000., 1))))
	axes[0].plot(range(-k,M), resp_curve2, label='%s-%s'%(str(round(volcs_gs[N1]/1000., 1)), str(round(volcs_gs[N2]/1000., 1))))
	axes[0].plot(range(-k,M), resp_curve3, label='%s-%s'%(str(round(volcs_gs[N2]/1000., 1)), str(round(volcs_gs[N3]/1000., 1))))
	axes[0].plot(range(-k,M), resp_curve4, label='%s-%s'%(str(round(volcs_gs[N3]/1000., 1)), str(round(volcs_gs[-1]/1000., 1))))
	axes[0].legend(loc='best', fontsize=11)
	axes[0].set_ylabel('$\delta^{18}$O anomaly');axes[0].set_xlim(-30,50); axes[0].set_xlabel('Time before eruption (years)')
	axes[0].grid()
	
	print('Uncertainty YoungGS response curve: ', np.std(resp_curve1[55:]))
	
	N1 = int(len(volcs_gi)/4)
	N2 = int(2*len(volcs_gi)/4)
	N3 = int(3*len(volcs_gi)/4)
	
	print('--- Interstadial eruptions youngest to oldest: d18O amplitude vs. mean deposition ---')
	print('Number of eruptions per bin: ', N1)
	resp, means0, base0 = calc_response(volcs_gi[:N1], t, data, len(volcs_gi[:N1]), M, a, b, k)
	resp_curve1, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve1, k)
	print(ampl, np.mean(mag_gi[:N1]))
	cut_a = find_nearest(t, volcs_gi[0]-100)
	cut_b = find_nearest(t, volcs_gi[N1]+100)
	#hypothesis_test(volcs_gi[:N1], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gi[N1:N2], t, data, len(volcs_gi[N1:N2]), M, a, b, k)
	resp_curve2, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve2, k)
	print(ampl, np.mean(mag_gi[N1:N2]))
	cut_a = find_nearest(t, volcs_gi[N1]-100)
	cut_b = find_nearest(t, volcs_gi[N2]+100)
	#hypothesis_test(volcs_gi[N1:N2], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gi[N2:N3], t, data, len(volcs_gi[N2:N3]), M, a, b, k)
	resp_curve3, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve3, k)
	print(ampl, np.mean(mag_gi[N2:N3]))
	cut_a = find_nearest(t, volcs_gi[N2]-100)
	cut_b = find_nearest(t, volcs_gi[N3]+100)
	#hypothesis_test(volcs_gi[N2:N3], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	resp, means0, base0 = calc_response(volcs_gi[N3:], t, data, len(volcs_gi[N3:]), M, a, b, k)
	resp_curve4, perc14, perc86 = mean_confidence_band(np.asarray(resp))
	area, dur, ampl = calc_area(resp_curve4, k)
	print(ampl, np.mean(mag_gi[N3:]))
	cut_a = find_nearest(t, volcs_gi[N3]-100)
	cut_b = find_nearest(t, volcs_gi[-1]+100)
	#hypothesis_test(volcs_gi[N3:], means0, data[cut_a:cut_b], t[cut_a:cut_b], a, b, M)
	
	axes[1].grid()
	axes[1].plot(range(-k,M), 2*M*[0.], color='gray')
	axes[1].plot(range(-k,M), resp_curve1, label='%s-%s'%(str(round(volcs_gi[0]/1000., 1)), str(round(volcs_gi[N1]/1000., 1))))
	axes[1].plot(range(-k,M), resp_curve2, label='%s-%s'%(str(round(volcs_gi[N1]/1000., 1)), str(round(volcs_gi[N2]/1000., 1))))
	axes[1].plot(range(-k,M), resp_curve3, label='%s-%s'%(str(round(volcs_gi[N2]/1000., 1)), str(round(volcs_gi[N3]/1000., 1))))
	axes[1].plot(range(-k,M), resp_curve4, label='%s-%s'%(str(round(volcs_gi[N3]/1000., 1)), str(round(volcs_gi[-1]/1000., 1))))
	axes[1].legend(loc='best', fontsize=11)
	#axes[1].set_ylabel('$\delta^{18}$O anomaly');
	axes[1].set_xlim(-30,50); axes[1].set_xlabel('Time before eruption (years)')
	
	print('Uncertainty YoungGI response curve: ', np.std(resp_curve1[55:]))
	

def analyze_magnitudes(volcs_gs, volcs_gi, mag_gs, mag_gi):

	print(len(mag_gs), len(mag_gi))
	
	### remove interstadial eruptions with <20kg/sqkm
	#mag_gi = np.sort(mag_gi)
	#mag_gi = mag_gi[29:]
	
	'''
	### also remove GS-2 part, which may be too dense in small eruptions?
	### remove 16.5 - 24.5 ka in unipolar sample.
	idc1= find_nearest(16500., np.asarray(volcs_gs))
	idc2= find_nearest(24500., np.asarray(volcs_gs))
	volcs_gs = np.concatenate((volcs_gs[:idc1], volcs_gs[idc2:]))
	mag_gs = np.concatenate((mag_gs[:idc1], mag_gs[idc2:]))
	print(len(mag_gs))
	'''

	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	xgrid = np.linspace(min(np.log(mag_gs))-1.*np.std(np.log(mag_gs)), max(np.log(mag_gs))+1.*np.std(np.log(mag_gs)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gs))/8.).fit(np.asarray(np.log(mag_gs))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='GS')
	#pl.xscale('log')
	
	xgrid = np.linspace(min(np.log(mag_gi))-1.*np.std(np.log(mag_gi)), max(np.log(mag_gi))+1.*np.std(np.log(mag_gi)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/8.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='black', alpha=0.4, label='GI')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	#pl.xscale('log')
	
	sorted_gs = np.sort(mag_gs)
	sorted_gi = np.sort(mag_gi)
	
	print(sorted_gs)
	print(sorted_gi)
	
	bins = np.asarray([20., 30., 50., 70., 100.]) ### deposition bins NGRIP/Greenland
	gi_dens = []; gs_dens = []
	for i in range(1,len(bins)):
		cut_gs0 = next(x[0] for x in enumerate(sorted_gs) if x[1] > bins[i-1])
		#print(cut_gs0, sorted_gs[cut_gs0])
		cut_gs1 = next(x[0] for x in enumerate(sorted_gs) if x[1] > bins[i])
		#print(cut_gs1, sorted_gs[cut_gs1])
		#print(cut_gs1-cut_gs0)
		gs_dens.append(cut_gs1-cut_gs0)
		
		cut_gi0 = next(x[0] for x in enumerate(sorted_gi) if x[1] > bins[i-1])
		print(cut_gi0, sorted_gi[cut_gi0])
		cut_gi1 = next(x[0] for x in enumerate(sorted_gi) if x[1] > bins[i])
		print(cut_gi1, sorted_gi[cut_gi1])
		print(cut_gi1-cut_gi0)
		gi_dens.append(cut_gi1-cut_gi0)
	
	gs_dens.append(len(sorted_gs)-cut_gs1)
	gi_dens.append(len(sorted_gi)-cut_gi1)
	#print(len(sorted_gs)-cut_gs1)
	print(len(sorted_gi)-cut_gi1)
	
	#GI duration:  19876.8
	#GS duration:  28423.2
	
	edges = np.asarray([30, 50., 70., 100., 150.])
	
	fig=pl.figure()
	pl.bar(bins, np.asarray(gs_dens)/28.4232, width=0.5*(edges-bins), color='tomato', alpha=0.5, align='edge')
	pl.bar(np.asarray(bins)+0.5*(edges-bins), np.asarray(gi_dens)/19.8768, width=0.5*(edges-bins), color='black', alpha=0.5, align='edge')
	pl.ylim(0, 8.)

	
def magnitude_gs_gi():
	volcs, mag = np.loadtxt('magnitude_age_sh_9k_publ.txt', unpack=True)
	#volcs, mag = np.loadtxt('magnitude_age_nh_9k_publ.txt', unpack=True)
	volcs_gs, volcs_gi, mag_gs, mag_gi = eruptions_gs_gi(volcs, mag)
	
	print(len(volcs))
	print(volcs)
	print(mag)
	print(np.sort(mag))
	

	### remove 16.5 - 24.5 ka in unipolar sample.
	### also remove GS-2 part, which may be too dense in small eruptions?
	idc1= find_nearest(16500., np.asarray(volcs_gs))
	idc2= find_nearest(24500., np.asarray(volcs_gs))
	volcs_gs = np.concatenate((volcs_gs[:idc1], volcs_gs[idc2:]))
	mag_gs = np.concatenate((mag_gs[:idc1], mag_gs[idc2:]))
	print(len(mag_gs))
	
	idc1= find_nearest(16500., np.asarray(volcs))
	idc2= find_nearest(24500., np.asarray(volcs))
	volcs = np.concatenate((volcs[:idc1], volcs[idc2:]))
	mag = np.concatenate((mag[:idc1], mag[idc2:]))
	print(len(mag))
	
	mag_young = mag[:int(len(mag)/2)]
	mag_old = mag[int(len(mag)/2):]
	
	fig=pl.figure(figsize=(8,3.5))
	pl.subplot(121)
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	xgrid = np.linspace(min(np.log(mag_young))-1.*np.std(np.log(mag_young)), max(np.log(mag_young))+1.*np.std(np.log(mag_young)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_young))/8.).fit(np.asarray(np.log(mag_young))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='Younger')
	#pl.xscale('log')
	
	xgrid = np.linspace(min(np.log(mag_old))-1.*np.std(np.log(mag_old)), max(np.log(mag_old))+1.*np.std(np.log(mag_old)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_old))/8.).fit(np.asarray(np.log(mag_old))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='Gold', alpha=0.4, label='Older')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	
	grid_cdf_young, cdf_target_young = calc_cumu_density(np.log(mag_young))
	grid_cdf_old, cdf_target_old = calc_cumu_density(np.log(mag_old))
	pl.subplot(122)
	pl.plot(grid_cdf_young, cdf_target_young, color='tomato')
	pl.plot(grid_cdf_old, cdf_target_old, color='gold')
	pl.grid(axis='both')
	pl.xlabel('Magnitude'); pl.ylabel('CDF')
	
	fig=pl.figure(figsize=(8,3.5))
	pl.subplot(121)
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	xgrid = np.linspace(min(np.log(mag_gs))-1.*np.std(np.log(mag_gs)), max(np.log(mag_gs))+1.*np.std(np.log(mag_gs)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gs))/8.).fit(np.asarray(np.log(mag_gs))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='GS')
	#pl.xscale('log')
	
	xgrid = np.linspace(min(np.log(mag_gi))-1.*np.std(np.log(mag_gi)), max(np.log(mag_gi))+1.*np.std(np.log(mag_gi)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/8.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='black', alpha=0.4, label='GI')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	
	grid_cdf_gs, cdf_target_gs = calc_cumu_density(np.log(mag_gs))
	grid_cdf_gi, cdf_target_gi = calc_cumu_density(np.log(mag_gi))
	pl.subplot(122)
	pl.plot(grid_cdf_gs, cdf_target_gs, color='tomato')
	pl.plot(grid_cdf_gi, cdf_target_gi, color='black')
	pl.grid(axis='both')
	pl.xlabel('Magnitude'); pl.ylabel('CDF')
	
	sorted_gs = np.sort(np.log(mag_gs))
	sorted_gi = np.sort(np.log(mag_gi))
	print(sorted_gs)
	print(sorted_gi)
	#bins = np.asarray([10., 20., 40., 70., 100.]) ### deposition bins Antarctica
	#bins = np.asarray([20., 30., 40., 70., 100., 150.]) ### deposition bins Greenland
	### use bins for log instead: <4; 4-5, 5-6, >6
	#bins = np.asarray([1., 3.5, 4., 5., 6.]) ### Greenland
	bins = np.asarray([1., 3., 3.5, 4., 5.]) ### Antarctica
	
	gi_dens = []; gs_dens = []
	for i in range(1,len(bins)):
		cut_gs0 = next(x[0] for x in enumerate(sorted_gs) if x[1] > bins[i-1])
		#print(cut_gs0, sorted_gs[cut_gs0])
		cut_gs1 = next(x[0] for x in enumerate(sorted_gs) if x[1] > bins[i])
		#print(cut_gs1, sorted_gs[cut_gs1])
		#print(cut_gs1-cut_gs0)
		gs_dens.append(cut_gs1-cut_gs0)
		
		cut_gi0 = next(x[0] for x in enumerate(sorted_gi) if x[1] > bins[i-1])
		print(cut_gi0, sorted_gi[cut_gi0])
		cut_gi1 = next(x[0] for x in enumerate(sorted_gi) if x[1] > bins[i])
		print(cut_gi1, sorted_gi[cut_gi1])
		print(cut_gi1-cut_gi0)
		gi_dens.append(cut_gi1-cut_gi0)
	
	gs_dens.append(len(sorted_gs)-cut_gs1)
	gi_dens.append(len(sorted_gi)-cut_gi1)
	#print(len(sorted_gs)-cut_gs1)
	print(len(sorted_gi)-cut_gi1)
	
	#GI duration:  19876.8
	#GS duration:  28423.2
	
	### make arbitrary equally spaced edges and then label them manually.
	edges = np.asarray([1., 2., 3., 4., 5.])

	
	### get uncertainties on estimates of eruption per kyr of different magnitudes
	### assuming Poisson process
	
	def stirling(x):
		return np.sqrt(2*np.pi*x)*(x/math.e)**x
	
	def get_poisson_percentiles(lamb, dT):
	
		def poisson_approx(x):
			if x==0:
				return ((lamb*dT)**x)/fact(x)*np.exp(-lamb*dT)
				
			else:
				return (lamb*dT/x)**x*np.exp(x)*np.exp(-lamb*dT)
		
		
		norm = np.sum([poisson_approx(x) for x in range(1,400)])
		#print(norm)
		
		def cumu_poisson(k):
			if k<5:
				return np.sum([((lamb*dT)**n)/fact(n)*np.exp(-lamb*dT) for n in range(k+1)])
			else:
				return np.sum([poisson_approx(n)/norm for n in range(k+1)])
			
		j = 0
		p = cumu_poisson(0)
		print(p)
		p0 = p
		while p<0.1:
			j+=1
			p = cumu_poisson(j)
			#print(j, p0, p)
			p0 = p
		
		k0 = j
		while p<0.9:
			j+=1
			p = cumu_poisson(j)
			#print(j, p0, p)
			p0 = p
		k1 = j
		return k0, k1
		
		
	'''
	def func1(x, lamb, dT):
		return (lamb*dT)**x/fact(x)*np.exp(-lamb*dT)
		
	def func2(x, lamb, dT):
		return (lamb*dT/x)**x*np.exp(x)*np.exp(-lamb*dT)
		
	def func3(x, lamb, dT):
		return (lamb*dT*math.e/x)**x*np.sqrt(2*np.pi*x)*np.exp(-lamb*dT)
		
	norm2 = np.sum([func2(x, gs_dens[-2]/28.4232, 28.4232) for x in range(1,100)])
	norm3 = np.sum([func3(x, gs_dens[-2]/28.4232, 28.4232) for x in range(1,100)])
	
	fig=pl.figure()
	pl.plot(range(1,100), [func1(x, gs_dens[-2]/28.4232, 28.4232) for x in range(1,100)], 'o')
	pl.plot(range(1,100), [func2(x, gs_dens[-2]/28.4232, 28.4232)/norm2 for x in range(1,100)], 'x')
	pl.plot(range(1,100), [func3(x, gs_dens[-2]/28.4232, 28.4232)/norm3 for x in range(1,100)], '+')
		
	print(gs_dens[-2]/28.4232)
	k0, k1 = get_poisson_percentiles(gs_dens[0]/28.4232, 28.4232)
	'''
	
	gs_time = 28.4232 #- 8 + 0.24 ### 2x 120y for GI 2.1 and 2.2
	gi_time = 19.8768 #- 0.24
	
	fig=pl.figure(figsize=(5,3.5))
	ax=pl.subplot(111)
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	pl.bar(edges, np.asarray(gs_dens)/gs_time, width=0.4, color='tomato', alpha=0.5, align='edge', label='GS')
	pl.bar(np.asarray(edges)+0.4, np.asarray(gi_dens)/gi_time, width=0.4, color='black', alpha=0.5, align='edge', label='GI')
	#pl.ylim(0, 8.)
	
	for i in range(5):
		k0, k1 = get_poisson_percentiles(gs_dens[i]/gs_time, gs_time)
		pl.plot(2*[1.2+i], [k0/gs_time,k1/gs_time], color='black')
		
	for i in range(5):
		k0, k1 = get_poisson_percentiles(gi_dens[i]/gi_time, gi_time)
		pl.plot(2*[1.6+i], [k0/gi_time,k1/gi_time], color='black')
	
	pl.ylabel('Eruptions per kyr'); pl.xlabel('Magnitude')
	pl.grid(axis='y')
	pl.xticks([1.4, 2.4, 3.4, 4.4, 5.4])
	#ax.set_xticklabels(['<3.5', '3.5-4', '4-5', '5-6', '>6']) ### greenland
	ax.set_xticklabels(['<3', '3-3.5', '3.5-4', '4-5', '>5']) ### antarctica
	pl.tick_params(bottom = False)
	#ax.get_xaxis().set_visible(False)
	#ax.axes.xaxis.set_ticks([])
	pl.minorticks_off()
	pl.legend(loc='best')
	
	#representative sample where there is equally much stadial and interstadial.
	
	ta = 31960. # first interstadial
	tb = 38200. # last point in interstadial
	#ta = 39930. # first interstadial
	#tb = 46880. # last point in interstadial
	idxa = find_nearest(ta, np.asarray(volcs_gi))
	idxb = find_nearest(tb, np.asarray(volcs_gi))
	#print(volcs_gi[idxa], volcs_gi[idxb])
	volcs_gi = volcs_gi[idxa:idxb+1]
	mag_gi = mag_gi[idxa:idxb+1]
	
	ta = 32500. # first stadial
	tb = 38802. # last point in stadial
	#ta = 40130. # first stadial
	#tb = 48530. # last point in stadial
	idxa = find_nearest(ta, np.asarray(volcs_gs))
	idxb = find_nearest(tb, np.asarray(volcs_gs))
	#print(volcs_gs[idxa], volcs_gs[idxb])
	volcs_gs = volcs_gs[idxa:idxb+1]
	mag_gs = mag_gs[idxa:idxb+1]
	
	fig=pl.figure(figsize=(8,3.5))
	pl.subplot(121)
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	xgrid = np.linspace(min(np.log(mag_gs))-1.*np.std(np.log(mag_gs)), max(np.log(mag_gs))+1.*np.std(np.log(mag_gs)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gs))/4.).fit(np.asarray(np.log(mag_gs))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='GS')
	#pl.xscale('log')
	
	xgrid = np.linspace(min(np.log(mag_gi))-1.*np.std(np.log(mag_gi)), max(np.log(mag_gi))+1.*np.std(np.log(mag_gi)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/4.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='black', alpha=0.4, label='GI')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	
	grid_cdf_gs, cdf_target_gs = calc_cumu_density(np.log(mag_gs))
	grid_cdf_gi, cdf_target_gi = calc_cumu_density(np.log(mag_gi))
	pl.subplot(122)
	pl.plot(grid_cdf_gs, cdf_target_gs, color='tomato')
	pl.plot(grid_cdf_gi, cdf_target_gi, color='black')
	pl.grid(axis='both')
	pl.xlabel('Magnitude'); pl.ylabel('CDF')
	
	### BIPOLAR eruptions
	#volcs_bi = np.loadtxt('bipolar_volcanos_published.txt')
	#idcs=np.argwhere(np.isin(volcs_bi,[12961., 33328., 43327., 59180.])) ### NGRIP
	### event at 43327 is much earlier in NH records. Too lazy to do manually...
	#volcs_bi = np.delete(volcs_bi, idcs)
	
	#volcs_bi = np.loadtxt('bipolar_volcanos_edc.txt')
	_, volcs_bi = np.loadtxt('bipolar_volcanos_edc_jiamei.txt', unpack=True)
	volcs_bi = np.loadtxt('age_bipolar_jiamei_sh.txt')
	
	### NEED ALSO TO CHECK FOR ERUPTIONS WITH <10 deposition in EDC.
	### -> Just one eruption (Taupo)
	
	k0=0
	idcs_rm = []; volcs_bi_ngrip = []; mag_bi_ngrip = []
	for i in range(len(volcs_bi)):
		if min(abs(volcs_bi[i]- volcs))<1:#5
			idx = np.argmin(abs(volcs_bi[i]-volcs))
			k0+=1
			print(k0, volcs_bi[i], volcs[idx], mag[idx])
			idcs_rm.append(idx)
			volcs_bi_ngrip.append(volcs[idx])
			mag_bi_ngrip.append(mag[idx])
			
	
	#volcs_uni = list(volcs)
	#mag_uni = list(mag)
	#for index in sorted(idcs_rm, reverse=True):
	#	del volcs_uni[index]
	#	del mag_uni[index]
	#mag_uni = np.asarray(mag_uni)
	#mag_bi_ngrip = np.asarray(mag_bi_ngrip)
	#print(len(volcs_uni), len(volcs))
	
	#print('Blab', len(mag_uni), len(mag))
	
	#print(volcs_bi_ngrip)
	volcs_gs, volcs_gi, mag_gs, mag_gi = eruptions_gs_gi(np.asarray(volcs_bi_ngrip), np.asarray(mag_bi_ngrip))
	
	print(volcs_gs)
	print(volcs_gi)
	
	fig=pl.figure(figsize=(8,3.5))
	pl.subplot(121)
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	xgrid = np.linspace(min(np.log(mag_gs))-1.*np.std(np.log(mag_gs)), max(np.log(mag_gs))+1.*np.std(np.log(mag_gs)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gs))/4.).fit(np.asarray(np.log(mag_gs))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='GS')
	#pl.xscale('log')
	
	xgrid = np.linspace(min(np.log(mag_gi))-1.*np.std(np.log(mag_gi)), max(np.log(mag_gi))+1.*np.std(np.log(mag_gi)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/4.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='black', alpha=0.4, label='GI')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	
	grid_cdf_gs, cdf_target_gs = calc_cumu_density(np.log(mag_gs))
	grid_cdf_gi, cdf_target_gi = calc_cumu_density(np.log(mag_gi))
	pl.subplot(122)
	pl.plot(grid_cdf_gs, cdf_target_gs, color='tomato')
	pl.plot(grid_cdf_gi, cdf_target_gi, color='black')
	pl.grid(axis='both')
	pl.xlabel('Magnitude'); pl.ylabel('CDF')
	
	
	### Global aerosol loading of bipolar eruptions.
	volcs, mag, _ , _= np.loadtxt('bipolar_loading_publ.txt', unpack=True)
	volcs_gs, volcs_gi, mag_gs, mag_gi = eruptions_gs_gi(volcs, mag)
	
	print(volcs_gs)
	print(volcs_gi)
	
	fig=pl.figure(figsize=(8,3.5))
	pl.subplot(121)
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	xgrid = np.linspace(min(np.log(mag_gs))-1.*np.std(np.log(mag_gs)), max(np.log(mag_gs))+1.*np.std(np.log(mag_gs)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gs))/3.).fit(np.asarray(np.log(mag_gs))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='tomato', alpha=0.4, label='GS')
	#pl.xscale('log')
	
	xgrid = np.linspace(min(np.log(mag_gi))-1.*np.std(np.log(mag_gi)), max(np.log(mag_gi))+1.*np.std(np.log(mag_gi)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/3.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='black', alpha=0.4, label='GI')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	
	grid_cdf_gs, cdf_target_gs = calc_cumu_density(np.log(mag_gs))
	grid_cdf_gi, cdf_target_gi = calc_cumu_density(np.log(mag_gi))
	
	pl.subplot(122)
	pl.plot(grid_cdf_gs, cdf_target_gs, color='tomato')
	pl.plot(grid_cdf_gi, cdf_target_gi, color='black')
	pl.grid(axis='both')
	pl.xlabel('Magnitude'); pl.ylabel('CDF')
	
	
def calc_area(resp_curve, k):
	area = 0
	dur = 0
	i=0
	while resp_curve[k-3+i]<0:# for WAIS take 3 years after eruption as 0. NGRIP 1.
		#print(resp_curve[k+i])
		area+=resp_curve[k-1+i]
		i+=1
	dur += i
	i=1
	while resp_curve[k-3-i]<0:
		#print(resp_curve[k-i])
		area+=resp_curve[k-1-i]
		i+=1
	dur += 1
	
	#ampl = (resp_curve[k-1]+resp_curve[k]+resp_curve[k+1])/3.
	#ampl = (resp_curve[k-3]+resp_curve[k-2]+resp_curve[k-1])/3.
	ampl = (resp_curve[k-5]+resp_curve[k-4]+resp_curve[k-3]+resp_curve[k-2]+resp_curve[k-1]+resp_curve[k])/6.
	
	return area, dur, ampl
	
	
def quantile_sample(volcs_uni, mag_uni, volcs_bi, mag_bi):
	
	### in NGRIP remove 2 largest bipolar eruptions to get better matching.
	mag_bi = list(mag_bi)
	#mag_bi.remove(max(mag_bi))
	#mag_bi.remove(max(mag_bi))
	
	print(len(mag_uni))
	### remove 16.5 - 24.5 ka in unipolar sample.
	idc1= find_nearest(16500., np.asarray(volcs_uni))
	idc2= find_nearest(24500., np.asarray(volcs_uni))
	volcs_uni = np.concatenate((volcs_uni[:idc1], volcs_uni[idc2:]))
	mag_uni = np.concatenate((mag_uni[:idc1], mag_uni[idc2:]))
	print(len(mag_uni))

	N = len(mag_uni)
	def sample_proposal():
		x_ind = np.random.multinomial(1, [1./N]*N, size=1)[0]
		x = np.where(x_ind==1)[0]
		x=x[0]
		return x#mag_uni[x]
		
	### choose such that all quantiles are actually non-empty in the target sample!!
	### steps of just one percent are not allowed, since the sample has less than 100 events.
	#perc = [0., 3., 5., 7., 9., 13., 16., 19., 25., 30., 35., 40., 50., 60., 70., 80., 100.]
	#perc = [0., 4., 8., 13., 16., 19., 25., 30., 35., 40., 50., 65., 80., 100.]
	# adjusted after removing 16.5-24.5 gap; NGRIP
	#perc = [0., 4., 8., 13., 16., 19., 25., 30., 37., 45., 55., 66., 80., 100.]
	#perc = [0., 4., 8., 12., 17., 23., 28., 34., 42., 52., 65., 80., 100.]
	
	### EDC. Not very good matching possible, because proposal data set 
	### quite small. That's why we need relatively few bins...
	#perc = [0., 4., 10., 17., 23., 29., 42., 60., 80., 100.]
	perc = [0., 4., 10., 16., 22., 28., 38., 45., 70., 85., 100.]
	
	#print(np.sort(mag_uni))
	#print(np.sort(mag_bi))
	
	print('Largest event in (truncated) target sample: ', max(mag_bi))
	print('Largest event in proposal sample: ', max(mag_uni))
	quantiles = [np.percentile(mag_bi, x) for x in perc]
	print(quantiles)
	print(np.log(np.asarray(quantiles)))
	counts_target = [((quantiles[i-1] <= mag_bi) & (mag_bi < quantiles[i])).sum() for i in range(1,len(perc))]
	densities_target = [counts_target[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
	print(counts_target)
	print(densities_target)
	print(len(mag_bi)/100.)
	
	counts = [((quantiles[i-1] < mag_uni) & (mag_uni < quantiles[i])).sum() for i in range(1,len(perc))]
	idcs = [np.where(np.logical_and(mag_uni>=quantiles[i-1], mag_uni<=quantiles[i])) for i in range(1,len(perc))]
	samples_quant = [np.asarray(mag_uni)[idcs[i]] for i in range(len(idcs))]
	print(perc[1:])
	print(counts)
	### density is counts per 1%-quantile
	densities = [counts[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
	print(densities)
	factors = [1.-min(densities)/densities[i] for i in range(len(densities))]
	print(factors)
	
	reps = 500
	resamples_all = []; accept_no = []
	volcs_resampled_all = []
	for k in range(reps):
		resamples = []
		resamples_volcs = []
		mag_samples = []
		volc_samples = []
		for i in range(N):
			idx_samp = sample_proposal()
			mag_samples.append(mag_uni[idx_samp])
			volc_samples.append(volcs_uni[idx_samp])
		#mag_samples = [mag_uni[sample_proposal()] for i in range(N)]
		counts = [((quantiles[i-1] <= mag_samples) & (mag_samples < quantiles[i])).sum() for i in range(1,len(perc))]
		idcs = [np.where(np.logical_and(mag_samples>=quantiles[i-1], mag_samples<quantiles[i])) for i in range(1,len(perc))]
		samples_quant = [np.asarray(mag_samples)[idcs[i]] for i in range(len(idcs))]
		volcs_quant = [np.asarray(volc_samples)[idcs[i]] for i in range(len(idcs))]
		densities = [counts[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
		if (np.asarray(densities).all() == 0.):
			print('problem')
			continue
		factors = [1.-min(densities)/densities[i] for i in range(len(densities))]
		
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
			resamples_volcs.append(volcs_resamples_quant)
		resamples = [item for sublist in resamples for item in sublist]
		resamples_volcs = [item for sublist in resamples_volcs for item in sublist]
		#print(k, len(resamples))
		accept_no.append(len(resamples))
		resamples_all.append(resamples)
		volcs_resampled_all.append(resamples_volcs)
	
	#print(resamples_all[0])
	#print(volcs_resampled_all[0])
	
	#print(len(resamples))
	
	return volcs_resampled_all, resamples_all
	
	
def simple_upper_limit():
	mag_do = [65.1, 263.9, 88.4, 37.6, 211.5, 68.5, 20.1] #NH
	#mag_do = [46.4, 17.7, 97.5, 68.8, 47.5, 46.3, 33.4] ## SH
	print(np.median(mag_do))
	print(np.mean(mag_do))
	#age_do = [14705.0, 27797.0, 28942.0, 38232.0, 40183.0, 49319.0, 55005.0]

	#age_forc, forc, _ , _= np.loadtxt('bipolar_forcing.txt', unpack=True)
	#age_forc, forc, _ , _= np.loadtxt('bipolar_loading.txt', unpack=True)
	age_forc, forc, _ , _= np.loadtxt('bipolar_loading_publ.txt', unpack=True)
	#forc_do = [20.8, 33.6, 37.2, 18.0, 33.6, 23.0, 10.5] ### old W/m2
	forc_do = [103.8, 168.0, 185.9, 90.2, 168.1, 114.8, 52.6] ### updated aerosol loading (published)
	
	p = bootstrap_mean_test(forc, forc_do)
	print('Mean magnitude before DO events vs. bipolar population (global forcing): %s vs. %s'%(np.mean(forc_do), np.mean(forc)))
	print('Median magnitude before DO events vs. bipolar population (global forcing): %s vs. %s'%(np.median(forc_do), np.median(forc)))
	print('Bootstrap test on mean p = ', p)
	
	age_bipolar = np.loadtxt('age_bipolar_jiamei_nh.txt')
	age, mag = np.loadtxt('magnitude_age_nh_9k_publ.txt', unpack=True)
	
	#age_bipolar = np.loadtxt('age_bipolar_jiamei_sh.txt')
	#age, mag = np.loadtxt('magnitude_age_sh_9k_publ.txt', unpack=True)
	
	print(len(age))
	
	### remove eruptions that are outside of the age interval of the DO paper:
	### 11.7 to 16.5 and 24.5 to 60
	### SH:
	#age = np.concatenate((age[46:112], age[224:]))
	#mag = np.concatenate((mag[46:112], mag[224:]))
	
	### NH
	
	age = np.concatenate((age[94:235], age[433:]))
	mag = np.concatenate((mag[94:235], mag[433:]))
	
	print(age[94], age[235], age[433])
	
	print(len(age))
	
	
	count=0; mag_nonbipolar = []; mag_bipolar = []
	age_bipolar_test = []
	for i in range(len(mag)):
		if np.abs(age[i] - age_bipolar[find_nearest(age[i], age_bipolar)]) > .1:
			mag_nonbipolar.append(mag[i])
			#print(i)
		else:
			#print(age[i-1], age[i], age[i+1], age_bipolar[find_nearest(age[i], age_bipolar)])
			mag_bipolar.append(mag[i])
			age_bipolar_test.append(age[i])
			#if mag[i]>cutoff:
			#	count+=1
				
	print('# bipolar eruptions: ', len(age) - len(mag_nonbipolar))
	print(len(mag_bipolar))
	print(age_bipolar_test)
	### NOTE here that 2 bipolar events are missing in the SH only data set
	### I think this is because the deposition is below the threshold.
	### no big deal of course.
	### For NH there are 3 missing.
	
	print('Mean magnitude of bipolar population: ', np.mean(mag_bipolar))
	print('Median magnitude of bipolar population: ', np.median(mag_bipolar))
	
	mag_sorted = np.sort(mag_nonbipolar)[::-1]
	#cutoff = 46.4 #median SH deposition of eruptions before DO events
	cutoff = 68.5 #median NH deposition of eruptions before DO events.
	#print(mag_sorted)
	
	median_largest = np.median(mag_sorted[:10])
	j=11
	while median_largest>cutoff:
		median_largest = np.median(mag_sorted[:j])
		#print(median_largest)
		j+=1
	
	print('Number of largest non-bipolar eruptions (out of total %s) with median of the DO average: %s'%(len(mag_nonbipolar), j))
	
	mean_largest = np.mean(mag_sorted[:10])
	j=11
	cutoff = np.mean(mag_do) ### mean SH or NH deposition of eruptions before DO events.
	while mean_largest>cutoff:
		mean_largest = np.mean(mag_sorted[:j])
		j+=1
		
	print('Number of largest non-bipolar eruptions (out of total %s) with mean of the DO average: %s'%(len(mag_nonbipolar), j))
	
	p = bootstrap_mean_test(mag_bipolar, mag_do)
	#print('Mean magnitude before DO events vs. bipolar NH population: %s vs. %s'%(107.871, np.mean(mag_bipolar)))
	#print('Bootstrap test on mean p = ', p)
	
	print('Mean magnitude before DO events vs. bipolar SH population: %s vs. %s'%(np.mean(mag_do), np.mean(mag_bipolar)))
	print('Bootstrap test on mean p = ', p)
	
	print(np.sort(forc))
	
	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.hist(forc, 22, density=True, color='black', alpha=0.7)
	for i in range(len(forc_do)):
		pl.plot(100*[forc_do[i]], np.linspace(0., 0.003, 100), color='tomato')
	pl.plot(100*[85.5], np.linspace(0., 0.004, 100), '--', color='gold')
	pl.plot(100*[163.8], np.linspace(0., 0.004, 100), '--', color='royalblue')
	#pl.xlabel('Global climate forcing (W/m2)')
	pl.xlabel('Stratospheric aerosol loading (Tg)')
	pl.ylabel('PDF')
	
	fig=pl.figure(figsize=(5,3.5))
	pl.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.hist(mag_nonbipolar, 50, density=True, color='royalblue', alpha=0.5, label='Unipolar')
	pl.hist(mag_bipolar, 20, density=True, color='black', alpha=0.5, label='Bipolar')
	for i in range(len(forc_do)):
		pl.plot(100*[mag_do[i]], np.linspace(0., 0.01, 100), color='tomato')
	pl.xlabel('Sulfur deposition (kg/km2)'); pl.ylabel('PDF')
	pl.legend(loc='best')
	
	mag_low = mag_sorted[83:]
	mag_high = mag_sorted[:83]
	
	print(np.mean(mag_low), np.mean(mag_high))
	
	fig=pl.figure(figsize=(8,4))
	pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98, wspace=0., hspace=0.17)
	#pl.hist(mag_bipolar, 20, density=True, color='black', alpha=0.5, label='Bipolar')
	pl.hist(mag_nonbipolar, 50, density=True, color='royalblue', alpha=0.5, label='Unipolar')
	pl.plot(100*[np.mean(mag_high)], np.linspace(0.,0.05,100), label='DO pop mean')
	pl.plot(100*[mag_sorted[83]], np.linspace(0.,0.05,100), '--', color='black')

	for i in range(len(forc_do)):
		pl.plot(100*[mag_do[i]], np.linspace(0., 0.01, 100), color='tomato')
	pl.xlabel('Sulfur deposition (kg/km2)'); pl.ylabel('PDF')
	pl.legend(loc='best')
	
	return mag_bipolar, mag_nonbipolar, mag_do, age_bipolar

def dist_matching_limit(mag_bipolar, mag_nonbipolar, mag_do, age_bipolar):
	
	#age_do = [14705.0, 27797.0, 28942.0, 38232.0, 40183.0, 49319.0, 55005.0]
	
	tamb_nh = 39.7; tamb_sh = 45.8
	samalas_nh = 90.4; samalas_sh = 73.4
	
	### remove 2 largest bipolar events for better matching of proposal and target.
	mag_bipolar.remove(max(mag_bipolar))
	mag_bipolar.remove(max(mag_bipolar))
	### 5 for SH eruptions.
	#mag_bipolar.remove(max(mag_bipolar))
	#mag_bipolar.remove(max(mag_bipolar))
	#mag_bipolar.remove(max(mag_bipolar))
	
	#fig=pl.figure()
	#pl.hist(mag_nonbipolar, 20, density=True)
	#pl.hist(mag_bipolar, 20, density=True, alpha=0.5, label='Bipolar')
	#pl.xlabel('Magnitude (kg/sqkm)'); pl.ylabel('Probability'); pl.legend(loc='best')
	
	### rejection sampling:
	# Idea: sample events that comply with the distribution of bipolar events (target)
	# from the distribution of nonbipolar events (proposal).
	# -> How many accepted samples do we get (on average) out of N sampling steps?
	# Take N fixed as the total number of 
	
	# choose random samples from the distribtion of non-bipolar events (proposal distr.)
	# accept sample with probability CDF(bipolar, X) - or ratio of probs?
	# do this N times (N nonbipolar events), to get number of potentially missed bipolar events.
	# repeat the procedure, to get distribution of missed bipolar events
	# check distribution of sampled events.
	
	# OR converse procedure, could be importance sampling.
	# sample from bipolar distribution and reject according to the ratio of probabilities.
	# or rather: weigh the samples by likelihood ratio to get probability of event 
	# over threshold. (importance sampling)
	# probably not relevant here, since we can just evaluate the probability from proposal.
		
	N = len(mag_nonbipolar)
	
	def sample_proposal():
		x_ind = np.random.multinomial(1, [1./N]*N, size=1)[0]
		x = np.where(x_ind==1)[0]
		x=x[0]
		return mag_nonbipolar[x]
		
	#samples_prop = [sample_proposal() for i in range(N)]
	
	### calculate EDF function of target (bipolar events)
	
	grid_cdf, cdf_target = calc_cumu_density(mag_bipolar)
	grid_cdf_prop, cdf_proposal = calc_cumu_density(mag_nonbipolar)
	### append larger data value to proposal CDF (with same probability)
	#grid_cdf_prop = np.concatenate((grid_cdf_prop, [601.]))
	#cdf_proposal = np.concatenate((cdf_proposal, [cdf_proposal[-1]]))

	### Standard KDE.
	grid_pdf = np.linspace(0., max(mag_bipolar)+50., 1000.)
	
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(mag_bipolar)/8.).fit(np.asarray(mag_bipolar)[:, np.newaxis])
	log_dens = kde.score_samples(grid_pdf[:, np.newaxis])
	dens = np.exp(log_dens)
	
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(mag_nonbipolar)/8.).fit(np.asarray(mag_nonbipolar)[:, np.newaxis])
	log_dens = kde.score_samples(grid_pdf[:, np.newaxis])
	dens_nonbipolar = np.exp(log_dens)
	
	### KDE of PDF with adaptive bandwidth.
	grid_pts = np.asarray((np.linspace(0.,max(mag_nonbipolar)+50., 1000),)).T#600.
	grid_pts_cdf = np.asarray((np.linspace(-100.,1500., 10000),)).T
	'''
	kde_prop = GaussianKDE(glob_bw=.25, alpha=.99, diag_cov=False)#alpha=.5#silverman
	kde_prop.fit(np.asarray((mag_nonbipolar,)).T)
	pdf_nonbipolar = kde_prop.predict(grid_pts)
	pdf_nonbipolar_full = kde_prop.predict(grid_pts_cdf)
	
	kde = GaussianKDE(glob_bw=.25, alpha=.99, diag_cov=False)#
	kde.fit(np.asarray((mag_bipolar,)).T)
	pdf_bipolar = kde.predict(grid_pts)
	pdf_bipolar_full = kde.predict(grid_pts_cdf)
	#print(pdf_bipolar)
	'''
	
	### Adaptive bandwidth different algorithm
	#y, t, optw, gs, C, confb95, yb = 
	#y, t, _, _, _, _, _= ssvkernel(np.asarray(mag_bipolar), tin=None)#np.linspace(15.,500., 100)
	
	### according CDFs:
	cdf_bipolar = [np.sum(pdf_bipolar_full[:i]) for i in range(len(pdf_bipolar_full))]
	cdf_nonbipolar = [np.sum(pdf_nonbipolar_full[:i]) for i in range(len(pdf_nonbipolar_full))]
	
	### factor for Rejection sampling:
	#print(pdf_bipolar[-1], pdf_nonbipolar[-1])
	#print(pdf_bipolar[-1]/pdf_nonbipolar[-1])
	#fact = 40.
	
	'''
	### Sampling distribution based on probability/likelihood ratio.
	### -> assign each non-bipolar sample a probability to be chosen.
	probs = [pdf_bipolar[find_nearest(grid_pts, x)]/pdf_nonbipolar[find_nearest(grid_pts, x)] for x in mag_nonbipolar]
	probs = np.asarray(probs)/np.sum(probs)
	
	def sample_ratio():
		x_ind = np.random.multinomial(1, probs, size=1)[0]#[1./N]*N
		x = np.where(x_ind==1)[0]
		x=x[0]
		return mag_nonbipolar[x]
		
	samples_test = [sample_ratio() for i in range(30000)]
		
	rndgen = np.random.RandomState()
	prop_samples1 = kde_prop.sample(n_samples=20000, random_state=rndgen)
	prop_samples0 = []
	for i in range(len(prop_samples1)):
		if (0<prop_samples1[i] and prop_samples1[i]<600):
			prop_samples0.append(prop_samples1[i])
	
	#probs_kde = [pdf_bipolar[find_nearest(grid_pts, x)]/pdf_nonbipolar[find_nearest(grid_pts, x)] for x in prop_samples0]
	#probs_kde = np.asarray(probs_kde)/np.sum(probs_kde)
	
	def sample_ratio_kde():
		x_ind = np.random.multinomial(1, probs_kde, size=1)[0]#[1./N]*N
		x = np.where(x_ind==1)[0]
		x=x[0]
		return prop_samples0[x][0]
		
	#samples_test_kde = [sample_ratio_kde() for i in range(100000)]
	
	'''
	#N0 = 30000 # to create target distribution from proposal
	# Sample new points from KDE model
	#rndgen = np.random.RandomState()#seed=3575
	#prop_samples0 = kde_prop.sample(n_samples=N0, random_state=rndgen)
	#prop_samples = []
	#for i in range(len(prop_samples0)):
	#	if (0<prop_samples0[i] and prop_samples0[i]<600):
	#		prop_samples.append(prop_samples0[i])
			
	
	#probs_kde = [pdf_bipolar[find_nearest(grid_pts, x)]/pdf_nonbipolar[find_nearest(grid_pts, x)] for x in prop_samples]
	#probs_kde = np.asarray(probs_kde)/np.sum(probs_kde)

	#N0 = len(prop_samples)
	
	#accept = []; accept_all = []
	#for i in range(N0):
		#sample = sample_proposal()
	#	sample = prop_samples[i][0]
		### Accept with probability CDF_target
		#cdf_idx = next(x[0] for x in enumerate(grid_cdf) if x[1] > sample)
		#print(sample, grid_cdf[cdf_idx], cdf_target[cdf_idx])
		#if np.random.uniform()<cdf_target[cdf_idx]:
		#	accept.append(sample)
		
		### Accept with probability CDF_proposal*CDF_target, meaning P(not in proposal; & in target)
		#cdf_idx = next(x[0] for x in enumerate(grid_cdf) if x[1] > sample)
		#cdf_idx_prop = next(x[0] for x in enumerate(grid_cdf_prop) if x[1] > sample)
		#if np.random.uniform()<cdf_proposal[cdf_idx_prop]*cdf_target[cdf_idx]: #elif
		#	accept.append(sample)
		
		### Recection sampling
		#pdf_idx = find_nearest(grid_pts, sample)#grid_pdf
		#ratio= dens[pdf_idx]/dens_nonbipolar[pdf_idx]
		#ratio= pdf_bipolar[pdf_idx]/pdf_nonbipolar[pdf_idx]/fact
		#print(sample, ratio)
		#if ratio>1.:
		#	accept.append(sample)
		#if np.random.uniform()<ratio: #elif
		#	accept_all.append(sample)
			
		### Rejecting with likelihood (or probability) ratio:
		#print(probs[i])
		#x_ind = np.random.multinomial(1, probs, size=1)[0]#[1./N]*N
		#x = np.where(x_ind==1)[0]; x=x[0]
		#sample = mag_nonbipolar[x]
		#prob = probs[x]
		#if np.random.uniform()<prob:
		#	accept.append(sample)
		
		
	#print(N0/len(accept))	
	
	'''
	### Accept samples from proposal via rejection sampling (minimal M factor)
	reps = 2000; accept_no = np.empty(reps)
	accept_all = []
	for j in range(reps):
		rndgen = np.random.RandomState()#seed=3575
		accept = []
		for i in range(len(mag_nonbipolar)):
			#flag = 0
			#while flag==0:
			#	sample = kde_prop.sample(n_samples=1, random_state=rndgen)[0][0]
				#print(sample)
			#	if (0<sample and sample<600):
			#		flag = 1
			sample = sample_proposal()
			pdf_idx = find_nearest(grid_pts, sample)
			ratio= pdf_bipolar[pdf_idx]/pdf_nonbipolar[pdf_idx]/fact
			#print(ratio)
			if np.random.uniform()<ratio:
				accept.append(sample)
				accept_all.append(sample)

		print(len(accept))
		accept_no[j] = len(accept)
	
	

	### Accept samples from proposal with probability CDF_target
	reps = 2000; accept_no = np.empty(reps)
	accept_all = []
	for j in range(reps):
		accept = []
		for i in range(len(mag_nonbipolar)):
			sample = sample_proposal()
			cdf_idx = next(x[0] for x in enumerate(grid_cdf) if x[1] > sample)
			#print(sample, grid_cdf[cdf_idx], cdf_target[cdf_idx])
			if np.random.uniform()<cdf_target[cdf_idx]:
				accept.append(sample)
				accept_all.append(sample)

		print(len(accept))
		accept_no[j] = len(accept)

	### Accept samples from proposal with probability ratio Target/Prop (also if >1)
	reps = 100; accept_no = np.empty(reps)
	accept_all = []
	for j in range(reps):
		accept = []
		for i in range(len(mag_nonbipolar)):
			sample = sample_proposal()
			pdf_idx = find_nearest(grid_pts, sample)
			ratio= dens[pdf_idx]/dens_nonbipolar[pdf_idx]
			if np.random.uniform()<ratio:
				accept.append(sample)
				accept_all.append(sample)

		print(len(accept))
		accept_no[j] = len(accept)

	### Accept samples from proposal with probability CDF_target*CDF_proposal
	reps = 2000; accept_no = np.empty(reps)
	accept_all = []
	for j in range(reps):
		accept = []
		for i in range(len(mag_nonbipolar)):
			sample = sample_proposal()
			cdf_idx = next(x[0] for x in enumerate(grid_cdf) if x[1] > sample)
			cdf_idx_prop = next(x[0] for x in enumerate(grid_cdf_prop) if x[1] > sample)
			if np.random.uniform()<cdf_proposal[cdf_idx_prop]*cdf_target[cdf_idx]: #elif
				accept.append(sample)
				accept_all.append(sample)

		print(len(accept))
		accept_no[j] = len(accept)


	
	### Algorithm to find largest subsample CONSISTENT with target distribution.
	### Use KS test to assess consistency
	### Remove sample where discrepancy of two CDFs is highest, with CDFprop>CDFtarg
	### (is probably not optimal; could be better to remove where PDF discrepancy is largest)
	### Stop when KS statistic gets larger again. Or when no sample can be removed anymore with CDFprop>CDFtarg
	
	#oversample target CDF to equidistant grid of 1/10 magnitudes
	grid_cdf = np.concatenate((grid_cdf, [601.]))
	cdf_target = np.concatenate((cdf_target, [cdf_target[-1]]))
	spl = UnivariateSpline(grid_cdf, cdf_target, s=0, k=1)
	grid_interpol = np.arange(min(mag_nonbipolar), 601., 1./10.)
	cdf_bipolar_interpol = spl(grid_interpol)
	
	#stop=0
	#while stop==0:
	
	stop_no = [700, 800, 900, 950, 975, 990, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045, 1050, 1055, 1060]
	
	fig=pl.figure()
	pl.plot(grid_cdf_prop, cdf_proposal, color='tomato')
	pl.plot(grid_interpol, cdf_bipolar_interpol, '--', color='royalblue')
	
	for j in range(len(stop_no)):
		cdf_proposal0 = list(cdf_proposal); grid_cdf_prop0 = list(grid_cdf_prop)
		mag_nonbipolar0 = list(mag_nonbipolar)
		for i in range(stop_no[j]):
			distances = [cdf_proposal0[i] - cdf_bipolar_interpol[find_nearest(grid_interpol, grid_cdf_prop0[i])] for i in range(len(grid_cdf_prop0))]
			idx = np.asarray(distances).argmax()
			print(grid_cdf_prop0[idx])
			mag_nonbipolar0.remove(grid_cdf_prop0[idx])
			#print(len(mag_nonbipolar0))
			grid_cdf_prop0, cdf_proposal0 = calc_cumu_density(mag_nonbipolar0)
		pl.plot(grid_cdf_prop0, cdf_proposal0, color='black')
	
	accept_all = mag_nonbipolar0
	
	print(len(mag_nonbipolar))
	
	'''	
	
	### quantile sampling.
	### randomly sample with replacement from proposal sample
	### divide target sample in quantiles (uneven is ok)
	### estimate density of proposal sample in the quantiles
	### choose lowest density (N/p_range= d) quantile
	### for each other quantile, randomly re-sample by removing samples with prob 1-d/d_samp
	
	#perc = [0., 2.5, 5., 10., 15., 20., 30., 40., 50., 55., 60., 65., 75., 85., 100.]
	#perc = [0., 2.5, 5., 10., 15., 20., 30., 40., 50., 55., 60., 70., 80., 100.]
	
	### choose such that all quantiles are actually non-empty in the target sample!!
	### steps of just one percent are not allowed, since the sample has less than 100 events.
	#perc = [0., 3., 5., 7., 9., 13., 16., 19., 25., 30., 35., 40., 50., 60., 70., 80., 100.]
	### SH
	#perc = [0., 3., 5., 7., 9., 13., 16., 19., 25., 30., 35., 40., 50., 65., 75., 87.5, 100.]
	### NH
	#perc = [0., 3., 5., 7., 9., 12., 16., 21., 25., 30., 35., 40., 50., 60., 70., 85., 100.]
	perc = [0., 2., 4., 6., 9., 12., 16., 21., 25., 30., 40., 50., 60., 70., 85., 100.]
	#perc = [0., 2., 3., 5., 7., 9., 13., 16., 18., 22., 30., 38., 45., 54., 60., 70., 85., 100.]
	
	print('-------------------- Quantile Resampling ---------------------')
	print(np.sort(mag_bipolar))
	
	print('Largest event in (truncated) target sample: ', max(mag_bipolar))
	quantiles = [np.percentile(mag_bipolar, x) for x in perc]
	print(quantiles)
	counts_target = [((quantiles[i-1] <= mag_bipolar) & (mag_bipolar < quantiles[i])).sum() for i in range(1,len(perc))]
	densities_target = [counts_target[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
	print(densities_target)
	print(len(mag_bipolar)/100.)
	
	counts = [((quantiles[i-1] < mag_nonbipolar) & (mag_nonbipolar < quantiles[i])).sum() for i in range(1,len(perc))]
	idcs = [np.where(np.logical_and(mag_nonbipolar>=quantiles[i-1], mag_nonbipolar<=quantiles[i])) for i in range(1,len(perc))]
	samples_quant = [np.asarray(mag_nonbipolar)[idcs[i]] for i in range(len(idcs))]
	print(perc[1:])
	print(counts)
	### density is counts per 1%-quantile
	densities = [counts[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
	print(densities)
	factors = [1.-min(densities)/densities[i] for i in range(len(densities))]
	print(factors)
	
	
	'''
	for i in range(100):
		mag_samples = [sample_proposal() for i in range(N)]
		counts = [((quantiles[i-1] < mag_samples) & (mag_samples < quantiles[i])).sum() for i in range(1,len(perc))]
		idcs = [np.where(np.logical_and(mag_samples>=quantiles[i-1], mag_samples<=quantiles[i])) for i in range(1,len(perc))]
		samples_quant = [np.asarray(mag_samples)[idcs[i]] for i in range(len(idcs))]
		densities = [counts[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
		print(densities)
		if (np.asarray(densities).all() == 0.):
			print('problem')
		factors = [1.-min(densities)/densities[i] for i in range(len(densities))]
	'''
	
	reps = 50000
	resamples_all = []; accept_no = []
	for k in range(reps):
		print(k)
		resamples = []
		mag_samples = [sample_proposal() for i in range(N)]
		### Note in the following one of the limits should be equal or greater/smaller
		counts = [((quantiles[i-1] <= mag_samples) & (mag_samples < quantiles[i])).sum() for i in range(1,len(perc))]
		### Same should be done here, more importantly
		idcs = [np.where(np.logical_and(mag_samples>=quantiles[i-1], mag_samples<quantiles[i])) for i in range(1,len(perc))]
		samples_quant = [np.asarray(mag_samples)[idcs[i]] for i in range(len(idcs))]
		densities = [counts[i]/(perc[i+1]-perc[i]) for i in range(len(perc)-1)]
		if (np.asarray(densities).all() == 0.):
			print('problem')
			continue
		factors = [1.-min(densities)/densities[i] for i in range(len(densities))]
		
		for i in range(len(counts)):
			resamples_quant = []
			samples0 = samples_quant[i]
			for j in range(len(samples0)):
				if np.random.uniform()>factors[i]:
					resamples_quant.append(samples0[j])
			#print(len(resamples_quant)/(perc[i+1]-perc[i]))
			resamples.append(resamples_quant)
		resamples = [item for sublist in resamples for item in sublist]
		#print(len(resamples))
		accept_no.append(len(resamples))
		resamples_all.append(resamples)
		
	
	accept_all = [item for sublist in resamples_all for item in sublist]
	#print(accept_all)
	print('Mean # samples', np.mean(accept_no))
	print('10-, 90-percentile # samples', np.percentile(accept_no, 10.), np.percentile(accept_no, 90.))
	
	### KDE of PDF with adaptive bandwidth.
	#grid_pts_sampled = np.asarray((np.linspace(0.,max(accept_all)+50., 1000),)).T
	#kde_sampled = GaussianKDE(glob_bw=.25, alpha=.99, diag_cov=False)#alpha=.5#silverman
	#kde_sampled.fit(np.asarray((accept_all,)).T)
	#pdf_sampled = kde_sampled.predict(grid_pts_sampled)
	
	grid_pts_sampled = np.linspace(0.,max(accept_all)+50., 1000)
	### SH
	#kde = KernelDensity(kernel='gaussian', bandwidth=8.).fit(np.asarray(accept_all)[:, np.newaxis])
	### NH
	kde = KernelDensity(kernel='gaussian', bandwidth=20.).fit(np.asarray(accept_all)[:, np.newaxis])
	log_dens = kde.score_samples(grid_pts_sampled[:, np.newaxis])
	pdf_sampled = np.exp(log_dens)
	
	fig=pl.figure(figsize=(12,4.5))
	pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.9, wspace=0.2, hspace=0.17)
	pl.suptitle('NH eruptions not identified as bipolar according to magnitude (Quantile matching)')
	#pl.suptitle('NH eruptions not identified as bipolar according to magnitude (Prob. ratio test)')

	
	#fig=pl.figure()
	pl.subplot(121)
	#pl.hist(mag_nonbipolar, 20, density=True, label='Not bipolar', color='black', alpha=0.7)
	#pl.hist(samples_prop, 20, density=True, alpha=0.5)
	#pl.hist(mag_bipolar, 20, density=True, alpha=0.5, label='Bipolar', color='tomato')
	#pl.hist(accept_all, 50, density=True, alpha=0.5, label='Sampled', color='royalblue')
	pl.plot(grid_pts, pdf_nonbipolar, color='black', label='Unipolar')#fact*
	pl.plot(grid_pts, pdf_bipolar, color='tomato', label='Bipolar')
	pl.plot(grid_pts_sampled, pdf_sampled, '--', color='royalblue', label='Sampled')
	#pl.hist(samples_test, 30, density=True, alpha=0.5)
	#pl.hist(samples_test_kde, 100, density=True, alpha=0.5)
	
	pl.xlabel('Sulfate deposition (kg/km$^2$)'); pl.ylabel('Probability'); pl.legend(loc='best')
	
	pl.subplot(122)
	pl.hist(accept_no, int(max(accept_no)-min(accept_no)), density=True, color='gray')
	pl.ylabel('PDF'); pl.xlabel('Number of eruptions')
	
	#fig=pl.figure()
	#pl.plot(grid_pts, pdf_bipolar/pdf_nonbipolar)
	#pl.plot(mag_nonbipolar, probs, 'x')
	#pl.plot(prop_samples0, probs_kde, 'x')
	#pl.hist(samples_test, 30, density=True)
	#pl.hist(samples_test_kde, 30, density=True, alpha=0.5)

	grid_cdf_sample, cdf_target_sample = calc_cumu_density(accept_all)
	fig=pl.figure(figsize=(6,4))
	pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.plot(grid_cdf, cdf_target, color='tomato', label='bipolar')
	pl.plot(grid_cdf_prop, cdf_proposal, color='black', label='unipolar')
	#pl.plot(grid_interpol, cdf_bipolar_interpol, '--')
	#pl.plot(grid_cdf_prop0, distances)
	#pl.plot(grid_cdf_prop0, cdf_proposal0)
	#pl.plot(grid_pts_cdf, cdf_bipolar)
	#pl.plot(grid_pts_cdf, cdf_nonbipolar)
	pl.plot(grid_cdf_sample, cdf_target_sample, '--', color='royalblue', label='sampled')
	pl.legend(loc='best'); pl.xlabel('Magnitude (kg/sqkm)'); pl.ylabel('CDF')
	
	fig=pl.figure(figsize=(6,7))
	pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98, wspace=0., hspace=0.17)
	pl.subplot(211)

	for i in range(len(mag_bipolar)):
		pl.plot(100*[mag_bipolar[i]], np.linspace(0,0.005,100), color='black')
	pl.plot(100*[np.percentile(mag_bipolar, 70.)], np.linspace(0,0.018,100), color='royalblue')
	pl.plot(100*[np.percentile(mag_bipolar, 80.)], np.linspace(0,0.018,100), color='royalblue')
	#for i in range(1,len(perc)-1):
	#	pl.plot(100*[np.percentile(mag_bipolar, perc[i])], np.linspace(0,0.018,100), color='royalblue')
	pl.plot(grid_pts, pdf_bipolar, color='tomato')
	pl.ylabel('Probability')
	pl.xlim(-5,200)
	pl.subplot(212)
	pl.plot(100*[np.percentile(mag_bipolar, 70.)], np.linspace(0,0.07,100), color='royalblue')
	pl.plot(100*[np.percentile(mag_bipolar, 80.)], np.linspace(0,0.07,100), color='royalblue')
	for i in range(len(mag_nonbipolar)):
		pl.plot(100*[mag_nonbipolar[i]], np.linspace(0,0.015,100), color='black')
	pl.plot(grid_pts, pdf_nonbipolar, color='tomato')
	
	pl.xlabel('Magnitude (kg/sqkm)'); pl.ylabel('Probability')
	pl.xlim(-5,200)
	
	'''
	accept = []
	for i in range(N):
		sample = sample_proposal()
		cdf_idx = next(x[0] for x in enumerate(grid_cdf) if x[1] > sample)
		print(sample, grid_cdf[cdf_idx], cdf_target[cdf_idx])
		if np.random.uniform()<cdf_target[cdf_idx]:
			print('accepted')
			accept.append(sample)
	print(len(accept))
	'''
	
def filter_gi_gs(t0):
	### given a time, say whether in GI (=1); GS (=-1) or in between (0)
	gs_end, gi_onset, gi_end, gs_onset = np.loadtxt('loh19_times.txt', unpack=True)
	gs_end=gs_end[::-1]; gi_onset=gi_onset[::-1]
	gs_onset=gs_onset[::-1]; gi_end=gi_end[::-1]
	
	near_gs_onset = gs_onset[np.searchsorted(gs_onset, t0, side='left')]
	near_gi_onset = gi_onset[np.searchsorted(gi_onset, t0, side='left')]
	near_gi_end = gi_end[np.searchsorted(gi_end, t0, side='left')-1]
	near_gs_end = gs_end[np.searchsorted(gs_end, t0, side='left')-1]
	
	### optional: discard times close to transitions.
	tcut = 40
	if t0+tcut-near_gs_onset>0:
		return 0

	if t0+tcut-near_gi_onset>0:
		return 0

	if -1000<t0-tcut-near_gi_end<0:
		return 0

	if -1000<t0-tcut-near_gs_end<0:
		return 0
		
	if near_gi_onset>near_gs_onset: # consider ONLY GS
		#print(t0, 'GS')
		return -1
	elif near_gi_onset<near_gs_onset: # consider ONLY GI
		return 1
	
	
def hypothesis_test(volcs, means, record, t, a, b, lead, period, plot='True'):

	### cut record to volcanic interval first:
	### -> DO this outside of function now.
	#cut_a = find_nearest(t, volcs[0]-100)
	#cut_b = find_nearest(t, volcs[-1]+100)
	### repr segment:
	#ta = 32039.
	#tb = 47503.
	#cut_a = find_nearest(t, ta-100)
	#cut_b = find_nearest(t, tb+100)
	#t = t[cut_a:cut_b]
	#record = record[cut_a:cut_b]
	
	def bootstrap_means(data, t, a, b, lead):
		N = 1000#20000
		K = len(data); M = 50#; L = 2
		means = []#np.empty(N)
		for i in range(N):
			idx = randrange(M,K-M-b)#randrange(a,K-M-b)
			### FILTER OUT GS OR GI samples.
			### -1 selects GS; +1 selects GI.
			#gi_test = filter_gi_gs(t[idx])
			if period=='GS':
				gi_test = filter_gi_gs(t[idx])
				if gi_test==-1:
					means.append(np.mean(data[idx-a:idx+b])-np.mean(data[idx+b:idx+M]))
			elif period=='GI':
				gi_test = filter_gi_gs(t[idx])
				if gi_test==1:
					means.append(np.mean(data[idx-a:idx+b])-np.mean(data[idx+b:idx+M]))
			elif period=='All':
				means.append(np.mean(data[idx-a:idx+b])-np.mean(data[idx+b:idx+M]))
		return np.asarray(means)
	
	means=np.asarray(means)
	means_boot = bootstrap_means(record, t, a, b, lead)
	
	if plot=='True':
		fig=pl.figure(figsize=(6,4.5))
		pl.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.98)
		xgrid = np.linspace(min(means_boot)-0.5*np.std(means_boot), max(means_boot)+0.5*np.std(means_boot),200)
		kde = KernelDensity(kernel='gaussian', bandwidth=np.std(means_boot)/12.).fit(means_boot[:, np.newaxis])
		log_dens = kde.score_samples(xgrid[:, np.newaxis])
		pl.fill(xgrid, np.exp(log_dens), fc='gray', alpha=0.5, label='Bootstrap')

		xgrid = np.linspace(min(means)-0.5*np.std(means), max(means)+0.5*np.std(means),200)
		kde = KernelDensity(kernel='gaussian', bandwidth=np.std(means)/4.).fit(means[:, np.newaxis])
		log_dens = kde.score_samples(xgrid[:, np.newaxis])
		pl.fill(xgrid, np.exp(log_dens), fc='royalblue', alpha=0.4, label='Volcanic events')
		
		pl.plot(100*[np.percentile(means_boot, 16.)], np.linspace(0,max(np.exp(log_dens)),100),'--',  color='black')
		
		pl.plot(100*[np.mean(means)], np.linspace(0,max(np.exp(log_dens)),100), color='tomato')

		print('------ Signal-to-Noise Ratio -------')
		print('Bootstrap 16-p.: ', round(np.percentile(means_boot, 16.),5))
		print('Average response: ', round(np.mean(means),5))
		print('SNR =', round(np.mean(means)/np.percentile(means_boot, 16.),5))

		pl.ylabel('PDF'); pl.xlabel('10-year isotopic anomaly'); pl.legend(loc='best')	
	
	return np.mean(means)/np.percentile(means_boot, 16.), np.percentile(means_boot, 16.)
	
	
	
def calc_response(volcs, t, record, N, M, a, b, k):
	resp_all = np.empty((N,k+M))
	anomalies = []
	rand_select = np.random.randint(N)
	baselines = []
	
	for i in range(N):
		idx = find_nearest(t, volcs[i])
		baselines.append(np.mean(record[idx+b:idx+M]))
		anomalies.append(np.mean(record[idx-a:idx+b]) - np.mean(record[idx+b:idx+M]))
		segment = detrend(record[idx-k:idx+M])
		#segment = record[idx-k:idx+M]
		segment = segment - np.mean(segment[k+b:])
		#segment = record[idx-M:idx+M] - np.mean(record[idx+b:idx+b+k])
		#print(t[idx], volcs[i])
		#t_segment = t[idx-M:idx+M]
		#print(t_segment[M], volcs[i], segment[M])
		
		resp_all[i,:] = segment
		#resp_all[i,:] = record[idx-M:idx+M] - np.mean(record[idx+15:idx+15+k])
		#if i == rand_select:
		#	fig=pl.figure()
		#	pl.subplot(121)
		#	pl.plot(range(-k,M), record[idx-M:idx+M])
		#	pl.subplot(122)
		#	pl.plot(range(-k,M), segment)
		
	return resp_all, anomalies, baselines


def lin_regr_uncertainty(x, y):
	# fit
	f = lambda x, *p: polyval(p, x)
	p, cov = curve_fit(f, x, y, [1, 1])
	# simulated draws from the probability density function of the regression
	#xi = linspace(np.min(x), np.max(x), 100)
	xi = linspace(0., np.max(x), 100)
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

def calc_cumu_density(data):
    	N = len(data)
    	grid = np.sort(data)
    	density = np.array(range(N))/float(N)
    	return [grid, density]

def mean_confidence_band(data):
        return [np.mean(data[:,j]) for j in range(len(data[0,:]))], [np.percentile(data[:,j], 14.) for j in range(len(data[0,:]))], [np.percentile(data[:,j], 86.) for j in range(len(data[0,:]))]

def bootstrap_mean_test(data, sample):
	N = 20000
	M = len(sample)
	mean_obs = np.mean(sample)
	
	means = np.empty(N)
	for i in range(N):
		boots = np.empty(M)
		for j in range(M):
			boot_idc = np.random.multinomial(1, len(data)*[1./len(data)])
			boot0 = data[np.where(boot_idc==1)[0][0]]
			boots[j] = boot0
		means[i] = np.mean(boots)
	
	statistics_sort = np.sort(means)
	p_index = np.argmin(np.abs(statistics_sort - mean_obs))
	pvalue_precise = 1-float(p_index)/(len(means))
	
	return 1- pvalue_precise

def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

if __name__ == '__main__':
        MAIN()

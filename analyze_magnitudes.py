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

from awkde import GaussianKDE
from ssvkernel import ssvkernel

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
	magnitude_gs_gi()
	#distributions()
	
	pl.show()
	
	
def distributions():

	volcs_sh, mag_sh = np.loadtxt('magnitude_age_sh_9k_publ.txt', unpack=True)
	volcs_nh, mag_nh = np.loadtxt('magnitude_age_nh_9k_publ.txt', unpack=True)
	
	### BIPOLAR eruptions
	volcs_bi_nh_ages = np.loadtxt('bipolar_volcanos_published.txt')
	idcs=np.argwhere(np.isin(volcs_bi_nh_ages,[12961., 33328., 43327., 59180.])) ### NGRIP
	## event at 43327 is much earlier in NH records. Too lazy to do manually...
	volcs_bi_nh_ages = np.delete(volcs_bi_nh_ages, idcs)
	volcs_bi_sh_ages = np.loadtxt('age_bipolar_jiamei_sh.txt')
	
	volcs_bi_sh, mag_bi_sh, volcs_uni_sh, mag_uni_sh = extract_bipolar(volcs_bi_sh_ages, volcs_sh, mag_sh)
	volcs_bi_nh, mag_bi_nh, volcs_uni_nh, mag_uni_nh = extract_bipolar(volcs_bi_nh_ages, volcs_nh, mag_nh)
	
	_, loading, _ , _, _, _= np.loadtxt('bipolar_loading_publ.txt', unpack=True)
	
	loading_do = [103.8, 168.0, 185.9, 90.2, 168.1, 114.8, 52.6] ### updated aerosol loading (published)
	mag_do_nh = [65.1, 263.9, 88.4, 37.6, 211.5, 68.5, 20.1] #NH
	mag_do_sh = [46.4, 17.7, 97.5, 68.8, 47.5, 46.3, 33.4] ## SH
	
	
	fig, axes = pl.subplots(1,3, figsize=(12,3.5))        	
	letter_subplots(axes, letters='a', yoffset=1.05) 
	pl.subplots_adjust(left=0.08, bottom=0.18, right=0.99, top=0.9, wspace=0.3, hspace=0.17)
	axes[0].set_title('Greenland deposition')
	plot_kde(axes[0], np.log(mag_bi_nh), 1/5., 'green', 'Bipolar')
	plot_kde(axes[0], np.log(mag_uni_nh), 1/8., 'black', 'Unipolar')
	#for i in range(len(mag_do_nh)):
	#	pl.plot(100*[np.log(mag_do_nh[i])], np.linspace(0., 0.4, 100), color='tomato')
	axes[0].legend(loc='best')
	axes[0].set_xlabel('Deposition magnitude')
	axes[0].set_ylabel('PDF')
	
	axes[1].set_title('Antarctic deposition')
	plot_kde(axes[1], np.log(mag_bi_sh), 1/6., 'green', 'Bipolar')
	plot_kde(axes[1], np.log(mag_uni_sh), 1/8., 'black', 'Unipolar')
	#for i in range(len(mag_do_sh)):
	#	pl.plot(100*[np.log(mag_do_sh[i])], np.linspace(0., 0.6, 100), color='tomato')
	axes[1].set_xlabel('Deposition magnitude')
	
	axes[2].set_title('Global climate forcing')
	plot_kde(axes[2], loading, 1/4., 'royalblue', 'Bipolar')
	#for i in range(len(loading_do)):
	#	pl.plot(100*[loading_do[i]], np.linspace(0., 0.005, 100), color='tomato')
	axes[2].set_xlabel('Aerosol loading')
		
	
	
def extract_bipolar(volcs_bi_ages, volcs, mag):

	k0=0
	idcs_rm = []; volcs_bi = []; mag_bi = []
	for i in range(len(volcs_bi_ages)):
		if min(abs(volcs_bi_ages[i]- volcs))<5:#5
			idx = np.argmin(abs(volcs_bi_ages[i]-volcs))
			k0+=1
			print(k0, volcs_bi_ages[i], volcs[idx], mag[idx])
			idcs_rm.append(idx)
			volcs_bi.append(volcs[idx])
			mag_bi.append(mag[idx])
			
			
	volcs_uni = list(volcs)
	mag_uni = list(mag)
	for index in sorted(idcs_rm, reverse=True):
		del volcs_uni[index]
		del mag_uni[index]
	mag_uni = np.asarray(mag_uni)
	mag_bi = np.asarray(mag_bi)
	print(len(volcs_uni), len(volcs))
	
	print('Blab', len(mag_uni), len(mag))

	return volcs_bi, mag_bi, volcs_uni, mag_uni
	
	
def plot_kde(ax, x, width, color, label): ### width in standard deviations of data.
	xgrid = np.linspace(min(x)-1.*np.std(x), max(x)+1.*np.std(x),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=width*np.std(x)).fit(np.asarray(x)[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	ax.fill(xgrid, np.exp(log_dens), fc=color, alpha=0.4, label=label)
	
	
def magnitude_gs_gi():
	#volcs, mag = np.loadtxt('magnitude_age_sh_9k_publ.txt', unpack=True)
	volcs, mag = np.loadtxt('magnitude_age_nh_9k_publ.txt', unpack=True)
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
	
	xgrid_uni_gs = xgrid; log_dens_uni_gs = log_dens
	
	xgrid = np.linspace(min(np.log(mag_gi))-1.*np.std(np.log(mag_gi)), max(np.log(mag_gi))+1.*np.std(np.log(mag_gi)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/8.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='black', alpha=0.4, label='GI')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	
	xgrid_uni_gi = xgrid; log_dens_uni_gi = log_dens
	
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
	bins = np.asarray([1., 3.5, 4., 5., 6.]) ### Greenland
	#bins = np.asarray([1., 3., 3.5, 4., 5.]) ### Antarctica
	
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
	
	gs_time = 28.4232 - 8 + 0.24 ### 2x 120y for GI 2.1 and 2.2
	gi_time = 19.8768 - 0.24
	
	fig=pl.figure(figsize=(5,3.5))
	ax=pl.subplot(111)
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	pl.bar(edges, np.asarray(gs_dens)/gs_time, width=0.4, color='orangered', alpha=0.5, align='edge', label='GS')
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
	ax.set_xticklabels(['<3.5', '3.5-4', '4-5', '5-6', '>6']) ### greenland
	#ax.set_xticklabels(['<3', '3-3.5', '3.5-4', '4-5', '>5']) ### antarctica
	pl.tick_params(bottom = False)
	#ax.get_xaxis().set_visible(False)
	#ax.axes.xaxis.set_ticks([])
	pl.minorticks_off()
	pl.legend(loc='best')
	
	#REPRESENTATIVE sample where there is equally much stadial and interstadial.
	'''
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
	'''
	
	### BIPOLAR eruptions
	volcs_bi = np.loadtxt('bipolar_volcanos_published.txt')
	idcs=np.argwhere(np.isin(volcs_bi,[12961., 33328., 43327., 59180.])) ### NGRIP
	### event at 43327 is much earlier in NH records. Too lazy to do manually...
	volcs_bi = np.delete(volcs_bi, idcs)
	
	####volcs_bi = np.loadtxt('bipolar_volcanos_edc.txt')
	####_, volcs_bi = np.loadtxt('bipolar_volcanos_edc_jiamei.txt', unpack=True)
	#volcs_bi = np.loadtxt('age_bipolar_jiamei_sh.txt')
	
	### NEED ALSO TO CHECK FOR ERUPTIONS WITH <10 deposition in EDC.
	### -> Just one eruption (Taupo)
	
	k0=0
	idcs_rm = []; volcs_bi_ngrip = []; mag_bi_ngrip = []
	for i in range(len(volcs_bi)):
		if min(abs(volcs_bi[i]- volcs))<5:#1
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
	
	xgrid_bi_gs = xgrid; log_dens_bi_gs = log_dens
	
	xgrid = np.linspace(min(np.log(mag_gi))-1.*np.std(np.log(mag_gi)), max(np.log(mag_gi))+1.*np.std(np.log(mag_gi)),200)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(np.log(mag_gi))/4.).fit(np.asarray(np.log(mag_gi))[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.fill(xgrid, np.exp(log_dens), fc='black', alpha=0.4, label='GI')
	pl.legend(loc='best')
	pl.xlabel('Magnitude'); pl.ylabel('PDF')
	
	xgrid_bi_gi = xgrid; log_dens_bi_gi = log_dens
	
	grid_cdf_gs, cdf_target_gs = calc_cumu_density(np.log(mag_gs))
	grid_cdf_gi, cdf_target_gi = calc_cumu_density(np.log(mag_gi))
	pl.subplot(122)
	pl.plot(grid_cdf_gs, cdf_target_gs, color='tomato')
	pl.plot(grid_cdf_gi, cdf_target_gi, color='black')
	pl.grid(axis='both')
	pl.xlabel('Magnitude'); pl.ylabel('CDF')
	
	### COMBINED figure uni and bipolar.
	
	fig=pl.figure(figsize=(5.8,3.5))
	pl.subplots_adjust(left=0.12, bottom=0.18, right=0.97, top=0.98, wspace=0.2, hspace=0.17)
	pl.fill(xgrid_uni_gs, np.exp(log_dens_uni_gs), fc='orangered', alpha=0.4, label='GS')
	pl.fill(xgrid_uni_gi, np.exp(log_dens_uni_gi), fc='black', alpha=0.4, label='GI')
	
	pl.fill(xgrid_bi_gs, np.exp(log_dens_bi_gs)+1., fc='orangered', alpha=0.4)
	pl.fill(xgrid_bi_gi, np.exp(log_dens_bi_gi)+1., fc='black', alpha=0.4)
	
	pl.xlim(1.9, 7.6); pl.yticks([])
	pl.xlabel('Magnitude'); pl.ylabel('PDF (a.u.)')
	
	### Global aerosol loading of bipolar eruptions.
	volcs, mag, _ , _, _, _ = np.loadtxt('bipolar_loading_publ.txt', unpack=True)
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

def calc_cumu_density(data):
    	N = len(data)
    	grid = np.sort(data)
    	density = np.array(range(N))/float(N)
    	return [grid, density]	
	
def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

if __name__ == '__main__':
        MAIN()

import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys
sys.path.insert(0, '/Users/amirbitran/Dropbox/CurrentStuff/Postdoc/pyHXExpress') #inserting this path at position 0 means it's at the top of sys.path and thus, in the following import commands, python will priortize this version of pyhxex over the one that pip installed
import pyhxexpress.hxex as hxex
import warnings
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm

#import pyhxexpress.config as config
#def hxex_reload():
#    importlib.reload(hxex)
#    importlib.reload(config)
#    hxex.config=config
#
#
#hxex_reload()


import pandas as pd
pd.set_option('display.max_columns',None) 
pd.set_option('display.max_colwidth', None)
import os
from Bio import SeqIO
from Bio.Seq import Seq


def is_zero_dimensional(variable): #thanks GPT
    if isinstance(variable, (int, float, complex, str, type(None))):
        return True
    elif isinstance(variable, np.ndarray):
        return variable.ndim == 0  # Check if numpy array is zero-dimensional
    elif np.isscalar(variable):  # For numpy scalars
        return True
    return False


def pep2str(peptide): #converts (4,18) into '0004-0018'
    str1 = str(peptide[0])
    while len(str1)<4:
        str1='0{}'.format(str1)
    str2 = str(peptide[1])
    while len(str2)<4:
        str2='0{}'.format(str2)
    return '{}-{}'.format(str1, str2)
    

def binom_isotope(bins, n,p, Current_Isotope):
    '''
    binomial function using the Natural Abundance isotopic envelope
    '''
    bs = hxex.binom(bins,n,p) #AB: simple binomial distribution evaluted between 0 and bins-1. We compute this because this models the deuteration process
    newbs=np.zeros(len(bs) + len(Current_Isotope)+1) #AB: this will be our convolution between the natural isotopic distribution and our binomial deuteration
    for i in range(len(bs)): #AB: loop over all values for our natural isotopic distribution (j) and all possible values for # of deuterons governed by binomial (i) to form the convolution
        for j in range(len(Current_Isotope)):     
            newbs[i+j] += bs[i]*Current_Isotope[j]   #AB: the probabitly that the natural isotopic distr has value j and that the deuterated distribution has value i is simply their product
    return newbs[0:bins+1]

def n_binom_isotope_AB( bins, Current_Isotope, *params ): #allfracsversion
    '''
    n population binomial using the Natural Abundance isotopic envelope
    AB: this is a probabiltiy distribution that results from the convolution of the natural isotopic distribution with some number of binomials corresponding to different deuterated populations

    This function is a modificaiton by AB such that Current_Isotope is not treated as a global anymore
    '''
    # params takes the form [ scaler, mu_1, ..., mu_n, frac_1, ..., frac_n] 
    n_curves = int(( len(params) + 1) / 3.0 )
    log_scaler = params[0]
    n_array = np.array( params[1:n_curves+1] )
    mu_array = np.array( params[n_curves+1:2*n_curves+1] )
    frac_array = np.array( params[ -n_curves: ] )
    frac_array = frac_array/np.sum(frac_array)
    poissons = [ frac * binom_isotope( bins, n, mu, Current_Isotope ) for frac, n, mu in zip( frac_array, n_array, mu_array ) ]
    truncated = np.power( 10.0, log_scaler ) * np.sum( poissons, axis=0, )[0:bins+1]
    return truncated 

def Convert_to_pepstring(start, end): #oh lol I wrote two redundant functiosn that do this same thing
    start = str(start)
    while len(start)<4:
        start = '0{}'.format(start)
    
    end = str(end)
    while len(end)<4:
        end = '0{}'.format(end)
    return '{}-{}'.format(start, end)
    
def Plot_spectrum(deutdata, fitparams, sample, peptide, charge, deut_time, reps, deut_color='blue', undeut_color='gray', plot_undeut=True, norm='Sum', labelx=True, labely=True, ax=None, rawdata = [], Spurious_peak_thresh=5, LimitMZRange=True, best_n_curves = np.nan, fontsize=20):
    #By default plots only the intensity at the expected m/z peaks and the fit
    #you can also have it plot the raw data if you pass it a rawdata dataframe
    #TODO: Have it incorporate the limited m/z range
    if ax is None:
        fig, ax = plt.subplots()

    if type(reps)==int:
        reps = [reps]

    deut = hxex.filter_df(deutdata, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time for i in range(len(reps))], rep=reps) #deut data specifically evaluated at m/z values correspondign to the peptide
    undeut = hxex.filter_df(deutdata, sample,  peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=0, rep=1) #deut data specifically evaluated at m/z values correspondign to the peptide
    
       
    ydeut=np.array(deut.Intensity.copy())
    ydeut = hxex.Filter_spurious_peaks(ydeut, thresh=Spurious_peak_thresh)

    mz = deut.mz.copy()
    mz = mz.to_numpy()
    if LimitMZRange and len(deut['peptide'])>0:
        #print(deut['peptide'].iloc[0])
        mz, ydeut = hxex.Limit_MZ(mz, ydeut, deut['peptide'].iloc[0], charge)
    scale_ydeut = np.nansum(ydeut)
    xdeut= charge*np.array(mz) - charge
    if norm=='Sum':
        ydeutnorm = scale_ydeut
    elif norm=='Max':
        ydeutnorm = np.nanmax(ydeut)
    else:
        ydeutnorm = 1
    
    if type(rawdata)==pd.core.frame.DataFrame: #then we plot the raw spectra
        rawdeut = hxex.filter_df(rawdata, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time for i in range(len(reps))], rep=reps)
        rawint = rawdeut.Intensity
        ax.plot(charge*rawdeut.mz - charge, rawint, alpha=0.5, color='k')


    
    
    ax.vlines(xdeut, [0 for ii in range(len(xdeut))],ydeut/ydeutnorm, color=deut_color) #this ydeutnorm is a bit too small
    n_binsdeut = len(ydeut)-1

    yundeut = np.array(undeut.Intensity.copy())
    yundeut = hxex.Filter_spurious_peaks(yundeut, thresh=Spurious_peak_thresh)
    xundeut= charge*np.array(undeut.mz.copy()) - charge
    if norm=='Sum':
        yundeutnorm = scale_ydeut
    elif norm=='Max':
        yundeutnorm = np.nanmax(yundeut)
    else:
        yundeutnorm = 1
    if plot_undeut:
        ax.vlines(xundeut, [0 for ii in range(len(xundeut))],yundeut/yundeutnorm, color=undeut_color, linestyle=':')

    #get the fit function for deuterated sample and plot it
    deutfit = hxex.filter_df(fitparams, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time for i in range(len(reps))], rep=reps)
    
    if not deutfit.empty:
        Current_Isotope = hxex.get_na_isotope(deut['peptide'].values[0],charge,npeaks=None,mod_dict={}) #theoretical isotopic distribution for undeuterated
        fits = deutfit.copy() 
        nboot_list = list(fits['nboot'].unique())
        if np.isnan(best_n_curves): #you haven't inputted best_n_curves, so use the value selected by the origianl hxex algorithm (which may vary wildly from replicate to replicate and lenght to length)
            if min(nboot_list) > 0:
                best_n_curves = fits[fits['nboot']==1]['ncurves'].values[0] #from bootstrap #1, we can extract how many curves were used
            else: best_n_curves = fits['ncurves'].values[-1]
        #for nb in nboot_list:
        for nb in [0]: #rather than looping over all bootstrap iterations, just use the zeroeth 
            params_best_fit = fits.copy()[(fits['nboot']==nb) & (fits['ncurves']==best_n_curves)]['Fit_Params'].values[0]
            params_best_fit = [float(x) for x in params_best_fit.split()]
            scaler,nexs,mus,fracs = hxex.get_params(*params_best_fit,sort=True,norm=True,unpack=True) #AB: If we removed spurious peaks, the scaler is gonna be off...
            fracsum = np.sum(fracs)
            #print('nbins is {}'.format(n_binsdeut)) #AB
            fit_y = n_binom_isotope_AB( n_binsdeut, Current_Isotope, *params_best_fit ) * scale_ydeut  #The n_binom_isotope function comes out pre-normalized to 1, but then we scale it up based on the intensity of the curve
            ax.plot(xdeut, fit_y/ydeutnorm, color=deut_color)  #if norm = 'Sum', then ydeutnorm is 1, and so we undo the multiplication by scale_ydeut in previosu line, yeidling a normalized curve
            #print('First 10 values of fit_y are {}'.format(fit_y[0:10])) #AB
            #print('Current isotope is: {}'.format(Current_Isotope))
            #plot the different modes individually as a dashed line

            linestyles=[':','--', '-.']
            for k in range( best_n_curves ):
                nex = nexs[k]
                mu = mus[k]                
                frac = fracs[k]/fracsum                                    
                fit_yk = np.power( 10.0, scaler )  * frac * binom_isotope( n_binsdeut, nex, mu,Current_Isotope) * scale_ydeut
                ax.plot( xdeut, fit_yk/ydeutnorm, linestyle=linestyles[k], color=deut_color)
                

    if labelx:
        ax.set_xlabel('mass (Da)', fontsize=fontsize)
    if labely:
        if norm=='Sum':
            ax.set_ylabel('Norm. Intensity', fontsize=fontsize)
        elif norm=='Max':
            ax.set_ylabel('Intensity/Max Int.', fontsize=fontsize)
        else:
            ax.set_ylabel('Intensity', fontsize=fontsize)
    
    # Get current x limits
    x_min, x_max = ax.get_xlim()
    x_min = np.round(x_min)
    x_max = np.round(x_max)
    #x_min = 5*np.floor(x_min/5) #convert so that it's always multiples of 5 on the axis
    #x_max = 5*np.ceil(x_max/5)

    # Set x ticks every 5 units
    ax.set_xticks(np.arange(x_min, x_max + 1, 5))
    ax.tick_params(labelsize=fontsize)

    return deut


def Spec_vs_len(deutdata, fitparams,samples, lengths, peptide, charge, deut_time, reps, figsize = (16,6), xlim= None, norm='Max', plot_undeut=True, rawdata=[], savepath = None, LimitMZRange=True, smartn_curves = False, pvalue_thresh=np.nan, titles=[], fontsize=12, ax = None, return_ncurves=False ):
    #plots spectrum vs length
    #You can decide to let it choose # of curves at each length if you set smartn_curves to True
    #if you set return_ncurves = True, then it gives you back the values for smartn_curves that it chose at each lenght

    if len(titles)==0:
        titles = ['{} AA'.format(length) for length in lengths]
    if smartn_curves:
        if np.isnan(pvalue_thresh):
            raise RuntimeError('Must input a p-value threshold if using smartn_curves feature. \n Recommended to use config.Ncurve_p_accept to ensure consistency w fit')
        ncurves = Choose_ncurves(fitparams,samples, lengths, peptide, charge, deut_time, pvalue_thresh, verboise=False)
    else:
        ncurves = np.nan*np.ones(len(samples))

    if ax==None:
        fig, ax = plt.subplots(2, len(lengths))
        if len(lengths)==1:
            ax = np.expand_dims(ax, axis=1)
    #display(reps)
    for rep in reps:
        plt.subplots_adjust(left=0.03, right=0.97, wspace=0.5, hspace=0.5, bottom=0.03, top=0.97)
        fig.set_size_inches(*figsize)
        for i, (sample, length) in enumerate(zip(samples, lengths)):
            if i==0:
                labelx, labely = (True, True)
            else:
                labelx, labely = (False, False)
        
            Plot_spectrum(deutdata, fitparams,sample, peptide, charge, deut_time, rep, norm=norm,ax = ax[rep-1,i],labelx=labelx, labely=labely, plot_undeut=plot_undeut, rawdata=rawdata,LimitMZRange=LimitMZRange, best_n_curves = ncurves[i], fontsize=fontsize)
            
            ax[rep-1,i].set_title(titles[i], fontsize=20)
            if xlim is not None:
                ax[rep-1,i].set_xlim(xlim)
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print('Plots of all spectra vs length saved to {}'.format(savepath))
    
    if return_ncurves:
        return ncurves
        
def ReadRawData(metadf_run, sample, peptide, charge, deut_time, rep ):
    #A simple function to get raw data from SpecExport csv file, haven't extensivly tested
    row = hxex.filter_df(metadf_run, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge)
    spec_path = os.path.join(config.Data_DIR,row['sample'].values[0],row['file'].values[0])
    csv_files = [ os.path.join(spec_path, f) for f in os.listdir(spec_path) if f[-5:]==str(int(charge))+'.csv'  ]
    
    csvtimes = [file.split('/')[-1] for file in csv_files]
    csvtimes = [file.split('s-')[0] for file in csvtimes]
    display(csv_files)

    for t, time in enumerate(csvtimes):
        if time[0:5]=='Non-D':
            #print('moo')
            csvtimes[t]='0'
    reps = [file.split('-')[-2] for file in csv_files]

    #display(csvtimes)
    csv_file = [file for t, file, r in zip(csvtimes, csv_files, reps) if t==str(deut_time) and r==str(rep) ][0]
    #display(csv_file)

    rawdf = pd.read_csv(csv_file, names=['m/z', 'Intensity'])
    mz = rawdf['m/z'].to_numpy()
    Int = rawdf['Intensity'].to_numpy()
    return mz, Int

def Ndeut_vs_len(fitparams,samples, lengths, peptide, charge, deut_time, reps,markerscale= 600, plot_thresh = 0.0, savepath = None, smartn_curves = False, pvalue_thresh=np.nan):
    #only plot if fraction of population is more than plot_thresh

    colors = ['r', 'b', 'g'] #for diff populations
    markers = ['*', 'o', 'v']

    fig, ax = plt.subplots()
    
    if smartn_curves:
        if np.isnan(pvalue_thresh):
            raise RuntimeError('Must input a p-value threshold if using smartn_curves feature. \n Recommended to use config.Ncurve_p_accept to ensure consistency w fit')
        ncurves = Choose_ncurves(fitparams,samples, lengths, peptide, charge, deut_time, pvalue_thresh, verboise=False)
    else:
        ncurves =np.nan*np.ones(len(samples))

    for r, rep in enumerate(reps):
        markersizes = []
        means = []
        ls = []
        colorlist = []

        for i, (sample, length) in enumerate(zip(samples, lengths)):
            fits = hxex.filter_df(fitparams, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time for i in range(len(reps))], rep=rep)
            nboot_list = list(fits['nboot'].unique())
            if len(nboot_list)>0: #a good proxy for the fact that this sample's parameters even exist
                if np.isnan(ncurves[i]):  #you haven't selected smart n curves, so use the value selected by the origianl hxex algorithm (which may vary wildly from replicate to replicate and lenght to length)
                    if min(nboot_list) > 0:
                        best_n_curves = fits[fits['nboot']==1]['ncurves'].values[0] #from bootstrap #1, we can extract how many curves were used
                    else: best_n_curves = fits['ncurves'].values[-1]
                else:
                    best_n_curves = ncurves[i]
                    
                params_best_fit = fits[(fits['nboot']==0) & (fits['ncurves']==best_n_curves)]['Fit_Params'].values[0]
                params_best_fit = [float(x) for x in params_best_fit.split()]
                scaler,nexs,mus,fracs = hxex.get_params(*params_best_fit,sort=True,norm=True,unpack=True) #AB: If we removed spurious peaks, the scaler is gonna be off...
                
                i=0
                for nex, mu, frac in zip(nexs, mus, fracs):
                    if frac>plot_thresh:
                        means.append(nex*mu) #mean # of deuterons is nex*mu, since nex is like N in binomial distribution while mu is like p
                        markersizes.append(frac)
                        ls.append(length)
                        colorlist.append(colors[i])
                        i+=1

        markersizes = np.array(markersizes)
        means = np.array(means)
        ls = np.array(ls)

        ax.scatter(ls, means,s=markerscale*markersizes, c=colorlist, marker=markers[r])
    ax.set_xlabel('Length (AA)', fontsize=20)
    ax.set_ylabel('# deuterons', fontsize=20)
    ax.tick_params(labelsize=20)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print('Plots of parameters vs length saved to {}'.format(savepath))



# def Ndeut_vs_len_3D(fitparams, samples, lengths, peptide, charge, deut_time, reps, savepath=None, smartn_curves=False, pvalue_thresh=np.nan, colors=[], plot_avg=False, ignore_mode=np.nan, elev=40, azim=-60, bar_width=1, bar_depth=0.05):
#     """
#     Makes bar plots that are stacked one behind the other for different lengths
#     if plot_avg, then this averages over technical replicates, but also lightly plots the individual replicates as nearly transparent bars

#     You can tell it to ignore one of the modes (if it's spurious), IN ONE INDEXING
#     If so, it renormalizes the others to add up to one
#     By default doesn't do this

#     """
#     fig = plt.figure()
#     fig.set_size_inches((10, 7.5))
#     fig.patch.set_facecolor('white')  # Set figure background color to white
#     ax = fig.add_subplot(projection='3d')
#     ax.xaxis.pane.set_facecolor('white')
#     ax.yaxis.pane.set_facecolor('white')
#     ax.zaxis.pane.set_facecolor('white')
#     # set the view angle
#     ax.view_init(elev=elev, azim=azim)  # Adjust these values to change the view angle. By default matplotlib uses 30 and -60--in my function the defaults are 40 and -60
#     if len(colors) == 0:
#         colormap = plt.cm.jet
#         values = np.linspace(0, 1, len(samples))
#         colors = [colormap(value) for value in values]
#     if smartn_curves:
#         if np.isnan(pvalue_thresh):
#             raise RuntimeError('Must input a p-value threshold if using smartn_curves feature. \n Recommended to use config.Ncurve_p_accept to ensure consistency w fit')
#         ncurves = Choose_ncurves(fitparams, samples, lengths, peptide, charge, deut_time, pvalue_thresh, verboise=False)
#     else:
#         ncurves = np.nan * np.ones(len(samples))
#     linestyles = [':', '--', '-.']
#     allmeans = np.nan * np.ones((len(reps), len(samples), 3))  # stores the mean # of deuterons, replicates x lengths x nmodes. Allows for up to 3 nmodes, but doesn't have to use all of them
#     allfracs = np.nan * np.ones((len(reps), len(samples), 3))  # same as above for the fracs parameter
    
#     bar_data = []

#     for i, (sample, length) in enumerate(zip(samples, lengths)):
#         for r, rep in enumerate(reps):
#             fits = hxex.filter_df(fitparams, sample, range=peptide, charge=charge, timept=[deut_time for i in range(len(reps))], rep=rep)
#             nboot_list = list(fits['nboot'].unique())
#             if len(nboot_list) > 0:  # a good proxy for the fact that this sample's parameters even exist
#                 if np.isnan(ncurves[i]):  # you haven't selected smart n curves, so use the value selected by the original hxex algorithm (which may vary wildly from replicate to replicate and length to length)
#                     if min(nboot_list) > 0:
#                         best_n_curves = fits[fits['nboot'] == 1]['ncurves'].values[0]  # from bootstrap #1, we can extract how many curves were used
#                     else:
#                         best_n_curves = fits['ncurves'].values[-1]
#                 else:
#                     best_n_curves = ncurves[i]
#                 params_best_fit = fits[(fits['nboot'] == 0) & (fits['ncurves'] == best_n_curves)]['Fit_Params'].values[0]
#                 params_best_fit = [float(x) for x in params_best_fit.split()]
#                 scaler, nexs, mus, fracs = hxex.get_params(*params_best_fit, sort=True, norm=True, unpack=True)  # AB: If we removed spurious peaks, the scaler is gonna be off...
#                 means = np.multiply(nexs, mus)  # for each mode, mean # of deuterons
#                 if ~np.isnan(ignore_mode) and ignore_mode <= len(means):
#                     means[ignore_mode - 1] = np.nan
#                     fracs[ignore_mode - 1] = np.nan
#                     fracs = fracs / np.nansum(fracs)  # renormalize fractions
                
#                 if plot_avg:
#                     alpha = 0.2
#                 else:
#                     alpha = 0.8
                
#                 nni = ~np.isnan(means)
#                 for mean, frac in zip(means[nni], fracs[nni]):
#                     bar_data.append((i, mean, frac, colors[i], linestyles[r], alpha))
                
#                 allmeans[r, i, 0:len(means)] = means
#                 allfracs[r, i, 0:len(fracs)] = fracs
    
#     # Sort bars by the z-index to plot them in the correct order
#     bar_data.sort(key=lambda x: x[0])
    
#     for i, mean, frac, color, linestyle, alpha in bar_data:
#         ax.bar(mean, frac, zs=i, zdir='y', edgecolor=color, facecolor=color, linestyle=linestyle, width=bar_width, alpha=alpha)
    
#     if plot_avg:  # plots the avg over technical reps
#         avg_bar_data = []
#         for i, (sample, length) in enumerate(zip(samples, lengths)):
#             x = np.nanmean(allmeans[:, i, :], axis=0)
#             y = np.nanmean(allfracs[:, i, :], axis=0)
#             nni = ~np.isnan(x)
#             for mean, frac in zip(x[nni], y[nni]):
#                 avg_bar_data.append((i, mean, frac, colors[i]))
        
#         # Sort bars by the z-index to plot them in the correct order
#         avg_bar_data.sort(key=lambda x: x[0])
        
#         for i, mean, frac, color in avg_bar_data:
#             ax.bar(mean, frac, zs=i, zdir='y', alpha=0.9, facecolor=color, width=bar_width, zorder=10 * (len(samples) - i))
    
#     ax.set_ylim((-0.5, len(samples)))
#     ax.set_yticks([i for i in range(len(samples))])
#     ax.set_yticklabels(['{} AA'.format(l) for l in lengths], ha='left',)
#     ax.set_xlabel('Mean # deut.', labelpad=0.01)
#     ax.set_zlabel('Pop. frac.', labelpad=0.01)
#     for label in ax.get_yticklabels():
#         label.set_verticalalignment('center')
#     if savepath is not None:
#         plt.savefig(savepath, dpi=300, bbox_inches='tight')
#         plt.close()


def Ndeut_vs_len_3D(fitparams, samples, lengths, peptide, charge, deut_times, reps, savepath=None, smartn_curves=False, pvalue_thresh=np.nan, colors=[], plot_avg = True, plot_replicates=True, ignore_mode = np.nan, elev=40, azim=-60, bar_width=1, bar_depth=0.05, yticklabels=[], patterns = [], ax=None):
    """
    Makes bar plots that are stacked one behind the other for different lengths
    if plot_avg, then this averages over technical replicates, but also lgihtly plots the individual replicates as nearly transparent bars

    You can tell it to ignore one of the modes (if it's spurious), IN ONE INDEXING
    If so, it renormalizes the others to add up to one
    By default doesn't do this
    New on Dec 2 2024: This ignore_mode can either be a single value (in which case it's applied to all samples), or a list
    of the same lenght as samples
    (in case you want different modes to be ignored for diff samples, e.g. if one sample has a spuriously fitted mode)
    Any or all in that list can be nan

    New on November 18 2024: the deut_times paramter can either be a single deuteration time that's used for all samples,
    or a list of deuteration times, one for each sample
    If the latter, it gives you a way to also plot mode parameters vs deuteration time for a given condition--in that case just make sure that every entry of "samples" is the same condition

    Note that this isn't compatible with Choose_n_curves at the moment--that funciton still expects a single deuteration time, and by default will use the first entry if you provdie a list
    """

    if np.shape(deut_times)==():
        deut_times = [deut_times for s in range(len(samples))]
    if len(yticklabels)==0:
        #yticklabels = ['{} AA'.format(l) for l in lengths]
        yticklabels = ['{}'.format(l) for l in lengths]
    
    if is_zero_dimensional(ignore_mode):
        ignore_mode = [ignore_mode for l in range(len(samples))]
    
    if ax==None:
        fig = plt.figure()
        fig.set_size_inches((10, 7.5))
        fig.patch.set_facecolor('white')  # Set figure background color to white
        ax = fig.add_subplot(projection='3d')
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    #set the view angle
    ax.view_init(elev=elev, azim=azim)  # Adjust these values to change the view angle. By default matplotlib uses 30 and -60--in my fucntion the defaults are 40 and -60
        
        
    if len(colors) == 0:
        colormap = plt.cm.jet
        values = np.linspace(0, 1, len(samples))
        colors = [colormap(value) for value in values]
    if smartn_curves:
        if np.isnan(pvalue_thresh):
            raise RuntimeError('Must input a p-value threshold if using smartn_curves feature. \n Recommended to use config.Ncurve_p_accept to ensure consistency w fit')
        ncurves = Choose_ncurves(fitparams, samples, lengths, peptide, charge, deut_times[0], pvalue_thresh, verboise=False)
    else:
        ncurves = np.nan * np.ones(len(samples))
    linestyles = [':', '--', '-.']
    allmeans = np.nan*np.ones((len(reps), len(samples), 3)) #stores the mean # of deuterons,  replicates x lengths x nmodes. Allows for up to 3 nmodes, but doesn't have to use all of them
    allfracs = np.nan*np.ones((len(reps), len(samples), 3)) #same as above for the fracs parameter
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        deut_time = deut_times[i]
        for r, rep in enumerate(reps):
            fits = hxex.filter_df(fitparams, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time for i in range(len(reps))], rep=rep)
            nboot_list = list(fits['nboot'].unique())
            if len(nboot_list) > 0:  # a good proxy for the fact that this sample's parameters even exist
                if np.isnan(ncurves[i]):  # you haven't selected smart n curves, so use the value selected by the origianl hxex algorithm (which may vary wildly from replicate to replicate and lenght to length)
                    if min(nboot_list) > 0:
                        best_n_curves = fits[fits['nboot'] == 1]['ncurves'].values[0]  # from bootstrap #1, we can extract how many curves were used
                    else:
                        best_n_curves = fits['ncurves'].values[-1]
                else:
                    best_n_curves = ncurves[i]
                params_best_fit = fits[(fits['nboot'] == 0) & (fits['ncurves'] == best_n_curves)]['Fit_Params'].values[0]
                params_best_fit = [float(x) for x in params_best_fit.split()]
                scaler, nexs, mus, fracs = hxex.get_params(*params_best_fit, sort=True, norm=True, unpack=True)  # AB: If we removed spurious peaks, the scaler is gonna be off...
                means = np.multiply(nexs, mus) #for each mode, mean # of deuterons
                if ~np.isnan(ignore_mode[i]) and ignore_mode[i] <=len(means):
                    means[ignore_mode[i]-1] = np.nan
                    fracs[ignore_mode[i]-1] = np.nan
                    fracs = fracs/np.nansum(fracs) #renormalize fractions
                #print(means)
                #ax.bar(means, fracs, zs=i, zdir='y', edgecolor=colors[i], alpha=1.0, facecolor='none', linestyle=linestyles[r])
                if plot_replicates:
                    if plot_avg:
                        alpha=0.2
                    else:
                        alpha=0.8
                    #ax.bar(means, fracs, zs=i, zdir='y', edgecolor=colors[i], alpha=alpha, facecolor=colors[i], linestyle=linestyles[r])    
                    nni = ~np.isnan(means)
                    #ax.bar3d(means[nni], [i for count in range(nni.sum())], [0 for count in range(nni.sum())], dx = [bar_width for count in range(nni.sum())], dy = [bar_depth for count in range(nni.sum())], dz = fracs[nni],  alpha=0.9, color=colors[i], shade=False)
                    ax.bar(means[nni], fracs[nni], zs=i, zdir='y', edgecolor=colors[i],  facecolor='none', linestyle=linestyles[r], width=bar_width)         
                    #print('Here is the color from data plot: {}'.format(colors[i]))
                allmeans[r,i,0:len(means)] = means
                allfracs[r,i,0:len(fracs)] = fracs
    if plot_avg: #plots the avg over technical reps
        for i, (sample, length, color) in enumerate(zip(np.flip(samples), np.flip(lengths), np.flip(colors, axis=0))):
            #print('Here is the color from means plot: {}'.format(color))
            #print(allmeans[:,i,:])
            z=len(samples)-i-1
            x = np.nanmean(allmeans[:,z,:], axis=0)
            y = np.nanmean(allfracs[:,z,:], axis=0)
            nni = ~np.isnan(x)

            #if sample=='l35-Released':
            #    alpha=0.2
            #elif sample=='Halotag297':
            #    alpha=1
            #else:
            #    alpha=0.9
            #alpha = 1 - (i+1)/len(samples)
    

            if plot_replicates:
                edgecolor=None
                alpha=0.8
            else:
                #edgecolor=color
                edgecolor='k'
                alpha=0.8
            #print(y[nni])
            if len(patterns)>0:
                ax.bar(x[nni], y[nni], zs=z, zdir='y',  alpha=alpha, facecolor=color, edgecolor = edgecolor,width=bar_width, hatch = patterns[len(samples)-i-1],linestyle='-')
                #ax.bar(x[nni], y[nni], zs=z, zdir='y',  facecolor=color, width=bar_width, hatch = patterns[len(samples)-i-1])
            else:
                ax.bar(x[nni], y[nni], zs=z, zdir='y',  alpha=alpha, facecolor=color, width=bar_width, edgecolor = edgecolor, linestyle='-')
                #ax.bar(x[nni], y[nni], zs=z, zdir='y',  facecolor=color, width=bar_width)

            #print([i for count in range(nni.sum())])
            #ax.bar3d(x[nni]-bar_width/2, [i for count in range(nni.sum())], [0 for count in range(nni.sum())], dx = [bar_width for count in range(nni.sum())], dy = [bar_depth for count in range(nni.sum())], dz = y[nni],  alpha=0.9, color=colors[i], shade=True)
    ax.set_ylim((-0.5, len(samples)))
    ax.set_yticks([i for i in range(len(samples))])
    ax.set_yticklabels(yticklabels, ha='left',)
    ax.set_ylabel('Length (AA)', labelpad=10)
    ax.set_xlabel('Mean # deut.', labelpad=0.01)
    ax.set_zlabel('Pop. frac.', labelpad=0.01)
    for label in ax.get_yticklabels():
        label.set_verticalalignment('center')
    #fig.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)  # Adjust the margins
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()


def Ndeut_vs_len_3DGauss(fitparams, samples, lengths, peptide, charge, deut_times, reps, savepath=None, smartn_curves=False, pvalue_thresh=np.nan, colors=[], plot_avg = True, plot_replicates=True, ignore_mode = np.nan, elev=60, azim=-90, bar_width=1, bar_depth=0.05, yticklabels=[], patterns = [], ax=None, sd=0.2):
    """
    For each of multiple conditions c (e.g. translation times or lengths), 
    plots in 3D the deuteration d of each mode as a Gaussian centered at x=c, y=d with height f,
    corresponding to the population fraction of that mode
    The standard dev for each of these Gaussians is sd


    if plot_avg, then this averages over technical replicates, but also lgihtly plots the individual replicates as nearly transparent bars

    You can tell it to ignore one of the modes (if it's spurious), IN ONE INDEXING
    If so, it renormalizes the others to add up to one
    By default doesn't do this
    New on Dec 2 2024: This ignore_mode can either be a single value (in which case it's applied to all samples), or a list
    of the same lenght as samples
    (in case you want different modes to be ignored for diff samples, e.g. if one sample has a spuriously fitted mode)
    Any or all in that list can be nan

    New on November 18 2024: the deut_times paramter can either be a single deuteration time that's used for all samples,
    or a list of deuteration times, one for each sample
    If the latter, it gives you a way to also plot mode parameters vs deuteration time for a given condition--in that case just make sure that every entry of "samples" is the same condition

    Note that this isn't compatible with Choose_n_curves at the moment--that funciton still expects a single deuteration time, and by default will use the first entry if you provdie a list
    """

    if np.shape(deut_times)==():
        deut_times = [deut_times for s in range(len(samples))]
    if len(yticklabels)==0:
        #yticklabels = ['{} AA'.format(l) for l in lengths]
        yticklabels = ['{}'.format(l) for l in lengths]
    
    if is_zero_dimensional(ignore_mode):
        ignore_mode = [ignore_mode for l in range(len(samples))]
    
    if ax==None:
        fig = plt.figure()
        fig.set_size_inches((10, 7.5))
        fig.patch.set_facecolor('white')  # Set figure background color to white
        ax = fig.add_subplot(projection='3d')
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    #set the view angle
    ax.view_init(elev=elev, azim=azim)  # Adjust these values to change the view angle. By default matplotlib uses 30 and -60--in my fucntion the defaults are 40 and -60
        
    xrange = np.arange(-0.5, 0.8*(peptide[1] - peptide[0]), 0.01)
    if len(colors) == 0:
        colormap = plt.cm.jet
        values = np.linspace(0, 1, len(samples))
        colors = [colormap(value) for value in values]
    if smartn_curves:
        if np.isnan(pvalue_thresh):
            raise RuntimeError('Must input a p-value threshold if using smartn_curves feature. \n Recommended to use config.Ncurve_p_accept to ensure consistency w fit')
        ncurves = Choose_ncurves(fitparams, samples, lengths, peptide, charge, deut_times[0], pvalue_thresh, verboise=False)
    else:
        ncurves = np.nan * np.ones(len(samples))
    linestyles = [':', '--', '-.']
    allmeans = np.nan*np.ones((len(reps), len(samples), 3)) #stores the mean # of deuterons,  replicates x lengths x nmodes. Allows for up to 3 nmodes, but doesn't have to use all of them
    allfracs = np.nan*np.ones((len(reps), len(samples), 3)) #same as above for the fracs parameter
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        deut_time = deut_times[i]
        for r, rep in enumerate(reps):
            fits = hxex.filter_df(fitparams, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time for i in range(len(reps))], rep=rep)
            nboot_list = list(fits['nboot'].unique())
            if len(nboot_list) > 0:  # a good proxy for the fact that this sample's parameters even exist
                if np.isnan(ncurves[i]):  # you haven't selected smart n curves, so use the value selected by the origianl hxex algorithm (which may vary wildly from replicate to replicate and lenght to length)
                    if min(nboot_list) > 0:
                        best_n_curves = fits[fits['nboot'] == 1]['ncurves'].values[0]  # from bootstrap #1, we can extract how many curves were used
                    else:
                        best_n_curves = fits['ncurves'].values[-1]
                else:
                    best_n_curves = ncurves[i]
                params_best_fit = fits[(fits['nboot'] == 0) & (fits['ncurves'] == best_n_curves)]['Fit_Params'].values[0]
                params_best_fit = [float(x) for x in params_best_fit.split()]
                scaler, nexs, mus, fracs = hxex.get_params(*params_best_fit, sort=True, norm=True, unpack=True)  # AB: If we removed spurious peaks, the scaler is gonna be off...
                means = np.multiply(nexs, mus) #for each mode, mean # of deuterons
                if ~np.isnan(ignore_mode[i]) and ignore_mode[i] <=len(means):
                    means[ignore_mode[i]-1] = np.nan
                    fracs[ignore_mode[i]-1] = np.nan
                    fracs = fracs/np.nansum(fracs) #renormalize fractions
                if plot_replicates:
                    if plot_avg:
                        alpha=0.2
                    else:
                        alpha=0.8
                    nni = ~np.isnan(means)
                    ax.plot(xrange, MultiGaussian(xrange, means[nni], fracs[nni], sd=sd),zs=i, zdir = 'y', color=colors[i], alpha=alpha)        
                    #print('Here is the color from data plot: {}'.format(colors[i]))
                allmeans[r,i,0:len(means)] = means
                allfracs[r,i,0:len(fracs)] = fracs
    if plot_avg: #plots the avg over technical reps
        for i, (sample, length, color) in enumerate(zip(np.flip(samples), np.flip(lengths), np.flip(colors, axis=0))):
            #print('Here is the color from means plot: {}'.format(color))
            #print(allmeans[:,i,:])
            z=len(samples)-i-1
            x = np.nanmean(allmeans[:,z,:], axis=0)
            y = np.nanmean(allfracs[:,z,:], axis=0)
            nni = ~np.isnan(x)

            if plot_replicates:
                alpha=0.8
            else:
                #edgecolor=color
                alpha=1
            ax.plot(xrange, MultiGaussian(xrange, x[nni], y[nni], sd=sd), zs=z, zdir = 'y', color=colors[i], alpha=alpha)

    #deal with y axis cosmetics
    ax.set_ylim((-0.5, len(samples)))
    ax.set_yticks([i for i in range(len(samples))])
    ax.set_yticklabels(yticklabels, ha='left', verticalalignment='bottom')
    
    #setting the y axis padding is a bit tricky sicne I want it to align wtih the z axis, whose position I"ve set in data units (1 to the right of end of plot) rather than poitns
    #so we convert 1 data unit to points so that we can use the labelpad option which takes points as the unit
    #ALTHOUGH THIS DOESN'T WORK ANYWAYS SO COMMENTING IT OUT
    # Desired label padding in data units
    # desired_padding_data_units = 1.5
    # # Convert data units to points for the y-axis
    # y_data_to_display = ax.transData.transform((0, desired_padding_data_units))  # Transform data to display
    # y_origin_to_display = ax.transData.transform((0, 0))  # Transform origin to display
    # padding_points = y_data_to_display[1] - y_origin_to_display[1]  # Difference in display (points)
    # # Set the y-axis label with calculated padding
    # ax.set_ylabel('Length (AA)', labelpad=padding_points)
    ax.set_ylabel('Length (AA)', labelpad=13)



    #deal with x axis, pretty easy
    ax.set_xlabel('Mean # deut.', labelpad=0.01)

    #Deal with z axis. It turns out that to optimize alignment, it's better to set both the tick labels and the axis label as text objects rather than using the built in features
    #z axis label, worked best as a text object
    ax.text2D(0.95, 0.99, "Pop. frac.", transform=ax.transAxes, rotation=0, ha='left', va='center')
    #ax.tick_params(axis='z', pad=10)
    #ax.set_zticks([0, 0.25, 0.5, 0.75])
    #plt.setp(ax.get_zticklabels(), verticalalignment='top')
    #set the z ticks as text instances to improve alignment
    ax.set_zticks([0, 0.25, 0.5, 0.75], labels=[])
    zticks = [0, 0.25, 0.5, 0.75]
    for tick in zticks:
        ax.text(ax.get_xlim()[1]+1 , ax.get_ylim()[1]+0 , tick, '{}'.format(tick), ha='left', va='center')



    #fig.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)  # Adjust the margins
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()


def MultiGaussian(xrange, deuts, fracs, sd=0.2):
    """
    This returns a sum of Gaussians, evaluated over a pre-specified domain (e.g. a np.arange )
    with respective means given by the values in the array deuts, and heights given by the values in the array fracs
    The standard deviation for each is given by s.d.
    """
    if len(deuts)==0:
        y = np.nan*np.zeros(len(xrange)) #don't even plot anything if no fit parameters were obtained
    else:
        y = np.zeros(len(xrange))
    for i, (deut, frac) in enumerate(zip(deuts, fracs)):
        y+=frac*norm.pdf(xrange, deut, sd)*np.sqrt(2*np.pi)*sd #multiplying by sqrt(2pi)*sigma ensures that, if frac=1, the height will be 1
    return y
    


def FracFolded_vs_len(fitparams, samples, lengths, peptide, charge, deut_time, reps,  smartn_curves=False, pvalue_thresh=np.nan,color=None, ax=None, label=None, ignore_mode=np.nan, normalize = True, folded_index=-1, linestyle = '-', markerstyle = '.', Return = False, plot=True, backups = {} ):
    """
    Plots fraction protected vs sample for a given peptide and charge
    If normalize = True, then normalizes that fraction protected by the value in a sample that is deemed to be "folded"
    The index for that sample is the one given by folded_index

    New on Dec 11 2024, added an option to provide a backup peptide/charge combo in case first choice is missing
    That's in the form of a dic, where the keys are (peptide, charge) with peptide itself being a tuple, and the values are (backup_peptide, backup_charge)
    """
    if ax==None and plot:
        fig, ax = plt.subplots()
    
    if is_zero_dimensional(ignore_mode):
        ignore_mode = [ignore_mode for l in range(len(samples))]

    if color==None:
        color='blue'
    if smartn_curves:
        if np.isnan(pvalue_thresh):
            raise RuntimeError('Must input a p-value threshold if using smartn_curves feature. \n Recommended to use config.Ncurve_p_accept to ensure consistency w fit')
        ncurves = Choose_ncurves(fitparams, samples, lengths, peptide, charge, deut_time, pvalue_thresh, verboise=False)
    else:
        ncurves = np.nan * np.ones(len(samples))
    #allfracs = np.nan*np.ones((len(reps), len(samples), 3)) #same as above for the fracs parameter
    fracs_vs_len = np.nan*np.ones((len(reps), len(samples))) #only plotting the fraction in the lightest mode
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        #print('moo')
        for r, rep in enumerate(reps):
            #print(deut_time)
            fits = hxex.filter_df(fitparams, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time], rep=rep)
            nboot_list = list(fits['nboot'].unique())

            if len(nboot_list) ==0: #means parameters don't exist. Try the backup
                if (peptide, charge) in backups.keys():
                    (backup_peptide, backup_charge) = backups[(peptide, charge)] #replace with the backup, then try the fitting again
                    fits = hxex.filter_df(fitparams, sample, peptide_ranges = Convert_to_pepstring(*backup_peptide), charge=backup_charge, timept=[deut_time], rep=rep)
                    nboot_list = list(fits['nboot'].unique())

            if len(nboot_list) > 0:  # a good proxy for the fact that this sample's parameters even exist
                if np.isnan(ncurves[i]):  # you haven't selected smart n curves, so use the value selected by the origianl hxex algorithm (which may vary wildly from replicate to replicate and lenght to length)
                    if min(nboot_list) > 0:
                        best_n_curves = fits[fits['nboot'] == 1]['ncurves'].values[0]  # from bootstrap #1, we can extract how many curves were used
                    else:
                        best_n_curves = fits['ncurves'].values[-1]
                else:
                    best_n_curves = ncurves[i]
                params_best_fit = fits[(fits['nboot'] == 0) & (fits['ncurves'] == best_n_curves)]['Fit_Params'].values[0]
                params_best_fit = [float(x) for x in params_best_fit.split()]
                scaler, nexs, mus, fracs = hxex.get_params(*params_best_fit, sort=True, norm=True, unpack=True)  # AB: If we removed spurious peaks, the scaler is gonna be off...
                if ~np.isnan(ignore_mode[i]) and ignore_mode[i] <=len(fracs):
                    print('ignoring mode {} in peptide {}'.format(ignore_mode[i], peptide))
                    fracs[ignore_mode[i]-1] = np.nan
                    fracs = fracs/np.nansum(fracs) #renormalize fractions

                # fracs_vs_len[r,i] = 0
                # nmodes = len(fracs)
                # for n in range(nmodes):
                #     if nexs[n]*mus[n]/(peptide[1] - peptide[0] -2)<=0.2: #the mode is less than 20% exchanged
                #         fracs_vs_len[r,i]+=fracs[n]
                nmodes_kept = np.sum(~np.isnan(fracs))
                protected_mode_index = np.where(~np.isnan(fracs))[0][0]
                if nmodes_kept>1:
                    fracs_vs_len[r,i] = fracs[protected_mode_index]

                else: #the fitting was only to one mode
                    #print('moo')
                    print('For peptide {} at time {}, the number deuterons is {} and the # of exchangable amides is {}'.format(peptide,sample, nexs[0]*mus[0], peptide[1] - peptide[0] -2))
                    #if nexs[0]*mus[0]/hxex.count_amides(peptide)>0.3: #if the mode is more than 33% exchanged, then we deem it unfolded
                    if nexs[protected_mode_index]*mus[protected_mode_index]/(peptide[1] - peptide[0] -2)>0.3: #if the mode is more than 33% exchanged, then we deem it unfolded. WE use a quick and direty way to account for total # of amides--prolines will throw this off
                        fracs_vs_len[r,i]=0
                    else:
                        fracs_vs_len[r,i]=1

    #fracs_vs_len = fracs_vs_len.flatten()
    fracs_vs_len = np.nanmean(fracs_vs_len, axis=0)
    if normalize:
        fracs_vs_len = fracs_vs_len/fracs_vs_len[folded_index]  #divide by the final value which corresponds to fully folded halotag

    if plot:
        ax.plot(lengths, fracs_vs_len, color=color, label=label, linestyle = linestyle)
        ax.scatter(lengths, fracs_vs_len, color=color, marker = markerstyle, label = label)
        ax.set_ylabel('Fraction protected mode')
    if Return:
        return fracs_vs_len
    
    

def Choose_ncurves(fitparams,samples, lengths, peptide, charge, deut_time, pvalue_thresh, verboise=False): #smartly choose # of curves for each length
    """
    The algorithm:  for determining optimal # of modes to fit, with length continuity, for more than 3 lengths

    1. Initialize by fitting all lengths to just 1 binomial
    2. Now try to fit all lengths to 2 binomials. For each length calculate the F test score. Each length for which the F test passes a significnace criterion in BOTH replicates will then "propose" a mode increase
    3. See which lengths proposed a mode increase. Accept the mode increase only for lengths where at least one neighboring length also proposed a mode increase
    4. Iterate steps 2-3 for those lengths that passed steps 3--this time increase from 2 to 3 binomials (stop after max_n_modes, which is usually 3 anyways)

    For less than 3 lengths, we always accept a proposal--this function noenetheless serves as a "filter" of sorts in the sense that it insists that we only augment # of modes if both technical replciates make the proposal
    """

    
    all_len_scores = hxex.filter_df(fitparams,samples, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=deut_time)
    all_len_scores = all_len_scores[all_len_scores['nboot'].isin([0])]

    nsamples = len(samples)
    best_ns = np.ones(nsamples)  #start w one curve for each




    ncurve_proposing = 2 #how many curves are we proposing to fit to? Start off with 2
    proposals = [False for i in  range(nsamples) ] #who has proposed an increase?
    for s, sample in enumerate(samples):
        if sample not in all_len_scores['sample'].unique(): #raise RuntimeError('Sample {} not found'.format(sample))
            print('Warning: Sample {} not found for peptide-z {}-{} '.format(sample, peptide, charge))
            proposals[s]=False
        else:
            reps = hxex.filter_df(all_len_scores, sample)['rep'].unique()
            pvalues = np.ones(len(reps))
            for r, rep in enumerate(reps):
                curr_ps = hxex.filter_df(all_len_scores, sample, rep=rep)['p-value'].to_numpy()
                if best_ns[s]==ncurve_proposing-1 and len(curr_ps)>=ncurve_proposing: # for 1->2 transition, first clause is trivially satisfied--second is to check whether algroithm even attempted 2--it may not have if # points < # parameters
                    pvalues[r] = curr_ps[ncurve_proposing-1]
            if verboise: print('For {} in {}, the 1->2 p values are: {}'.format(peptide, sample, pvalues))
            if np.max(pvalues)<pvalue_thresh:
                proposals[s]=True


    #check which lengths satisfy neighbour criteria

    for s, sample in enumerate(samples):
        if proposals[s]: #check that at least one neighbour is also making the proposal
            if len(samples)>3:  #for 3 or less samples, we don't worry about the neighbour criteria
                if s==0:
                    if proposals[1]: best_ns[s]=ncurve_proposing
                elif s==nsamples-1:
                    if proposals[nsamples-2]: best_ns[s]=ncurve_proposing
                else:
                    if proposals[s+1] or proposals[s-1]: best_ns[s]=ncurve_proposing
            else: #just accept a proposal by default
                best_ns[s] = ncurve_proposing



    #now increment to 3
    ncurve_proposing = 3 #how many curves are we proposing to fit to? Start off with 2
    proposals = [False for i in  range(nsamples) ] #who has proposed an increase?
    for s, sample in enumerate(samples):
        if sample not in all_len_scores['sample'].unique(): #raise RuntimeError('Sample {} not found'.format(sample))
            #print('Warning: Sample {} not found for peptide-z {}-{} '.format(sample, peptide, charge))
            proposals[s]=False
        else:
            reps = hxex.filter_df(all_len_scores, sample)['rep'].unique()
            pvalues = np.ones(len(reps))
            for r, rep in enumerate(reps):
                curr_ps = hxex.filter_df(all_len_scores, sample, rep=rep)['p-value'].to_numpy()
                if best_ns[s]==ncurve_proposing-1 and len(curr_ps)>=ncurve_proposing: #only even consider a proposal to 3 if you've previously chosen that 2, AND if the algorithm initially attempted 3 which may not happen if # points < # paramteres
                    pvalues[r] = curr_ps[ncurve_proposing-1]

            if verboise: print('For {} in {}, the 2->3 p values are: {}'.format(peptide, sample, pvalues))    
            if np.max(pvalues)<pvalue_thresh:
                proposals[s]=True

    #check which lengths satisfy neighbour criteria
    for s, sample in enumerate(samples):
        if proposals[s]: #check that at least one neighbour is also making the proposal
            if len(samples)>1:
                if s==0:
                    if proposals[1]: best_ns[s]=ncurve_proposing
                elif s==nsamples-1:
                    if proposals[nsamples-2]: best_ns[s]=ncurve_proposing
                else:
                    if proposals[s+1] or proposals[s-1]: best_ns[s]=ncurve_proposing
            else: #no neighbours so just acept by default
                best_ns[s] = ncurve_proposing


    if verboise: print(best_ns)
    return  np.array([int(n) for n in best_ns])


def Filter_spurious_peaks(Y, thresh=5):  #function by AB
    #Y is an array. We change the ith value to 0 if that value is more than thresh times the value at i-1 and i+1
    #TODO: Suppress warnings due to divide-by-zero!
    filteredy = cp.deepcopy(Y)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        for i in range(1, len(Y)-1):
            y = Y[i]
            if y/Y[i-1]>thresh and y/Y[i+1]>thresh :
                filteredy[i]=0
                #print('moo!!')
            else:
                filteredy[i]=y
    return filteredy


from matplotlib.colors import LinearSegmentedColormap

def modify_colormap_yellow(cmap, darken_factor=0.6):
    # Get the RGB colors of the colormap
    colors = cmap(np.linspace(0, 1, 256))
    # Identify the yellow region based on high R and G and low B
    yellow_indices = (colors[:, 0] > 0.8) & (colors[:, 1] > 0.8) & (colors[:, 2] < 0.2)
    # Darken the yellow region
    colors[yellow_indices, :3] *= darken_factor
    # Return the modified colormap
    return LinearSegmentedColormap.from_list('modified_colormap', colors)


#import turbo_colormap
#from matplotlib.colors import ListedColormap

# def GetCoreColor(pep):
#     #colormap = plt.cm.jet
#     colormap = modify_colormap_yellow(plt.cm.jet)
#     #colormap = ListedColormap(turbo_colormap_data)
#     meanvalue = np.mean([pep[0], pep[1]])
#     if meanvalue>217: #it's after the lid
#         meanvalue = (meanvalue - 217) + 129  #stich together the core regions
#     frac = meanvalue/((297-217)+ 129) #the denominator is the TOTAL length of the core region. This frac tells you the fraction of the way through the stiched together core that the peptide is
#     #print('{} : {}'.format(pep, frac))
#     frac = 1- np.exp(-2.4*frac)  #a nonlinear transformation that makes the color change a bit faster towards the n terminus
#     return colormap(frac)
    
def GetCoreColor(pep):
    meanvalue = np.mean([pep[0], pep[1]])
    if 0<=meanvalue<=25:
        color = 'darkblue'
    elif 25<=meanvalue<=50:
        color = 'blue'
    elif 50<=meanvalue<=75:
        color = 'darkcyan'
    elif 75<=meanvalue<=100:
        color = 'green'
    elif 100<=meanvalue<=130:
        color = 'olive'
    elif 217<=meanvalue<=250:
        color = 'goldenrod'
    elif 250<=meanvalue<=275:
        color = 'red'  
    else:
        color = 'maroon'
    return color
    
def GetLidColor(pep):
    colormap = plt.cm.RdPu
    meanvalue = np.mean([pep[0], pep[1]])

    frac = (meanvalue - 130)/(217-130) #fraction of the way through the lid

    frac = 0.2+ frac*0.8  #compress frac between 0.2 and 1, since colors close to 0 are too white..
    #print('{} : {}'.format(pep, frac))
    #frac = 1- np.exp(-2.4*frac)  #a nonlinear transformation that makes the color change a bit faster towards the n terminus
    return colormap(frac)
    
    
def Write_pml(avg_res_fdeut, key, length, full_prot_len, output_dir, headerpath):
    """
    Takes as input a dictionary whose keys correspond to sub-dictionaries for the respective lenghts
    Within each subdictionary, the keys are residue in one indexing and values tell you the fraction deuterated (or can also be fraction heavy mode) for each residue
    Also requires a header to give instructions on how to color things. ONe can be produced from a colormap using the next function

    It then writes a .pml file in output_dir to visualize residue level deuteration

    """
    outputpath = '{}/{}.pml'.format(output_dir, key)
    headerstr = ''
    with open(headerpath) as headeropen:
        for line in headeropen.readlines():
            headerstr = '{}{}'.format(headerstr, line)
    headeropen.close()

    with open(outputpath, 'w') as outputopen:
        outputopen.write(headerstr)
        for r, fdeut in avg_res_fdeut[key].items():
            if not np.isnan(fdeut):
                if fdeut>0.1:
                    string = str(int(100*fdeut))
                    if len(string)<2:
                        string = '{}{}'.format(0, string)
                    string = '{}x{}'.format(string[0], string[1])
                else:
                    string='1x0'
            
                writestring = 'color deutColor{},resi {}-{}\n'.format(string, r,r)
                outputopen.write(writestring)
        outputopen.write('\ndeselect\n')
    
        outputopen.write('select exittunnel, resi {}-{}\n'.format(length-35, length))
        outputopen.write('hide cartoon, exittunnel \nshow ribbon, exittunnel \ncolor grey40, exittunnel \n')
    
        if length<full_prot_len:
            #print('moo')
            for r in range(length+1, full_prot_len+1):
                outputopen.write('remove resi {}\n'.format(r))
            #outputopen.write('select absent, resi {}-{}\n'.format(length+1, full_prot_len))
            #outputopen.write('delete absent\n')        
    outputopen.close()

 


def generate_colormap_header(output_path, cmap='hot'):
    """
    This takes a colormap from matplotlib and extracts from it 100 different colors and 
    converts it to a header file for use with the previous Write_pml function
    Thanks GPT!!
    """
    # Create the colormap
    colormap = plt.cm.get_cmap(cmap, 100)
    
    # Open the header file for writing
    with open(output_path, 'w') as f:
        for i in range(100):
            # Get RGB values from the colormap
            r, g, b, _ = colormap(i / 99)
            
            # Scale RGB values to [0, 1] range
            r, g, b = round(r, 3), round(g, 3), round(b, 3)
            
            # Map i to your "1x0", "1x1", etc. format
            tens = i // 10
            units = i % 10
            color_name = f"deutColor{tens}x{units}"
            
            # Write PyMOL color setting
            f.write(f"set_color {color_name}=[{r},{g},{b}]\n")
    
        f.write('set_color neutral=[1.000,1.000,1.000]\n \nhide everything \nshow cartoon \ncolor neutral,all\n')
    print(f"New header file saved to {output_path}")


def Make_colorbar(cmap, spacing=0.25):
    #make a color bar for the cm with the help of GPT
    fig, ax = plt.subplots(figsize=(1, 6))
    # Create a scalar mappable object for the colorbar
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar to the plot
    cbar = fig.colorbar(sm, cax=ax, orientation='vertical', ticks=np.arange(0, 1+spacing, spacing))
    cbar.set_label('Frac. heavy peak', rotation=270, labelpad=30, fontsize=20,)
    cbar.ax.set_yticklabels([f'{x:.2f}' for x in np.arange(0, 1+spacing, spacing)], fontsize=20)

    # Display the color bar
    plt.show()

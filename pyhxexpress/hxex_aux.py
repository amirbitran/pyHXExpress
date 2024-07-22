import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys
sys.path.insert(0, '/Users/amirbitran/Dropbox/CurrentStuff/Postdoc/pyHXExpress') #inserting this path at position 0 means it's at the top of sys.path and thus, in the following import commands, python will priortize this version of pyhxex over the one that pip installed
import pyhxexpress.hxex as hxex
import warnings
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

def Convert_to_pepstring(start, end):
    start = str(start)
    while len(start)<4:
        start = '0{}'.format(start)
    
    end = str(end)
    while len(end)<4:
        end = '0{}'.format(end)
    return '{}-{}'.format(start, end)
    
def Plot_spectrum(deutdata, fitparams, sample, peptide, charge, deut_time, reps, deut_color='blue', undeut_color='gray', plot_undeut=True, norm='Sum', labelx=True, labely=True, ax=None, rawdata = [], Spurious_peak_thresh=5, LimitMZRange=True):
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
        rawdeut = hxex.filter_df(deutdata, sample, peptide_ranges = Convert_to_pepstring(*peptide), charge=charge, timept=[deut_time for i in range(len(reps))], rep=reps)
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
        ax.set_xlabel('mass (Da)', fontsize=20)
    if labely:
        if norm=='Sum':
            ax.set_ylabel('Norm. Intensity', fontsize=20)
        elif norm=='Max':
            ax.set_ylabel('Intensity/Max Int.', fontsize=20)
        else:
            ax.set_ylabel('Intensity', fontsize=20)
    ax.tick_params(labelsize=20)

    return deut


def Spec_vs_len(deutdata, fitparams,samples, lengths, peptide, charge, deut_time, reps, figsize = (16,6), xlim= None, norm='Max', plot_undeut=True, rawdata=[], savepath = None, LimitMZRange=True):
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
            Plot_spectrum(deutdata, fitparams,sample, peptide, charge, deut_time, rep, norm=norm,ax = ax[rep-1,i],labelx=labelx, labely=labely, plot_undeut=plot_undeut, rawdata=rawdata,LimitMZRange=LimitMZRange)
            ax[rep-1,i].set_title('{} AA'.format(length), fontsize=20)
            if xlim is not None:
                ax[rep-1,i].set_xlim(xlim)
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print('Plots of all spectra vs length saved to {}'.format(savepath))
        
def ReadRawData(metadf_run, sample, peptide, charge, deut_time, rep ):
    #A simple function to get raw data from SpecExport csv file, haven't extensivly tested
    row = hxex.filter_df(metadf_run, sample, range = peptide, charge=charge)
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

def Ndeut_vs_len(fitparams,samples, lengths, peptide, charge, deut_time, reps,markerscale= 100, plot_thresh = 0.2, savepath = None):
    #only plot if fraction of population is more than plot_thresh

    colors = ['r', 'b', 'g'] #for diff populations
    markers = ['*', 'o', 'v']

    fig, ax = plt.subplots()
    

    for r, rep in enumerate(reps):
        markersizes = []
        means = []
        ls = []
        colorlist = []


        for i, (sample, length) in enumerate(zip(samples, lengths)):
            fits = hxex.filter_df(fitparams, sample, range = peptide, charge=charge, timept=[deut_time for i in range(len(reps))], rep=rep)
            nboot_list = list(fits['nboot'].unique())
            if len(nboot_list)>0: #a good proxy for the fact that this sample's parameters even exist
                if min(nboot_list) > 0:
                    best_n_curves = fits[fits['nboot']==1]['ncurves'].values[0] #from bootstrap #1, we can extract how many curves were used
                else: best_n_curves = fits['ncurves'].values[-1]
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


    
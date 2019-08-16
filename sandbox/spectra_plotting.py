import numpy as np
import matplotlib.pyplot as pp

def plot_snz_modulations(spectra, fc=None):
    
    s1 = spectra['dark']
    ncond = s1.shape[1]
    
    if spectra.units.lower()=='db':
        pcomp = lambda x,y: x-y
        pfun = pp.plot
    else:
        pcomp = lambda x,y: x/y
        pfun = pp.semilogy
    
    for c in range(ncond):
        
        f = pp.figure()
        f.add_subplot(211)
        for n in ('dark', 'bright'):
            pxx = spectra[n]
            pfun(spectra.fx, pxx[:,c,:].T)
        pfun(spectra.fx, spectra.gray[:,c,:].T)
        if fc:
            pp.xlim(0, fc)
        pp.xlabel('Hz')
        pp.ylabel(spectra.units+'/Hz')
            
        f.add_subplot(212)
        for n in ('dark', 'bright'):
            pxx = spectra[n]
            pref = spectra['gray']
            pfun(spectra.fx, pcomp(pxx[:,c,:], pref[:,c,:]).T)
        if fc:
            pp.xlim(0, fc)
        pp.xlabel('Hz')
        pp.ylabel('Modulation '+spectra.units+'/Hz')
    
        
        
    
    


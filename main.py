def run_pandexo_on_planet(Jmag,Teff,Rstar,Rp,dur,logg=4.5,FeH=0.0,JWST_mode='NIRSpec G395M'):
    """This runs pandexo for a transit of a single planet in a given JWST mode.

    Input:
        Jmag, float: J-band magnitude (1.25 microns).
        Teff, float: Effective temperature of the host star in Kelvin.
        Rstar, float: Radius of the star in solar radii.
        Rp, float: Radius of the planet in Jupiter radii.
        dur, float: Transit duration in hours.
        logg, float: log(g) value of the star in cgs.
        FeH, float: Metallicity of the star [Fe/H].
        JWST_mode, str: The mode of JWST to simulate.
        subarray, str: The subarray being used for read-out."""
    import warnings
    warnings.filterwarnings('ignore')
    import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
    import pandexo.engine.justplotit as jpi
    import pickle as pk
    import numpy as np
    import os
    import matplotlib.pyplot as plt


    #Check the input:
    NS_L = ['NIRSpec Prism']
    NS_M = ['NIRSpec G140M','NIRSpec G295M','NIRSpec G395M']
    NS_H = ['NIRSpec G140H','NIRSpec G295H','NIRSpec G395H']
    NRSS = ['NIRISS SOSS']
    MI = ['MIRI LRS']
    NC = ['NIRCam F322W2', 'NIRCam F444W']#https://jwst-docs.stsci.edu/near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series (very bright stars possible, ~5th magnitude)
    allowed_modes = NS_L+NS_M+NS_H+NRSS+MI
    if JWST_mode not in allowed_modes:
        str = ''
        for m in allowed_modes:
            str+=m+','
        raise ValueError("JWST_mode %s not in allowed modes %s"%(JWST_mode,str[0:-1]))

    #Available subarrays:
    #NIRSpec: sub1024a,sub1024b,sub2048,sub512
    #NIRIS SOSS: substrip96,substrip256

    #JDocs says for NIRSpec:
    #SUB1024A captures the shorter wavelength portion of the spectrum on the NRS1 detector, and the longer wavelength portion of the spectrum on NRS2. Alternatively, SUB1024B captures the longer wavelength portion on NRS1 and the shorter wavelength portion on NRS2. The medium resolution (M) grating and PRISM spectra for BOTS map entirely to NRS1. However, for the high resolution (H) gratings, the use of SUB1024B will enable capturing a (semi-)contiguous mid-wavelength portion of the spectrum across the detector gap.
    #SUB1024A should not be used with the PRISM since the spectra do not project to the region of the detector covered but this subarray.

    #And for SOSS:
    #SUBSTRIP96  for bright targets, samples only 1st order
    #SUBSTRIP256 SOSS mode default, samples orders 1–3

    #For MIRI its only one option.
    if JWST_mode in NS_L+NS_M:
        subarray = 'sub1024b'
    if JWST_mode in NS_H:
        subarray = 'sub2048'
    if JWST_mode in NRSS:
        subarray = 'substrip96'#NIRISS SOSS for bright stars.
    if JWST_mode in MI:
        subarray = 'slitlessprism'




    #################### OBSERVATION INFORMATION
    exo_dict = jdi.load_exo_dict()
    exo_dict['observation']['sat_level'] = 80
    exo_dict['observation']['sat_unit'] = '%'
    exo_dict['observation']['noccultations'] = 1
    exo_dict['observation']['R'] = None
    exo_dict['observation']['baseline'] = 1.0
    exo_dict['observation']['baseline_unit'] = 'frac'
    exo_dict['observation']['noise_floor'] = 0

    #################### STAR INFORMATION
    exo_dict['star']['type'] = 'phoenix'
    exo_dict['star']['mag'] = Jmag
    exo_dict['star']['ref_wave'] = 1.25    	#If ['mag'] in: J = 1.25, H = 1.6, K = 2.22
    exo_dict['star']['temp'] = Teff
    exo_dict['star']['metal'] = FeH
    exo_dict['star']['logg'] = logg
    exo_dict['star']['radius'] = Rstar#In solar units.
    exo_dict['star']['r_unit'] = 'R_sun'

    #################### EXOPLANET INFORMATION
    exo_dict['planet']['type'] = 'constant'
    exo_dict['planet']['w_unit'] = 'um'
    exo_dict['planet']['radius'] = Rp
    exo_dict['planet']['r_unit'] = 'R_jup'
    exo_dict['planet']['transit_duration'] = dur
    exo_dict['planet']['td_unit'] = 'h'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'

    #################### BEGIN RUN
    inst_dict = jdi.load_mode_dict(JWST_mode)
    inst_dict["configuration"]["detector"]["subarray"]='sub2048'
    jdi.run_pandexo(exo_dict,inst_dict, save_file=True, output_file='temp.p')
    out = pk.load(open('temp.p','rb'))

    wl,spec,err= jpi.jwst_1d_spec(out,plot=False)
    # wl=wl[0]
    # spec=spec[0]
    # err = err[0]
    return(np.nanmean(err[0]),out['timing']['Transit+Baseline, no overhead (hrs)'])


def simulate_pandexo():
    """This takes a predefined table of exoplanets and runs pandexo on each of them, adding the output to the columns.
    Planets for which a mandatory input is missing are ignored."""
    import pickle
    import numpy as np
    transiting = pickle.load(open( "table.p", "rb" ))

    JWST_mode = 'NIRSpec G395M'
    #Available modes:
    # NIRSpec Prism - NIRSpec G395M - NIRSpec G395H - NIRSpec G235H - NIRSpec G235M - NIRCam F322W - NIRSpec G140M - NIRSpec G140H - MIRI LRS - NIRISS SOSS

    transiting[JWST_mode+'_error']=-1.0
    transiting[JWST_mode+'_time']=-1.0
    for i,row in enumerate(transiting):
        Jmag = row['st_j']
        Teff = row['st_teff'].to('K').value
        Rstar = row['st_rad'].to('solRad').value
        Rp = row['pl_radj'].to('jupiterRad').value
        dur = row['pl_trandur']*24.0
        logg = row['st_logg']
        FeH = row['st_metfe']

        trigger = 1
        errstr = ''
        if not isinstance(Jmag,np.float64): errstr+='no Jmag'
        if not isinstance(Teff,np.float64): errstr+=', no Teff'
        if Teff < 2000: errstr+=', Teff out of bounds'
        if not isinstance(Rstar,np.float64):  errstr+=', no Rstar'
        if not isinstance(Rp,np.float64):  errstr+=', no Rp'
        if not isinstance(dur,np.float64):  errstr+=', no duration'
        if not isinstance(logg,np.float64):  logg=4.5
        if not isinstance(FeH,np.float64):  logg=0.0
        if len(errstr) >= 1:
            print('Skipping '+row['pl_name']+' ('+errstr+').')
            print(Jmag,Teff,Rstar,Rp,dur,logg,FeH)
        else:
            err,time = run_pandexo_on_planet(Jmag,Teff,Rstar,Rp,dur,logg=4.5,FeH=0.0,JWST_mode=JWST_mode,subarray='sub2048')
            row[JWST_mode+'_error']=err
            row[JWST_mode+'_time']=time
            print(err)
        print('%s / %s'%(i,len(transiting)))
simulate_pandexo()
pickle.dump(transiting,open("table_with_pandexo.p", "wb" ))
# err,time = run_pandexo_on_planet(Jmag,Teff,Rstar,Rp,dur,logg=4.5,FeH=0.0,JWST_mode='NIRSpec G395M',subarray='sub2048')
# err,time = run_pandexo_on_planet(10.0,5000.0,1.0,1.0,2.0,logg=4.5,FeH=0.0,JWST_mode='NIRSpec G395M',subarray='sub2048')

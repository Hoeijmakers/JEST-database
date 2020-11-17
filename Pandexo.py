#This is a script that takes the Astropy table of transiting exoplanets output by the accompanied
#Jupyter notebook, and takes the columns for Jmag, Teff, Tstar, Rp and transit duration to prime
#Pandexo simulations for a selected JWST mode (see below).

#First below is a script that runs Pandexo for a single planet. This script is looped over in the script below, for all
#planets in the table, provided that they have the correct information.

#Running the entire transiting database for 2 modes of JWST takes about 8 hours on my mac. Adding simulation columns
#for other modes requires modifying the calls to "simulate_pandexo()" at the very bottom of this script.

# -Jens Hoeijmakers, May 2020.


def run_pandexo_on_planet(Jmag,Teff,Rstar,Rp,dur,logg=4.5,FeH=0.0,JWST_mode='NIRSpec G395M',planetname=''):
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
        planetname, str: The name of the planet, just for printing to terminal."""
    import warnings
    warnings.filterwarnings('ignore')
    import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
    import pandexo.engine.justplotit as jpi
    import pickle as pk
    import numpy as np
    import os
    import pdb
    import matplotlib.pyplot as plt


    #Check the input:
    NS_L = ['NIRSpec Prism']
    NS_M = ['NIRSpec G140M','NIRSpec G235M','NIRSpec G395M']
    NS_H = ['NIRSpec G140H','NIRSpec G235H','NIRSpec G395H']
    NRSS = ['NIRISS SOSS']
    MI = ['MIRI LRS']
    NC = ['NIRCam F322W2', 'NIRCam F444W']#https://jwst-docs.stsci.edu/near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series (very bright stars possible, ~5th magnitude)
    allowed_modes = NS_L+NS_M+NS_H+NRSS+MI+NC
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
    if JWST_mode in NC:
        subarray = 'subgrism128'


    print('Running %s for %s and subarray %s.'%(planetname,JWST_mode,subarray))
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


    try:
        inst_dict = jdi.load_mode_dict(JWST_mode)
        inst_dict["configuration"]["detector"]["subarray"]=subarray
    except:
        print('Unknown error in selecting mode from instrument dictionary. Entering debug mode.')
        pdb.set_trace()
    try:
        jdi.run_pandexo(exo_dict,inst_dict, save_file=True, output_file='temp.p')
    except:
        print('Unknown error in running pandexo. Are the exoplanet system parameters physical? Entering debug mode.')
        pdb.set_trace()

    try:
        out = pk.load(open('temp.p','rb'))
        wl,spec,err= jpi.jwst_1d_spec(out,plot=False)
    except:
        print('Unknown error in collecting Pandexo output. Entering debug mode.')
        pdb.set_trace()
    return(np.nanmean(err[0]),out['timing']['Transit+Baseline, no overhead (hrs)'])


def simulate_pandexo(JWST_mode,tablename='table_pandexo.p',table_outname='table_pandexo.p'):
    """This takes a predefined table of exoplanets and runs pandexo on each of them, adding the Pandexo output to the columns.
    Planets for which a mandatory input is missing are ignored.

    Set the desired JWST_mode below. The default subarray will be selected accordingly.

    Planets for which values are missing get a -1.

    Set the TESS keyword if you supply TESS magnitudes instead of Jmags."""
    import pickle
    import numpy as np

    with open(tablename, "rb" ) as f:
        transiting = pickle.load(f)

    #Prepare output columns to be added to the table:
    transiting[JWST_mode+'_error']=-1.0
    transiting[JWST_mode+'_time']=-1.0
    transiting['dur_approx_flag']=0.0

    #We are going to loop through each entry in the table and run Pandexo on it if the planet has sufficient data provided.
    for i,row in enumerate(transiting):
        Jmag = row['st_j']
        Teff = row['st_teff'].to('K').value
        Rstar = row['st_rad'].to('solRad').value
        Rp = row['pl_radj'].to('jupiterRad').value
        dur = row['pl_trandur']*24.0
        dur_approx = row['duration_predicted'].to('h').value
        logg = row['st_logg']
        FeH = row['st_metfe']
        err = np.nan
        time = np.nan#Set to NaN at the start.

        #Now we check the integrity of the needed variables in this row:
        errstr = ''#Message saying why a certain planet is skipped (i.e. for the reason of which value being missing).
        if not isinstance(Jmag,np.float64): errstr+='no Jmag'#If Jmag is not filled in, start building up the error string.
        if not isinstance(Teff,np.float64): errstr+=', no Teff'#If Teff is not filled in.
        if Teff < 2000: errstr+=', Teff out of bounds'#or if it is out of bounds...
        if not isinstance(Rstar,np.float64):  errstr+=', no Rstar'#If Rstar is not filled in.
        if Rstar <= 0.0: errstr+=', Rstar is zero??'
        if not isinstance(Rp,np.float64):  errstr+=', no Rp'#If Rp is not filled in.
        if not isinstance(dur,np.float64):
            if np.isfinite(dur_approx):
                row['dur_approx_flag']=1.0
                print('      Using duration approximation to proceed')
            else:
                errstr+=', no duration'#If the transit duration is not filled in.....
        if not isinstance(logg,np.float64):  logg=4.5#Put log(g) to a standard value if not provided.
        if not isinstance(FeH,np.float64):  logg=0.0#as well as Fe/H.

        #Test if any errors were triggered:
        if len(errstr) >= 1:#.....then the error string has a length greater than 0, and we skip to the next planet.
            print('      Skipping '+row['pl_name']+' ('+errstr+').')
            print('      Jmag: %s, Teff: %s, Rstar: %s, Rp: %s, duration: %s, duration_approx: %s, log(g): %s, FeH: %s'%(Jmag,Teff,Rstar,Rp,dur,dur_approx,logg,FeH))
        else:#else, meaning if all values were accounted for, we try to run Pandexo and collect the output.
            if row['dur_approx_flag']==1.0:
                print('      Jmag: %s, Teff: %s, Rstar: %s, Rp: %s, duration_approx: %s'%(Jmag,Teff,Rstar,Rp,dur_approx))
                err,time = run_pandexo_on_planet(Jmag,Teff,Rstar,Rp,dur_approx,logg=4.5,FeH=0.0,JWST_mode=JWST_mode,planetname=row['pl_name'])
            else:
                print('      Jmag: %s, Teff: %s, Rstar: %s, Rp: %s, duration: %s'%(Jmag,Teff,Rstar,Rp,dur))
                err,time = run_pandexo_on_planet(Jmag,Teff,Rstar,Rp,dur,logg=4.5,FeH=0.0,JWST_mode=JWST_mode,planetname=row['pl_name'])
            row[JWST_mode+'_error']=err
            row[JWST_mode+'_time']=time
    with open(table_outname,"wb") as f:
        pickle.dump(transiting,f)

def simulate_pandexo_TOI(JWST_mode,tablename='table_pandexo.p',table_outname='table_pandexo.p'):
    """This takes a predefined table of exoplanets and runs pandexo on each of them, adding the Pandexo output to the columns.
    Planets for which a mandatory input is missing are ignored.

    Set the desired JWST_mode below. The default subarray will be selected accordingly.

    Planets for which values are missing get a -1.."""
    import pickle
    import numpy as np
    import astropy.units as u
    import pdb
    import urllib
    with open(tablename, "rb" ) as f:
        transiting = pickle.load(f)

    #Prepare output columns to be added to the table:
    transiting[JWST_mode+'_error']=-1.0
    transiting[JWST_mode+'_time']=-1.0

    url_stem = 'https://exofop.ipac.caltech.edu/tess/target.php?id='

    #We are going to loop through each entry in the table and run Pandexo on it if the planet has sufficient data provided.
    for i,row in enumerate(transiting):
        # Jmag = row['MT']
        #We need to scrape the Jmag from the exofop website:

        Jmag = row['Jmag']
        if Jmag < 0.0:
            print('Need to scrape the Jmag:')
            url = url_stem+str(row['TIC ID'])
            try:
                new_html = ((urllib.request.urlopen(url)).read()).decode('utf-8')
                Jmag = np.float64(new_html.split('Band')[1].split('J</td>\n<td>')[1].split(' <div class')[0])
                row['Jmag']=Jmag#Update the table. Will be written later so the scraping only happens once.
            except:
                print('Error occured in scraping. Skipping this planet.')
            print('Jmag = %s'%Jmag)
        Teff = row['Teff']
        Rstar = row['Rstar']
        Rp = (row['r_earth']*u.earthRad).to('jupiterRad').value
        dur = row['duration']

        logg = 4.5
        FeH = 0.0
        err = np.nan
        time = np.nan#Set to NaN at the start.

        #Now we check the integrity of the needed variables in this row:
        errstr = ''#Message saying why a certain planet is skipped (i.e. for the reason of which value being missing).
        if not isinstance(Jmag,np.float64):
            errstr+='no mag'#If Jmag is not filled in, start building up the error string.
            print(type(Jmag))
        else:
            if Jmag < 1.0 or Jmag > 26:
                errstr+=', Jmag out of bounds?'
        if not isinstance(Teff,np.float64): errstr+=', no Teff'#If Teff is not filled in.
        if Teff < 2000: errstr+=', Teff out of bounds'#or if it is out of bounds...
        if not isinstance(Rstar,np.float64):  errstr+=', no Rstar'#If Rstar is not filled in.
        if Rstar <= 0.0: errstr+=', Rstar is zero??'
        if not isinstance(Rp,np.float64):  errstr+=', no Rp'#If Rp is not filled in.
        if not isinstance(dur,np.float64): errstr+=', no duration'#If the transit duration is not filled in.....

        #Test if any errors were triggered:
        if len(errstr) >= 1:#.....then the error string has a length greater than 0, and we skip to the next planet.
            print('      Skipping '+row['pl_name']+' ('+errstr+').')
            print('      Jmag: %s, Teff: %s, Rstar: %s, Rp: %s, duration: %s:'%(Jmag,Teff,Rstar,Rp,dur))
        else:#else, meaning if all values were accounted for, we try to run Pandexo and collect the output.
            err,time = run_pandexo_on_planet(Jmag,Teff,Rstar,Rp,dur,logg=4.5,FeH=0.0,JWST_mode=JWST_mode,planetname=row['pl_name'])
            row[JWST_mode+'_error']=err
            row[JWST_mode+'_time']=time
        print('%s / %s'%(i,len(transiting)))
    with open(table_outname,"wb") as f:
        pickle.dump(transiting,f)


#Calling all of the above:
simulate_pandexo('NIRSpec G395M',tablename='table.p')#table.p is specified for the first mode. Output written to table_pandexo.p, which is used for the others by default.
# simulate_pandexo('NIRSpec G235M')
# simulate_pandexo('NIRSpec Prism')
# simulate_pandexo('NIRSpec G140M')
# simulate_pandexo('NIRISS SOSS')
# simulate_pandexo('NIRCam F322W2')
# simulate_pandexo('NIRCam F444W')
# simulate_pandexo('MIRI LRS')

#And calling it again for the TOIs.
# simulate_pandexo_TOI('NIRSpec G140M',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')
# simulate_pandexo_TOI('NIRSpec G235M',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')
# simulate_pandexo_TOI('NIRSpec G395M',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')
# simulate_pandexo_TOI('NIRSpec Prism',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')
# simulate_pandexo_TOI('NIRISS SOSS',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')
# simulate_pandexo_TOI('NIRCam F322W2',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')
# simulate_pandexo_TOI('NIRCam F444W',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')
# simulate_pandexo_TOI('MIRI LRS',tablename='TOI_table_pandexo.p',table_outname='TOI_table_pandexo.p')

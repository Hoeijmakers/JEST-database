# JEST-database
A database of exoplanets.


This repo contains a Jupyter notebook that reads exoplanets and TESS candidates from the NASA exoplanet archive and Exofop, and prints relevant parameters to table, to enable system comparison. Pandexo.py is used to pass the systems through Pandexo, in order to predict which planets are observable. The user may impose constraints in equilibrium temperature, radius and brightness to narrow down the selection, and order planets along any of the output columns.

<br/>
For many planets, some parameters were missing that inhibited the application of Pandexo, or the computation of the scale height. For most confirmed planets, this was due to missing measurements of the planet mass, or missing transit durations. For this reason, when passing planets through Pandexo we assume (always) that the mass of the planet follows an empirical mass-radius relationship (in the table, the user may judge whether that approximation is good, for planets that have a mass measurement). For cases where the transit duration was missing (a minority of cases), we approximate the duration by assuming an impact parameter of 0.6.
<br/>
For unconfirmed TOIs, stellar masses were not provided, so these were computed according to a mass-radius relationship as well.

<br/><br/>
To run, simply open the iPython notebook file: `jupyter notebook selection_database.ipynb`.

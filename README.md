# Finite_source_Microlensing

I have several codes to evaluate the observing features for finite-source microlensing events. The codes are: 

"EFSE_1.cpp": generates many finite-source microlensing events from uniform source stars and determines the observing features for each light curve numerically 
"EFSE_2.cpp": generates many finite-source microlensing events from uniform source stars and determines the observing features for each light curve numerically 
"FWHM_limb.cpp":  calculates the FWHM of finite-source microlensing events from different rho*, u0, and limb-darkening coeffcients.  
"DeltaTHLimb.cpp":  calculates the Delta_TH of finite-source microlensing events from different rho*, u0, and limb-darkening coeffcients.  
"FPL.cpp":  calculate the FPL of finite-source microlensing events from different rho*, u0, and limb-darkening coeffcients.  

These codes should be performed with VBBinaryLensing.cpp (made by Valerio Bozza) to make lightcurves

"Analyze_without.py":  makes a machine learning model to find teh relation between observing features and lensing parameters for finite-source microlensing events  from uniform source stars 
"Analyze_withLimb.py": makes a machine learning model to find teh relation between observing features and lensing parameters for finite-source microlensing events  from limb-darkened source stars 
"Ntree.py": calculates the errors for the machine learning model (Random Forest from decision trees) versus the number of trees
"ntree_plot.py":  plots R2-score vereus the number of trees
"lightcure.py":  plots a list of lightcurves 


These codes should be performed by Python3.  


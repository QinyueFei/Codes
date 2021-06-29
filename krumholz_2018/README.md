### What's in This Repository ###

This repository contains python code and the associated data necessary to produce all the figures in [Krumholz, Burkhart, Forbes, & Crocker, 2018, MNRAS, 477, 2716](https://ui.adsabs.harvard.edu/#abs/2018MNRAS.477.2716K/abstract).

The contents are as follows:

* *data*: this directory contains two subdirectories, *data/sfr* and *data/kinematic* that contain the data shown in figures 2-3 and 4, respectively, in the paper. Details on the data sets themselves are given in the `README.txt` files included in each subdirectory.
* *example_sol.py*: this script produces figure 1 in the paper
* *sflaw.py*: this script produces figures 2 and 3 in the paper
* *sfrvdisp.py*: this script produces figure 4 in the paper
* *inflow_sfr.py*: this script produces figures 5 and 6 in the paper
* *galform.py*: this script produces figure 7 in the paper
* *kmtnew.py*: this module contains the routine fH2KMTnew that implements the KMT+ model of Krumholz, 2013, MNRAS, 436, 2747; it is called by the other scripts
* *sigma_sf.py*: this module contains the routine sigma_sf that computes the quantity sigma_sf defined in equation 39 of the paper; it is called by the other scripts

The scripts require the following libraries:

* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [astropy](http://www.astropy.org/)


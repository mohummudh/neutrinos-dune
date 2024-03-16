# Characterisation of neutrino-induced particles at DUNE

The code presented achieved two tasks: 
1. classifying individual particles as tracks or showers
2. classifying whole events as CC muon or CC electron or NC neutrino induced interactions.

We implemented a naive Bayes estimator and a BDT for track-shower classification, achieving a best accuracy of 97% using the BDT. We implemented a logic flow using BDTs to classify events, and implemented a CNN to improve classification performance - this acheived an accuracy of 85%. 

5 original features were developed for the traditional ML methods, the CNNs used 256x256 pixel images. 

Simulated data from the DUNE experiment was used, which two methods of particle reconstruction: real and perfect reconstruction. 

Detailed report given in the folder: mohammed

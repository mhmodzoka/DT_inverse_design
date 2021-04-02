# DT_inverse_design
We show how a decision tree can perform inverse design, and return design rules.

The codes here are mainly for predicting spectral emissivity of particles given material and geometry using decision trees and random forest. Also, the codes perform inverse design (i.e., given a desired spectral emissivity, the code return particle geometries that can achieve it, along with design rules).

More details:
Interpretable Forward and Inverse Design of Particle Spectral Emissivity Using Common Machine-Learning Models
https://www.sciencedirect.com/science/article/pii/S2666386420302812

# How to run?
The first step, run the file "src/forward.py" to create the forward ML models (DT, RF and DTGEN), which will be saved in "./cache"

The second step, run the file "src/inverse.py" to perform the inverse design.



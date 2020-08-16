
# Neural Network Dipole Model
This is a neural network model to predict dipole moments of water under extreme conditions ( high pressure and high temperature ). The training data are from first-principles MD simulations at 1000K-1GPa, 1000K-5GPa, 1000K-10GPa, 2000K-5GPa, 2000K-10GPa, and ambient conditions. The recommended TensorFlow version is 1.14 or later. This model covers a large P-T range and may be even extrapolated up to 2000K and 30GPa. Here, you can use this model to predict the dipole moment of condensed water.
#### Usage:
1. Use command `python generate_dataset.py` to generate input data for the model. The format of input trajectory is [gro file format](http://manual.gromacs.org/archive/5.0.4/online/gro.html) for Gromacs. 

2. Use command `python predict.py` to load the model from TensorFlow checkpoint file located in the TF_weight folder and then predict dipoles. 


# Neural Network Dipole Model
This is a neural network model to predict dipole moments of water under extreme conditions ( high pressure and high temperature ). It is built base on data from first-principle MD simulations at various conditions using TensorFlow(v1.14), including 1000K-1GPa, 1000K-5GPa, 1000K-10GPa, 2000K-5GPa, 2000K-10GPa,  and ambient conditions. We believe this model can cover a wide PT range as predicted dipoles at 2000K-30GPa is in good agreement with DFT results. Here, you can use this model to 
#### Usage:
1. Use command `python generate_dataset.py` to generate input data for the model. The format of input trajectory is [gro file format](http://manual.gromacs.org/archive/5.0.4/online/gro.html) for Gromacs. 

2. Use command `python predict.py` to load the model from TensorFlow checkpoint file located in the TF_weight folder and then predict dipoles. 

#!/home/yquanac/Programs/anaconda3/envs/pytorchenv/bin/python
# gro2data.py
# Convert the .gro file to training set including H,O positions and corresponding dipole moments.
import string
import math
import sys
import os
import random
import itertools
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt

e_charge = 1.60217653E-19
c_0 = 299792458
enm2D = (1.0E-9)*e_charge/(1.0E-21/c_0)

class atom_type:
    def __init__(self):
        self.xyz = [] # three float

class molecule_type:
    def __init__(self):
        self.O_xyz = [] # three float
        self.H_xyz = [] # six float
        self.H_dist = [] # OH length list, float
        self.M_xyz = [] # 4*3 float
        self.dipole_vector = []  # three float
        self.dipole = 0.0 # single float
        self.local_unit_vectors = []
    def test_complete(self):
        if len(self.M_xyz) == 12 and len(self.H_xyz) >= 6:
            return True
        else:
            return False

def distance_old(O1, O2, a, b, c):
# move it from imagine far away to the original cell
    O1[0] = O1[0]%a
    O1[1] = O1[1]%b
    O1[2] = O1[2]%c
    O2[0] = O2[0]%a
    O2[1] = O2[1]%b
    O2[2] = O2[2]%c
# End move

    if abs(O1[0]-O2[0])>(a/2.0):
        xx = a - abs(O1[0]-O2[0])
    else:
        xx = abs(O1[0]-O2[0])

    if abs(O1[1]-O2[1])>(b/2.0):
        yy = b - abs(O1[1]-O2[1])
    else:
        yy = abs(O1[1]-O2[1])

    if abs(O1[2]-O2[2])>(c/2.0):
        zz = c - abs(O1[2]-O2[2])
    else:
        zz = abs(O1[2]-O2[2])

    return math.sqrt(xx**2 + yy**2 + zz**2)


def distance(O1, O2, a, b, c):
    '''
    Calculate PBC distance without modifying passed in variables.
    '''
    pbc_vec = pbc_vector(O1, O2, a, b, c)
    return np.linalg.norm(pbc_vec)

def projection(vector_1,vector_2):
    '''
    The projection of vector_1 on vector_2. 

    Args:
        vector_1 and vector_2: lists with length of 3.
    Return: 
        A float representing the projection. 
    '''
    if len(vector_1)==3 and len(vector_2)==3:
        norm_2 = np.linalg.norm(vector_2)
        if norm_2!=0:
            inner_product=np.inner(vector_1,vector_2)
            return inner_product / norm_2
        else:
            sys.exit('Error: attempts to project a vector on a zero vector.')
    else:
         sys.exit(f'Error: arguments of projection have different dimensions {len(vector_1)} and {len(vector_2)}')



def pbc_vector_old(Olist, plist, a, b, c):
    '''
    Compute the vector pointing from O to p with periodic boundary condition. Olist and plist are changed in place by the function.

    Args:
        Olist: A list with length 3 representing position of O. 
        plist: A list with length 3 representing position of p. 
        a,b,c: Lattice constants.

    Return:
        A list with length 3 representing the vector pointing from O to p with PBC implemented.  
    '''
    Olist[0] = Olist[0]%a
    Olist[1] = Olist[1]%b
    Olist[2] = Olist[2]%c
    plist[0] = plist[0]%a
    plist[1] = plist[1]%b
    plist[2] = plist[2]%c

    dx = plist[0] - Olist[0]
    if dx > a/2.0:
        dx -= a
    if dx <= -a/2.0:
        dx += a

    dy = plist[1] - Olist[1]
    if dy > b/2.0:
        dy -= b
    if dy <= -b/2.0:
        dy += b

    dz = plist[2] - Olist[2]
    if dz > c/2.0:
        dz -= c
    if dz <= -c/2.0:
        dz += c

    return [dx, dy, dz]

def pbc_vector(O1, O2, a, b, c):
    '''
    Calculate PBC distance without modifying the passed in variable. 
    '''
    x_diff = (O2[0] - O1[0]) % a
    y_diff = (O2[1] - O1[1]) % b
    z_diff = (O2[2] - O1[2]) % c
    if x_diff > 0.5 * a:
        x_diff -= a
    elif x_diff < -0.5 * a:
        x_diff += a
    if y_diff > 0.5 * b:
        y_diff -= b
    elif y_diff < -0.5 * b:
        y_diff += b
    if z_diff > 0.5 * c:
        z_diff -= c
    elif z_diff < -0.5 * c:
        z_diff += c
    
    return [x_diff, y_diff, z_diff]

def assemble(Olist, Hlist, Mlist, a, b, c):
    Molist = []
    for O_i in Olist:
        Mol = molecule_type()
        Mol.O_xyz = O_i.xyz
        H_nn_dict = {}
        for i in range(len(Hlist)):
            dist = distance(O_i.xyz, Hlist[i].xyz, a, b, c)
            if dist > 0.05 and dist < 0.20:
                H_nn_dict[i] = dist
        # find the nn one      
        H_sorted = sorted(H_nn_dict.items(), key=lambda x:x[1])
        for x,y in H_sorted:
            Mol.H_xyz += Hlist[x].xyz
        #print H_sorted, Mol.H_xyz
        if len(Mol.H_xyz) < 6:
            print("H Error:", len(Mol.H_xyz), O_i.xyz, Mol.H_xyz)
        
        for M_i in Mlist:
            dist = distance(O_i.xyz, M_i.xyz, a, b, c)
            if dist > 0.001 and dist < 0.09:
                Mol.M_xyz += M_i.xyz
                if len(Mol.M_xyz) > 12:
                    print("M Error:", len(Mol.M_xyz), O_i.xyz, M_i.xyz)
        
        Molist.append(Mol)
    
    return Molist

def dipole_moment_vector_D(molecule,a,b,c):
    '''
    Return an array with 3 dipole moment components of the molecule. 

    Args:
        molecule: a water molecule object.
        a,b,c: lattice parameters.
    Return:
        dipole_vector: an array with 3 dipole moment components (x,y,z). The unit is Debye.
    '''
    dipole_vector_enm=[1.0*pbc_vector(molecule.O_xyz, molecule.H_xyz[0:3], a, b, c)[j]\
                                        + 1.0*pbc_vector(molecule.O_xyz, molecule.H_xyz[3:6], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[0:3], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[3:6], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[6:9], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[9:12], a, b, c)[j]\
                                    for j in range(3)]
    dipole_vector = [enm2D * dipole for dipole in dipole_vector_enm]
    return dipole_vector

def dipole_moment_vector_enm(molecule,a,b,c):
    '''
    Return an array with 3 dipole moment components of the molecule. 

    Args:
        molecule: a water molecule object.
        a,b,c: lattice parameters.
    Return:
        dipole_vector: an array with 3 dipole moment components (x,y,z). The unit is e*nm.
    '''
    dipole_vector_enm=[1.0*pbc_vector(molecule.O_xyz, molecule.H_xyz[0:3], a, b, c)[j]\
                                        + 1.0*pbc_vector(molecule.O_xyz, molecule.H_xyz[3:6], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[0:3], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[3:6], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[6:9], a, b, c)[j]\
                                        + (-2.0)*pbc_vector(molecule.O_xyz, molecule.M_xyz[9:12], a, b, c)[j]\
                                    for j in range(3)]
    return dipole_vector_enm

def group_nn(train_nn_num,water_list,a,b,c):
    '''
    Group specific number of nearest neighbors. Treat the oxygen as the water position.
    Return: 
        needed_nn_list: a 2-d array of nearest neighbors indices. 
                        The i-th row j-th coloumn is the index of the j-th nearest neighbor index of the i-th molecule.       

    '''
    if train_nn_num<len(water_list):
        # The i,j element in the matrix is the distance between the i-th and j-th molecules.  
        distance_matrix=np.zeros((len(water_list),len(water_list)))
        for idx_1 in range(len(water_list)):
            for idx_2 in range(idx_1,len(water_list)):
                distance_matrix[idx_1][idx_2]=distance(water_list[idx_1].O_xyz,water_list[idx_2].O_xyz,a,b,c)
                distance_matrix[idx_2][idx_1]=distance_matrix[idx_1][idx_2]
        
        full_nn_list=np.argsort(distance_matrix)
        # Exclude the center molecule itself. 
        needed_nn_list=full_nn_list[:,1:train_nn_num+1]
        return needed_nn_list
    else:
        sys.exit('Specified nearest neighbors number out of upper bound')

def global2local_old(molecules, center_water,a,b,c):
    '''
    Transform global reference frame coordintates to local reference frame defined according to the center water molecule.

    The 
    Args:
        molecules: a list of molecules.
        center_water: the center water molecule.
        a,b,c: Lattice constants

    Return: 
        A list of molecules represented in local reference frame (no M included).
    '''
    molecules_local_frame=[]
    local_origin=np.array(center_water.O_xyz)
    # Use coordinate of H1 and the inverted coordinate of H2 to calculate the middle point of H1 and H2.
    neg_H_2_pos=[-coordinate for coordinate in center_water.H_xyz[3:6]]
    H_mid=0.5*(np.array(pbc_vector(center_water.H_xyz[0:3],neg_H_2_pos,a,b,c)))
    # Local orthorgonal base vectors expressed in global reference frame. 
    local_vectors=np.zeros((3,3))
    local_vectors_norms=np.zeros(3)
    local_unit_vectors=np.zeros((3,3))
    local_axes_labels=['x','y','z']
    # Local x vector.
    local_vectors[0] = pbc_vector(center_water.H_xyz[0:3],center_water.H_xyz[3:6],a,b,c)
    # Vector pointing from origin to the midpoint of 2 H atoms. 
    origin2H_mid = pbc_vector(local_origin, H_mid, a, b, c)
    # Local z vector
    local_vectors[2] = np.cross(local_vectors[0],origin2H_mid)
    # Local y vector. In the plane of H2O and perpendicular to x vector. 
    local_vectors[1] = np.cross(local_vectors[2], local_vectors[0])

    for i in range(len(local_vectors)):
        local_vectors_norms[i]=np.linalg.norm(local_vectors[i])
        if local_vectors_norms[i]==0:
            sys.exit('Local base vectors error: {} vector has zero norm'.format(local_axes_labels[i]))

    local_unit_vectors=[local_vectors[i] / local_vectors_norms[i] for i in range(len(local_vectors))]
    
    molecules_local_frame=[]
    for molecule_global_frame in molecules:
        local_molecule=molecule_type()
        # Project the vector pointing from local reference origin to the atom, to the corresponding local reference frame axis.
        local_molecule.O_xyz=[projection(pbc_vector(local_origin,molecule_global_frame.O_xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
        local_molecule.H_xyz[0:3]=[projection(pbc_vector(local_origin,molecule_global_frame.H_xyz[0:3],a,b,c),local_unit_vectors[i]) for i in range(3)]
        local_molecule.H_xyz[3:6]=[projection(pbc_vector(local_origin,molecule_global_frame.H_xyz[3:6],a,b,c),local_unit_vectors[i]) for i in range(3)]
        local_molecule.dipole_vector[0:3]=[projection(molecule_global_frame.dipole_vector[0:3],local_unit_vectors[i]) for i in range(3)]
        molecules_local_frame.append(local_molecule)

    return molecules_local_frame

# def atom_global2local(atom_list, center_water,a,b,c):
#     '''
#     DEPRECATED becaused it might split molecules by individually implementing PBC. 
#     Transform global reference frame coordintates to local reference frame defined according to the center water molecule.
#     The local reference is differnt from global2local_old.

#     Args:
#         atom: a list of atoms.
#         center_water: the center water molecule.
#         a,b,c: Lattice constants

#     Return: 
#         A list of molecules represented in local reference frame (no M included).
#     '''
#     local_origin=np.array(center_water.O_xyz)
#     # Local orthorgonal base vectors expressed in global reference frame. 
#     local_vectors=np.zeros((3,3))
#     local_vectors_norms=np.zeros(3)
#     local_unit_vectors=np.zeros((3,3))
#     local_axes_labels=['x','y','z']
#     # Vector pointing from O to an H atom.
#     origin2first_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[0:3], a, b, c)
#     origin2second_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[3:6], a, b, c)
#     # Local x vector, pointing from origin to the midpoint of 2 H atoms. 
#     local_vectors[0] = [(origin2first_H[i] + origin2second_H[i]) * 0.5 for i in range(len(origin2first_H))]
#     # Local z vector
#     local_vectors[2] = np.cross(origin2first_H, origin2second_H)
#     # Local y vector. In the plane of H2O and perpendicular to x vector. 
#     local_vectors[1] = np.cross(local_vectors[2], local_vectors[0])

#     for i in range(len(local_vectors)):
#         local_vectors_norms[i]=np.linalg.norm(local_vectors[i])
#         if local_vectors_norms[i]==0:
#             print('Local base vectors error: {} vector has zero norm'.format(local_axes_labels[i]))
#             break

#     local_unit_vectors = [local_vectors[i] / np.linalg.norm(local_vectors[i]) for i in range(len(local_vectors))]
#     atoms_local_frame=[]
#     for atom_global_frame in atom_list:
#         local_atom=atom_type()
#         # Project the vector pointing from local reference origin to the atom, to the corresponding local reference frame axis.
#         local_atom.xyz = [projection(pbc_vector(local_origin,atom_global_frame.xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_atom.xyz = [projection(pbc_vector(local_origin,atom_global_frame.xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_atom.xyz = [projection(pbc_vector(local_origin,atom_global_frame.xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
#         atoms_local_frame.append(local_atom)
#     return atoms_local_frame

def copy_enlarge_box(molecules, a,b,c,n, with_Wannier=False):
    '''
    Copy water molecules to fill a larger box of dimension na, nb, nc.
    Parameters:
        molecules: array_like
            ALL water molecules in original form.
        a,b,c: float
            lattice constants.
        n: int
            The factor of copy on each axis. 
    Return: array_like 
        more_molecules: molecules in a larger box
    '''
    # First fold back all molecules to one box of a*b*c
    more_molecules = []
    for unfolded_molecule in molecules:
        origin = [0.0, 0.0, 0.0]
        folded_molecule = molecule_type()
        folded_molecule.O_xyz = pbc_vector(origin, unfolded_molecule.O_xyz, a, b, c)
        folded_molecule.dipole_vector = dipole_moment_vector_D(unfolded_molecule, a, b, c)
        # Attach hydrogen to oxygen instead of directly calculate PBC position of hydrogen to prevent molecules splitting.
        for H_idx in range(2):
            OH = pbc_vector(unfolded_molecule.O_xyz, unfolded_molecule.H_xyz[3 * H_idx:3 * (H_idx + 1)], a, b, c)
            folded_molecule.H_xyz[3 * H_idx:3 * (H_idx + 1)] = np.array(folded_molecule.O_xyz) + np.array(OH)
        if with_Wannier:
            for M_idx in range(4):
                OM = pbc_vector(unfolded_molecule.O_xyz, unfolded_molecule.M_xyz, a, b, c)
                folded_molecule.M_xyz[3 * M_idx:3 * (M_idx + 1)] = np.array(folded_molecule.O_xyz) + np.array(OM)
        more_molecules.append(folded_molecule)
    for (i, j, k) in itertools.product(range(n), repeat=3):
        # Molecules in this box have been in more_molecules.
        if not ((i, j, k) == (0, 0, 0)):
            for folded_molecule in more_molecules[0:len(molecules)]:
                copied_molecule = molecule_type()
                shift_vec = np.array([i * a, j * b, k * c])
                copied_molecule.dipole_vector = folded_molecule.dipole_vector
                copied_molecule.O_xyz = np.array(folded_molecule.O_xyz) + shift_vec
                for H_idx in range(2):
                    copied_molecule.H_xyz[3 * H_idx:3 * (H_idx + 1)] = np.array(folded_molecule.H_xyz[3 * H_idx:3 * (H_idx + 1)]) + shift_vec
                if with_Wannier:
                    for M_idx in range(4):
                        copied_molecule.M_xyz[3 * M_idx:3 * (M_idx + 1)] = np.array(folded_molecule.M_xyz[3 * M_idx:3 * (M_idx + 1)]) + shift_vec
                more_molecules.append(copied_molecule)
    
    return more_molecules

def global2local_split(molecules, center_water,a,b,c):
    '''
    Transform global reference frame coordintates to local reference frame defined according to the center water molecule.
    The local reference is differnt from global2local_old.
    The problem is that water at the edge of the box might be splitted. 
    Args:
        molecules: a list of molecules.
        center_water: the center water molecule.
        a,b,c: Lattice constants

    Return: 
        A list of molecules represented in local reference frame (no M included).
    '''
    molecules_local_frame=[]
    local_origin=np.array(center_water.O_xyz)
    # Local orthorgonal base vectors expressed in global reference frame. 
    local_vectors=np.zeros((3,3))
    local_vectors_norms=np.zeros(3)
    local_unit_vectors=np.zeros((3,3))
    local_axes_labels=['x','y','z']
    # Vector pointing from O to an H atom.
    origin2first_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[0:3], a, b, c)
    origin2second_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[3:6], a, b, c)
    # Local x vector, pointing from origin to the midpoint of 2 H atoms. 
    local_vectors[0] = [(origin2first_H[i] + origin2second_H[i]) * 0.5 for i in range(len(origin2first_H))]
    # Local z vector
    local_vectors[2] = np.cross(origin2first_H, origin2second_H)
    # Local y vector. In the plane of H2O and perpendicular to x vector. 
    local_vectors[1] = np.cross(local_vectors[2], local_vectors[0])

    for i in range(len(local_vectors)):
        local_vectors_norms[i]=np.linalg.norm(local_vectors[i])
        if local_vectors_norms[i]==0:
            print('Local base vectors error: {} vector has zero norm'.format(local_axes_labels[i]))
            break

    local_unit_vectors = [local_vectors[i] / np.linalg.norm(local_vectors[i]) for i in range(len(local_vectors))]
    molecules_local_frame=[]
    for molecule_global_frame in molecules:
        local_molecule=molecule_type()
        # Project the vector pointing from local reference origin to the atom, to the corresponding local reference frame axis.
        local_molecule.O_xyz = [projection(pbc_vector(local_origin,molecule_global_frame.O_xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
        local_molecule.H_xyz[0:3] = [projection(pbc_vector(local_origin,molecule_global_frame.H_xyz[0:3],a,b,c),local_unit_vectors[i]) for i in range(3)]
        local_molecule.H_xyz[3:6] = [projection(pbc_vector(local_origin,molecule_global_frame.H_xyz[3:6],a,b,c),local_unit_vectors[i]) for i in range(3)]
        molecules_local_frame.append(local_molecule)
        if molecule_global_frame == center_water:
            local_molecule.dipole_vector = [projection(molecule_global_frame.dipole_vector, local_unit_vectors[i]) for i in range(3)]
    return molecules_local_frame

def global2local(molecules, center_water,a,b,c, with_Wannier = False, with_dipole = False, calculate_center_water_dipole=True):
    '''
    Transform global reference frame coordintates to local reference frame defined according to the center water molecule.
    Try to avoid breaking water molecules at the edge. 
    Use numpy array.
    Args:
        molecules: a list of molecules.
        center_water: the center water molecule.
        a,b,c: Lattice constants

    Return: 
        A list of molecules represented in local reference frame (no M included).
    '''
    local_origin=np.array(center_water.O_xyz)
    # Local orthorgonal base vectors expressed in global reference frame. 
    local_vectors=np.zeros((3,3))
    local_vectors_norms=np.zeros(3)
    local_unit_vectors=np.zeros((3,3))
    local_axes_labels=['x','y','z']
    # Vector pointing from O to an H atom.
    origin2first_H = np.array(pbc_vector(center_water.O_xyz, center_water.H_xyz[0:3], a, b, c))
    origin2second_H = np.array(pbc_vector(center_water.O_xyz, center_water.H_xyz[3:6], a, b, c))
    # Local x vector, pointing from origin to the midpoint of 2 H atoms. 
    local_vectors[0] = (origin2first_H + origin2second_H) * 0.5
    # Local z vector
    local_vectors[2] = np.cross(origin2first_H, origin2second_H)
    # Local y vector. In the plane of H2O and perpendicular to x vector. 
    local_vectors[1] = np.cross(local_vectors[2], local_vectors[0])

    for i in range(len(local_vectors)):
        local_vectors_norms[i]=np.linalg.norm(local_vectors[i])
        if local_vectors_norms[i]==0:
            print('Local base vectors error: {} vector has zero norm'.format(local_axes_labels[i]))
            break

    local_unit_vectors = [local_vectors[i] / np.linalg.norm(local_vectors[i]) for i in range(len(local_vectors))]
    center_water.local_unit_vectors = local_unit_vectors
    molecules_local_frame=[]
    for molecule_global_frame in molecules:
        local_molecule = molecule_type()
        # Vector pointing from O to H1.
        global_OH1 = pbc_vector(molecule_global_frame.O_xyz, molecule_global_frame.H_xyz[0:3], a, b, c)
        global_OH2 = pbc_vector(molecule_global_frame.O_xyz, molecule_global_frame.H_xyz[3:6], a, b, c)
        local_OH1 = np.array([projection(global_OH1, local_unit_vectors[i]) for i in range(3)])
        local_OH2 = np.array([projection(global_OH2, local_unit_vectors[i]) for i in range(3)])
        # Project the vector pointing from local reference origin to the atom, to the corresponding local reference frame axis.
        local_molecule.O_xyz = [projection(pbc_vector(local_origin,molecule_global_frame.O_xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
        local_molecule.H_xyz[0:3] = np.array(local_molecule.O_xyz) + local_OH1
        local_molecule.H_xyz[3:6] = np.array(local_molecule.O_xyz) + local_OH2
        if molecule_global_frame == center_water and calculate_center_water_dipole:
            local_molecule.dipole_vector = [projection(molecule_global_frame.dipole_vector, local_unit_vectors[i]) for i in range(3)]
        molecules_local_frame.append(local_molecule)
        
    if with_Wannier:
        for molecule_idx, molecule_global_frame in enumerate(molecules):
            # Vectors from O to Wannier center in global frame. 
            # Calculate these vectors first to avoid splitting caused by PBC.
            for M_idx in range(4):
                global_OM = pbc_vector(molecule_global_frame.O_xyz, molecule_global_frame.M_xyz[3 * M_idx:3 * (M_idx + 1)], a, b, c)
                local_OM = [projection(global_OM, local_unit_vectors[i]) for i in range(3)]
                molecules_local_frame[molecule_idx].M_xyz[3 * M_idx:3 * (M_idx + 1)] = np.array(molecules_local_frame[molecule_idx].O_xyz) + local_OM
    if with_dipole:
        for molecule_idx, molecule_global_frame in enumerate(molecules):
            molecules_local_frame[molecule_idx].dipole_vector = [projection(molecule_global_frame.dipole_vector, local_unit_vectors[i]) for i in range(3)]
            
    return molecules_local_frame

# def global2local_with_dipole(molecules, center_water,a,b,c):
#     '''
#     Transform global reference frame coordintates to local reference frame defined according to the center water molecule.
#     The local reference is differnt from global2local_old.

#     Args:
#         molecules: a list of molecules.
#         center_water: the center water molecule.
#         a,b,c: Lattice constants

#     Return: 
#         A list of molecules represented in local reference frame (no M included).
#     '''
#     molecules_local_frame=[]
#     local_origin=np.array(center_water.O_xyz)
#     # Local orthorgonal base vectors expressed in global reference frame. 
#     local_vectors=np.zeros((3,3))
#     local_vectors_norms=np.zeros(3)
#     local_unit_vectors=np.zeros((3,3))
#     local_axes_labels=['x','y','z']
#     # Vector pointing from O to an H atom.
#     origin2first_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[0:3], a, b, c)
#     origin2second_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[3:6], a, b, c)
#     # Local x vector, pointing from origin to the midpoint of 2 H atoms. 
#     local_vectors[0] = [(origin2first_H[i] + origin2second_H[i]) * 0.5 for i in range(len(origin2first_H))]
#     # Local z vector
#     local_vectors[2] = np.cross(origin2first_H, origin2second_H)
#     # Local y vector. In the plane of H2O and perpendicular to x vector. 
#     local_vectors[1] = np.cross(local_vectors[2], local_vectors[0])

#     for i in range(len(local_vectors)):
#         local_vectors_norms[i]=np.linalg.norm(local_vectors[i])
#         if local_vectors_norms[i]==0:
#             print('Local base vectors error: {} vector has zero norm'.format(local_axes_labels[i]))
#             break

#     local_unit_vectors = [local_vectors[i] / np.linalg.norm(local_vectors[i]) for i in range(len(local_vectors))]
#     molecules_local_frame=[]
#     for molecule_global_frame in molecules:
#         local_molecule=molecule_type()
#         # Project the vector pointing from local reference origin to the atom, to the corresponding local reference frame axis.
#         local_molecule.O_xyz = [projection(pbc_vector(local_origin,molecule_global_frame.O_xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_molecule.H_xyz[0:3] = [projection(pbc_vector(local_origin,molecule_global_frame.H_xyz[0:3],a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_molecule.H_xyz[3:6] = [projection(pbc_vector(local_origin,molecule_global_frame.H_xyz[3:6],a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_molecule.dipole_vector = [projection(molecule_global_frame.dipole_vector, local_unit_vectors[i]) for i in range(3)]
#         molecules_local_frame.append(local_molecule)
#     return molecules_local_frame

# def global2local_with_dipole_wannier_split(molecules, center_water,a,b,c):
#     '''
#     Transform global reference frame coordintates to local reference frame defined according to the center water molecule.
#     The local reference is differnt from global2local_old.

#     Args:
#         molecules: a list of molecules.
#         center_water: the center water molecule.
#         a,b,c: Lattice constants

#     Return: 
#         A list of molecules represented in local reference frame (no M included).
#     '''
#     molecules_local_frame=[]
#     local_origin=np.array(center_water.O_xyz)
#     # Local orthorgonal base vectors expressed in global reference frame. 
#     local_vectors=np.zeros((3,3))
#     local_vectors_norms=np.zeros(3)
#     local_unit_vectors=np.zeros((3,3))
#     local_axes_labels=['x','y','z']
#     # Vector pointing from O to an H atom.
#     origin2first_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[0:3], a, b, c)
#     origin2second_H = pbc_vector(center_water.O_xyz, center_water.H_xyz[3:6], a, b, c)
#     # Local x vector, pointing from origin to the midpoint of 2 H atoms. 
#     local_vectors[0] = [(origin2first_H[i] + origin2second_H[i]) * 0.5 for i in range(len(origin2first_H))]
#     # Local z vector
#     local_vectors[2] = np.cross(origin2first_H, origin2second_H)
#     # Local y vector. In the plane of H2O and perpendicular to x vector. 
#     local_vectors[1] = np.cross(local_vectors[2], local_vectors[0])

#     for i in range(len(local_vectors)):
#         local_vectors_norms[i]=np.linalg.norm(local_vectors[i])
#         if local_vectors_norms[i]==0:
#             print('Local base vectors error: {} vector has zero norm'.format(local_axes_labels[i]))
#             break

#     local_unit_vectors = [local_vectors[i] / np.linalg.norm(local_vectors[i]) for i in range(len(local_vectors))]
#     molecules_local_frame=[]
#     for molecule_global_frame in molecules:
#         local_molecule=molecule_type()
#         # Project the vector pointing from local reference origin to the atom, to the corresponding local reference frame axis.
#         local_molecule.O_xyz = [projection(pbc_vector(local_origin,molecule_global_frame.O_xyz,a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_molecule.H_xyz[0:3] = [projection(pbc_vector(local_origin,molecule_global_frame.H_xyz[0:3],a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_molecule.H_xyz[3:6] = [projection(pbc_vector(local_origin, molecule_global_frame.H_xyz[3:6], a, b, c), local_unit_vectors[i]) for i in range(3)]
#         for M_idx in range(4):
#             local_molecule.M_xyz[3*M_idx:3*(M_idx+1)]=[projection(pbc_vector(local_origin,molecule_global_frame.M_xyz[3*M_idx:3*(M_idx+1)],a,b,c),local_unit_vectors[i]) for i in range(3)]
#         local_molecule.dipole_vector = [projection(molecule_global_frame.dipole_vector, local_unit_vectors[i]) for i in range(3)]
#         molecules_local_frame.append(local_molecule)
#     return molecules_local_frame

def rough_demo_data(water_list, train_nn_num,samples_num_per_frame,a,b,c,train_x,train_y):
    '''
    train_nn_num is the number of nearest neighbors. 
    train_x and train_y are to be appended by the function.
    No return.
    '''
    molecules_num=len(water_list)
    if samples_num_per_frame<molecules_num:
        samples_indices=random.sample(range(molecules_num),samples_num_per_frame)
    elif samples_num_per_frame==molecules_num:
        samples_indices=np.arange(samples_num_per_frame)
    nn_list=group_nn(train_nn_num,water_list,a,b,c)
    for sample_index in samples_indices:
        center_water=water_list[sample_index]
        center_water.dipole_vector=dipole_moment_vector_D(center_water,a,b,c)
        its_nn=[water_list[i] for i in nn_list[sample_index]]
        for neighbor in its_nn:
            neighbor.dipole_vector = dipole_moment_vector_D(neighbor, a, b, c)            
        its_nn_local_frame=global2local(its_nn,center_water,a,b,c)
        itself_local_frame=global2local([center_water],center_water,a,b,c)[0]
        train_y.append(itself_local_frame.dipole_vector)

        sample_train_param = []
        for nn in its_nn_local_frame:
            sample_train_param.append(nn.O_xyz)
            sample_train_param.append(nn.H_xyz)
        sample_train_param=list(itertools.chain(*sample_train_param))
        # sample_train_param.reshape(-1,1)
        train_x.append(sample_train_param)

def rough_demo_data_another_order(water_list, train_nn_num,samples_num_per_frame,a,b,c,train_x,train_y):
    '''
    The order of data is coordinates of all oxygen, then all hydrogen. 
    train_nn_num is the number of nearest neighbors. 
    train_x and train_y are to be appended by the function.
    No return.
    '''   
    molecules_num=len(water_list)
    if samples_num_per_frame<molecules_num:
        samples_indices=random.sample(range(molecules_num),samples_num_per_frame)
    elif samples_num_per_frame==molecules_num:
        samples_indices=np.arange(samples_num_per_frame)
    nn_list=group_nn(train_nn_num,water_list,a,b,c)
    for sample_index in samples_indices:
        center_water=water_list[sample_index]
        center_water.dipole_vector=dipole_moment_vector_D(center_water,a,b,c)
        its_nn=[water_list[i] for i in nn_list[sample_index]]
        for neighbor in its_nn:
            neighbor.dipole_vector = dipole_moment_vector_D(neighbor, a, b, c)            
        its_nn_local_frame=global2local(its_nn,center_water,a,b,c)
        itself_local_frame=global2local([center_water],center_water,a,b,c)[0]
        train_y.append(itself_local_frame.dipole_vector)

        sample_train_param=[]
        for nn in its_nn_local_frame:
            sample_train_param.append(nn.O_xyz)
        for nn in its_nn_local_frame:
            sample_train_param.append(nn.H_xyz)
        sample_train_param=list(itertools.chain(*sample_train_param))
        # sample_train_param.reshape(-1,1)
        train_x.append(sample_train_param)

# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ConstantMean(), num_tasks=3
#         )
#         self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def test_pbc_vector(water_list, a, b, c, tolerance = 1e-12):
    '''
    Test whether two pbc_vector work in the same way. 
    '''
    for (waters_1, waters_2) in itertools.combinations(water_list, 2):
        new_output = pbc_vector(waters_1.O_xyz, waters_2.O_xyz, a, b, c)
        old_output = pbc_vector(waters_1.O_xyz, waters_2.O_xyz, a, b, c)
        for i in range(3):
            if abs(new_output[i] - old_output[i]) > tolerance:
                print('ERROR: DIFFERENCE')
                print('new_output:')
                print(new_output)
                print('old_output:')
                print(old_output)
                break

def test_pbc_distance(water_list, a, b, c, tolerance=1e-12):
    '''
    Test whether two distance work in the same way.
    '''
    for (waters_1, waters_2) in itertools.combinations(water_list, 2):
        new_output = distance(waters_1.O_xyz, waters_2.O_xyz, a, b, c)
        old_output = distance_old(waters_1.O_xyz, waters_2.O_xyz, a, b, c)
        if abs(new_output - old_output) > tolerance:
            print('ERROR: DIFFERENCE')
            print('new_output:')
            print(new_output)
            print('old_output:')
            print(old_output)
            break

def process_molecules(data_x, data_y, train_nn_water, frame_samples_num, sample_frame_separation, input_file_name='../Data/H2O-mlwfc.gro', lines_per_frame=451):
    '''
    Extract molecules information from the input file and generate data for training and testing.

    Args:
        data_x, data_y: 2-D lists to store the x data. They are appended by the function. 
        train_nn_water: number of neighbors water molecules for each data point. 
        frame_samples_num: data points number per frame. 
        sample_frame_separation: take 1 frame per ~ frames. 
        input_file_name: the name of the input trajectory file.
        lines_per_frame: the line number corresponding a frame in the trajectory file. 
    No return. 
    '''
    existing_data_size = 0
    frame_idx = 0
    for line_idx, line in enumerate(open(input_file_name, 'r')):
        if line_idx % (lines_per_frame*sample_frame_separation) >= 0 and line_idx % (lines_per_frame*sample_frame_separation) < lines_per_frame:
            Lwords = line.split()
            if Lwords[0] == 'MD,':
                O_list = []
                H_list = []
                M_list = []
                Num_tot = 0
            elif len(Lwords) == 1:
                Num_tot = int(line)
            elif len(Lwords) > 3:
                if Lwords[1] == 'O':
                    atom = atom_type()
                    atom.xyz = Lwords[3:6]
                    atom.xyz = [float(i) for i in atom.xyz]
                    O_list.append(atom)
                elif Lwords[1] == 'H':
                    atom = atom_type()
                    atom.xyz = Lwords[3:6]
                    atom.xyz = [float(i) for i in atom.xyz]
                    H_list.append(atom)
                elif Lwords[1] == 'M':
                    atom = atom_type()
                    atom.xyz = Lwords[3:6]
                    atom.xyz = [float(i) for i in atom.xyz]
                    M_list.append(atom)
                else:
                    sys.exit("Error in reading!")
            # The last line of a frame. 
            elif len(Lwords) == 3:
                frame_idx += 1
                a = float(Lwords[0])
                b = float(Lwords[1])
                c = float(Lwords[2])
                
                water_list = assemble(O_list, H_list, M_list, a, b, c)
                good_frame = True
                for water_i in water_list:
                    if water_i.test_complete() != True:
                        print("Error: water molecules are not completed!\n")
                        print(water_i.H_xyz)
                        print(water_i.M_xyz)
                        good_frame = False
                        break
                if good_frame:
                    if existing_data_size<data_set_size:
                        # rough_demo_data(water_list, train_nn_water,frame_samples_num,a,b,c,data_x,data_y)
                        test_pbc_vector(water_list, a, b,c)
                        existing_data_size += frame_samples_num
                    else:
                        break
                else:
                    continue
            else:
                continue 
    
# main() here:
if __name__ == "__main__":
    # The number of nearest neighbors water molecules
    train_nn_water = 4
    # Use all molecules in a frame. 
    frame_samples_num = 64
    # Data should not come from consecutive frame to obtain samples from a larger phase space.
    sample_frame_separation = 100
    train_set_size = 12000
    test_set_size = 12000
    data_set_size = train_set_size + test_set_size

    output_file_name = 'data{}-{}.dat'.format(data_set_size,datetime.now().date())
    log_file_name = 'data{}-{}.log'.format(data_set_size,datetime.now().date())
    output = open(output_file_name, 'a')
    log = open(log_file_name,'a')
    log.write('train_nn_water: {}\ntrain_set_size:{}\ntest_set_size:{}\nsample_frame_separation:{}\n'.format(train_nn_water,train_set_size,train_set_size,sample_frame_separation))

    input_file_name = '../Data/H2O-mlwfc.gro'

    # data set is training set and test set
    data_x = []
    data_y = []
    # existing_data_size = 0
    # frame_idx = 0
    # previous_sample_line_idx = 0

    process_molecules(data_x, data_y, train_nn_water, frame_samples_num, sample_frame_separation)

    if len(data_x) >= data_set_size:
        train_x=data_x[:train_set_size]
        train_y=data_y[:train_set_size]
        test_x=data_x[train_set_size:train_set_size+data_set_size]
        test_y=data_y[train_set_size:train_set_size+data_set_size]

        # output.write('# Data set size {} \n'.format(train_set_size))
        print('Start writing data')
        for i in range(len(data_x)):
            output.write('input ')
            for param in data_x[i]:
                output.write('{} '.format(param))
            
            output.write('output ')
            for dipole_component in data_y[i]:
                output.write('{} '.format(dipole_component))
            output.write('\n')
        print('End writing data')
    else:
        print(f'Warning: availabel data size {len(data_x)} < {data_set_size}')

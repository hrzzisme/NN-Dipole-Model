# Convert the .gro file to training set including H,O relative positions (as mentioned by Weinan E) and corresponding dipole moments.
import string
import math
import sys
import os
import glob
import random
import itertools
from datetime import datetime, timedelta
import numpy as np
# import torch
# import gpytorch
from matplotlib import pyplot as plt
from gro2data import atom_type, molecule_type, distance, projection, pbc_vector, assemble, dipole_moment_vector_D, group_nn, global2local, copy_enlarge_box
e_charge = 1.60217653E-19
c_0 = 299792458
enm2D = (1.0E-9)*e_charge/(1.0E-21/c_0)

full_info_O_num = 16
full_info_H_num = 32
# max_neighbor_num = 21
max_neighbor_O = 32 # it should be determined by your own situation
max_neighbor_H = 66 # it should be determined by your own situation
# 1000K 1GPa 33-1 62
# 1000K 10GPa 33-1 66
#rc-0.5 global_max_O_num:37, global_max_H_num:67
params_num = (full_info_O_num + full_info_H_num) * 4 + max_neighbor_O + max_neighbor_H - full_info_O_num - full_info_H_num
radius_cutoff = 0.6 #nm



def single_atom_relative_matrix_different_exp_data(water_list, train_nn_num, samples_num_per_frame, a, b, c, train_x, train_y):
    '''
    train_nn_num is the number of nearest neighbors.
    train_x and train_y are to be appended by the function.
    Only relative vector for a single atom is appended.
    {1/R^2, x/R^3, y/R^3, z/R^3}
    No return.
    '''
    molecules_num = len(water_list)
    if samples_num_per_frame < molecules_num:
        samples_indices = random.sample(
            range(molecules_num), samples_num_per_frame)
    elif samples_num_per_frame == molecules_num:
        samples_indices = np.arange(samples_num_per_frame)
    nn_list = group_nn(train_nn_num, water_list, a, b, c)
    for sample_index in samples_indices:
        center_water = water_list[sample_index]
        # center_water.dipole_vector = dipole_moment_vector_D(
        #     center_water, a, b, c)
        its_nn = [water_list[i] for i in nn_list[sample_index]]
        its_nn_local_frame = global2local(its_nn, center_water, a, b, c)
        itself_local_frame = global2local(
            [center_water], center_water, a, b, c,with_dipole=True)[0]
        itself_and_nn_local_frame = [itself_local_frame] + its_nn_local_frame

        atoms_local_frame = []
        atoms_local_frame_O, atoms_local_frame_H = [], []

        for local_molecule in itself_and_nn_local_frame:
            if len(local_molecule.O_xyz) != 3:
                sys.exit("local_molecule.O_xyz) != 3")
            if len(local_molecule.H_xyz[0:3]) != 3:
                sys.exit("local_molecule.H_xyz[0:3] != 3")
            if len(local_molecule.H_xyz[3:6]) != 3:
                sys.exit("local_molecule.H_xyz[3:6] != 3")
            # print(local_molecule.O_xyz)
            # exit(0)
            atoms_local_frame_O.append(local_molecule.O_xyz)
            atoms_local_frame_H.append(local_molecule.H_xyz[0:3])
            atoms_local_frame_H.append(local_molecule.H_xyz[3:6])

        # print(atoms_local_frame_H)
        # print(np.array(atoms_local_frame_H).shape)
        local_frame_O_distances = np.linalg.norm(atoms_local_frame_O, axis=1)
        local_frame_H_distances = np.linalg.norm(atoms_local_frame_H, axis=1)
        local_frame_O_indices = np.argsort(local_frame_O_distances)
        local_frame_H_indices = np.argsort(local_frame_H_distances)

        # local_frame_O_distances = local_frame_O_distances[local_frame_O_indices]
        # local_frame_H_distances = local_frame_H_distances[local_frame_H_indices]

        atoms_local_frame_O = np.array(atoms_local_frame_O)[local_frame_O_indices]
        atoms_local_frame_H = np.array(atoms_local_frame_H)[local_frame_H_indices]
        sample_train_param = []
        i = 0
        atom_i = atoms_local_frame_O[i]
        O_count, H_count = 0, 0
        full_O_count, full_H_count = 0, 0
        # Exclude O of center molecule. Include H of center molecule.
        for index, O_coords in enumerate(atoms_local_frame_O[i+1:]):
            d_ij = distance(atom_i, O_coords, a, b, c)
            if d_ij < radius_cutoff and O_count < max_neighbor_O:
                sample_train_param.append(1.0/d_ij)
                if O_count < full_info_O_num:
                    relative_pos_over_squared_dist = [(O_coords[direction]) / d_ij ** 2 for direction in range(3)]
                    sample_train_param = sample_train_param + relative_pos_over_squared_dist
                    full_O_count += 1
                O_count += 1
            else:
                break

        if O_count < full_info_O_num:
            for i in range(full_info_O_num - O_count):
                [sample_train_param.append(0.0) for j in range(4)]
            O_count = full_info_O_num

        if O_count < max_neighbor_O:
            for i in range(max_neighbor_O - O_count):
                sample_train_param.append(0.0)

        for index, H_coords in enumerate(atoms_local_frame_H):
            d_ij = distance(atom_i, H_coords, a, b, c)
            if d_ij <= radius_cutoff and H_count < max_neighbor_H:
                sample_train_param.append(1. / d_ij)
                if H_count < full_info_H_num:
                    relative_pos_over_squared_dist = [(H_coords[direction]) / d_ij ** 2 for direction in range(3)]
                    sample_train_param = sample_train_param + relative_pos_over_squared_dist
                    full_H_count += 1

                H_count += 1

                # if full_info_C_num <= C_count < max_neigh_C:
            else:
                break
        # print("H_count: ",H_count, full_H_count)
        if H_count < full_info_H_num:
            for i in range(full_info_H_num - H_count):
                [sample_train_param.append(0.0) for j in range(4)]
            H_count = full_info_H_num
        # print(len(sample_train_param))
        if H_count > max_neighbor_H:
            sample_train_param = sample_train_param[0: params_num]

        if H_count < max_neighbor_H:
            for i in range(max_neighbor_H - H_count):
                sample_train_param.append(0.0)
        # print(len(sample_train_param), params_num)
        if len(sample_train_param) != params_num:
            with open("error.log", "a+") as f:
                f.write(
                    "snap_count: {}, len: {}, params_num:{}\n".format(snap_count, len(sample_train_param), params_num))
            print("snap_count: {}, len: {}, params_num:{}".format(snap_count, len(sample_train_param), params_num))
        assert len(sample_train_param) == params_num
        train_x.append(sample_train_param)

        # print(train_x)
        # exit(0)
        target = list(itself_local_frame.dipole_vector)
        rotation_matrix = np.array(center_water.local_unit_vectors).flatten()
        assert(len(rotation_matrix) == 9)
        target += list(rotation_matrix)
        train_y.append(target)



def write_data(output_file_name, data_x, data_y):
    '''
    Write data to file.
    '''
    with open(output_file_name, 'a') as output:
        for i in range(len(data_x)):
            output.write('input ')
            for param in data_x[i]:
                output.write(f'{param} ')

            output.write('output ')
            for dipole_component in data_y[i]:
                output.write(f'{dipole_component} ')
            output.write('\n')

def gro_lines_num_per_frame(gro_file_name):
    '''
    Get the number of lines in each frame.
    '''
    with open(gro_file_name, 'r') as gro_file:
        for line_idx, line in enumerate(gro_file):
            # The line specifying atoms number.
            if (line_idx == 1) and len(line.split()) == 1:
                return int(line) + 3

# main() here:
if __name__ == "__main__":
    working_dir = '/media/hourui/Backup/hpc2backup/data/maching-learning/2water'
    os.chdir(working_dir)
    # The number of nearest neighbors water molecules
    train_nn_water = 60
    # Copy molecules in 1 box into n*n*n boxes such that the unit cell becomes na*nb*nc
    n = 1
    frame_samples_num = 128
    # Data should not come from consecutive frame to obtain samples from a larger phase space.
    sample_frame_separation = 1
    total_snap_in_files = {
        "1000K-1GPa": 1223,
        "1000K-5GPa": 3383,
        "1000K-10GPa": 4018,
        "2000K-5GPa": 1095,
        "2000K-10GPa": 1608,
        "2000K-30GPa": 1094,
    }# number of snapshots in the trajectory
    # TP_strs = ["1000K-1GPa", "1000K-5GPa", "1000K-10GPa", "2000K-5GPa", "2000K-10GPa"]
    TP_strs = ["1000K-1GPa", "1000K-5GPa", "1000K-10GPa", "2000K-5GPa", "2000K-10GPa",'2000K-30GPa']
    for TP_str in TP_strs:
        print(TP_str)
        with open("error-molecules.txt", "w") as f:
            f.write(TP_str + "\n")
        # TP_str = '2000K-10GPa'
        input_file_name = glob.glob(f'/media/hourui/Backup/hpc2backup/data/maching-learning/water/HP-HT-128water/{TP_str}/*gro')[0]
        # Each frame occupies this number of lines in the gro file.
        lines_num_per_frame = gro_lines_num_per_frame(input_file_name)
        # The index of the first frame to generate data.
        first_data_frame = 0
        # Write buffer to file.
        buffer_size = 12800*5
        set_range = False
        encoding = 'null'
        NORMALIZATION = True

        if set_range:
            ranges = [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]
            ranges_samples_num = [400, 1600, 400]
            ranges_remaining_samples_num = ranges_samples_num.copy()
            data_set_size = np.sum(ranges_remaining_samples_num)
        else:
            data_set_size = total_snap_in_files[TP_str]*128 # simulation box contains 128 molecules
        output_file_name = f'{TP_str}.dat'

        data_function = single_atom_relative_matrix_different_exp_data
        # data set is training set and test set
        data_x = []
        data_y = []
        existing_data_size = 0
        previous_sample_line_idx = 0

        for line_idx, line in enumerate(open(input_file_name, 'r')):
            frame_idx = line_idx // lines_num_per_frame
            if frame_idx >= first_data_frame and (frame_idx - first_data_frame) % sample_frame_separation == 0:
                # if line_idx % (lines_num_per_frame * sample_frame_separation) >= 0 and line_idx % (lines_num_per_frame * sample_frame_separation) < 451:
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
                    a = float(Lwords[0])
                    b = float(Lwords[1])
                    c = float(Lwords[2])

                    # print("O, H, M:", len(O_list), len(H_list), len(M_list))

                    water_list = assemble(O_list, H_list, M_list, a, b, c)
                    # print("# of Water molecules:", len(water_list))
                    good_frame = True
                    for water_i in water_list:
                        if water_i.test_complete() != True:
                            print("Error: water molecules are not completed!\n")
                            with open("error-molecules.txt", "w") as f:
                                f.write("Error: water molecules are not completed!\n")
                            print(water_i.H_xyz)
                            print(water_i.M_xyz)
                            good_frame = False
                            break
                    if good_frame:
                        if existing_data_size < data_set_size:
                            #enlarged_molecules = copy_enlarge_box(water_list, a, b, c, n)
                            data_function(enlarged_molecules, train_nn_water, frame_samples_num,n*a,n*b,n*c,data_x,data_y)
                            existing_data_size += frame_samples_num
                            if existing_data_size >= buffer_size:
                                print("write")
                                write_data(output_file_name, data_x, data_y)
                                data_x = data_x[buffer_size:]
                                data_y = data_y[buffer_size:]
                        else:
                            break
                    else:
                        continue
                else:
                    continue


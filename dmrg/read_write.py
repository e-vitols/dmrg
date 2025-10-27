import numpy as np
import h5py

def read_h5_to_numpy(file):
    f = h5py.File(file, 'r')
    return f

def write_to_h5(file, input_data):
    f = h5py.File(file, "w")
    f.create_dataset('data', data=input_data)
    f.close()
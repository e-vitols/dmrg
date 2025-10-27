import numpy as np
import h5py

def load_h5_to_array(file):
    """
    Load a numpy array to hdf5 file
    """
    with h5py.File(file, 'r') as f:
        return f['array_data'][:]

def write_array_to_h5(file, input_data):
    """
    Save a numpy array to hdf5 file
    """
    with h5py.File(file, 'w') as f:
        f.create_dataset('array_data', data=input_data)

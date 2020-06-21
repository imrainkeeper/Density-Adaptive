import h5py
import torch
import shutil
import os
import numpy as np

def save_checkpoint(state, checkpoint_save_dir, dataset, epoch, test_mae, test_mse, val_mae, val_mse):
    checkpoint_filepath = os.path.join(checkpoint_save_dir, 'Checkpoint_Dataset_' + dataset + str(epoch) + '_' + str(test_mae)  + '_' + str(test_mse) + '_' + str(val_mae)  + '_' + str(val_mse) + '.pth.tar')
    torch.save(state, checkpoint_filepath)

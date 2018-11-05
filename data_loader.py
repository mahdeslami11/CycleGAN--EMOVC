from torch.utils import data
import torch
import os
import random
import glob
from os.path import join, basename, dirname, split
import numpy as np

min_length = 256

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
    

class MyDataset(data.Dataset):
    
    def __init__(self, data_dir, target_speaker, source_emotion, target_emotion):
        mc_files_A = glob.glob(join(data_dir, f'{target_speaker}-{source_emotion}*.npy'))
        mc_files_B = glob.glob(join(data_dir, f'{target_speaker}-{target_emotion}*.npy'))
        self.mc_files_A = self.rm_too_short_utt(mc_files_A)
        self.mc_files_B = self.rm_too_short_utt(mc_files_B)
        self.num_files = min(len(self.mc_files_A), len(self.mc_files_B))
        print("\t Number of training samples: ", self.num_files)

    def rm_too_short_utt(self, mc_files, min_length=min_length):
        new_mc_files = []
        for mcfile in mc_files:
            mc = np.load(mcfile)
            if mc.shape[0] > min_length:
                new_mc_files.append(mcfile)
        return new_mc_files

    def sample_seg(self, feat, sample_len=min_length):
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename_A = self.mc_files_A[index]
        filename_B = self.mc_files_B[index]
        mc_A = np.load(filename_A)
        mc_B = np.load(filename_B)
        lf0_le_A = np.load(filename_A.replace('mc', 'lf0_le'))
        lf0_le_B = np.load(filename_B.replace('mc', 'lf0_le'))
        mc_lf0_le_A = np.concatenate((mc_A, lf0_le_A), -1)
        mc_lf0_le_A = self.sample_seg(mc_lf0_le_A)
        mc_lf0_le_A = np.transpose(mc_lf0_le_A, (1, 0))  # (T, D) -> (D, T)

        mc_lf0_le_B = np.concatenate((mc_B, lf0_le_B), -1)
        mc_lf0_le_B = self.sample_seg(mc_lf0_le_B)
        mc_lf0_le_B = np.transpose(mc_lf0_le_B, (1, 0))  # (T, D) -> (D, T)

        return torch.FloatTensor(mc_lf0_le_A), torch.FloatTensor(mc_lf0_le_B) 


class TestDataset(object):
    """Dataset for validation."""
    def __init__(self, data_dir, src_wav_dir, trg_spk, src_emo, trg_emo):
        self.trg_spk = trg_spk
        self.trg_emo = trg_emo
        self.src_emo = src_emo

        self.src_spk_emo = '{}-{}'.format(trg_spk, src_emo)
        self.trg_spk_emo = '{}-{}'.format(trg_spk, trg_emo)
        self.mc_files = sorted(glob.glob(join(data_dir, '{}*.npy'.format(self.src_spk_emo))))

        self.src_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(self.src_spk_emo)))
        
        self.trg_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(self.trg_spk_emo)))

        self.src_spk_stats_lf0_le = np.load(join(data_dir.replace('test', 'train').replace('mc', 'lf0_le'), '{}_stats.npz'.format(self.src_spk_emo)))        
        self.trg_spk_stats_lf0_le = np.load(join(data_dir.replace('test', 'train').replace('mc', 'lf0_le'), '{}_stats.npz'.format(self.trg_spk_emo)))        
        self.lf0_mean_src = self.src_spk_stats_lf0_le['log_f0s_mean']
        self.lf0_std_src = self.src_spk_stats_lf0_le['log_f0s_std']
        self.lf0_mean_trg = self.trg_spk_stats_lf0_le['log_f0s_mean']
        self.lf0_std_trg = self.trg_spk_stats_lf0_le['log_f0s_std']

        self.le_mean_src = self.src_spk_stats_lf0_le['log_energy_mean']
        self.le_std_src = self.src_spk_stats_lf0_le['log_energy_std']
        self.le_mean_trg = self.trg_spk_stats_lf0_le['log_energy_mean']
        self.le_std_trg = self.trg_spk_stats_lf0_le['log_energy_std']
        
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        
        self.src_wav_dir = src_wav_dir

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = glob.glob(join(f"{self.src_wav_dir}/*/{self.trg_spk}/{self.src_emo}", filename.replace('npy', 'wav')))[0]
            batch_data.append(wavfile_path)
        return batch_data       

def get_loader(data_dir, target_speaker, source_emotion, target_emotion, batch_size=32, mode='train', num_workers=1):
    dataset = MyDataset(data_dir, target_speaker, source_emotion, target_emotion)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader









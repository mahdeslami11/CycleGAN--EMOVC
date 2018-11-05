import argparse
from model import Generator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
import glob
from tqdm import tqdm
# from data_loader import TestDataset, TestDataset2

emotions = ['sad', 'normal', 'angry']
emo2idx = dict(zip(emotions, range(len(emotions))))
target_speaker = 'liuchang'

def low_pass_filter(x, fs=int(1.0 / (5.0 * 0.001)), cutoff=20, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER
    int(1.0 / (frame_period * 0.001))
    Args:
        x (ndarray): sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, src_wav_dir, trg_spk, src_emo, trg_emo):
        # data_dir: */mc/test
        # src_wav_dir: */liuchang_wavs_trimmed
        self.trg_spk = trg_spk
        self.trg_emo = trg_emo
        self.src_emo = src_emo

        self.src_spk_emo = '{}-{}'.format(trg_spk, src_emo)  # e.g., liuchang-normal
        self.trg_spk_emo = '{}-{}'.format(trg_spk, trg_emo)  # e.g., liuchang-angry
        self.mc_files = sorted(glob.glob(join(data_dir, '{}*.npy'.format(self.src_spk_emo))))

        # load means and stds, which stored in the */mc/train dir
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
        # self.emo_idx = emo2idx[trg_emo]
        # emo_cat = to_categorical([self.emo_idx], num_classes=len(emotions))
        # self.emo_c_trg = emo_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = glob.glob(join(f"{self.src_wav_dir}/*/{self.trg_spk}/{self.src_emo}", filename.replace('npy', 'wav')))[0]
            # wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data  

    def get_test_data(self):
        data_list = []
        for mcfile in self.mc_files:
            filename = basename(mcfile).split('-')[-1]
            wav_id = filename.split('.')[0]
            if int(wav_id) < 200: continue
            print(filename)
            wavfile_path = glob.glob(join(f"{self.src_wav_dir}/{self.src_emo}", filename.replace('npy', 'wav')))[0]
            data_list.append(wavfile_path)
        return data_list


def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple = 4)  # TODO
    # return wav


def test(config):
    current_dir = join(config.convert_dir, str(config.resume_iters))
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period=16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G_A2B = Generator().to(device)
    # G_B2A = Generator().to(device)

    test_loader = TestDataset(config.test_data_dir, 
                              config.src_wav_dir, 
                              config.target_speaker, 
                              config.source_emotion, 
                              config.target_emotion)
    # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...')
    G_A2B_path = join(config.model_save_dir, f'{config.resume_iters}-G_A2B.ckpt')
    # G_B2A_path = join(config.model_save_dir, f'{config.resume_iters}-G_B2A.ckpt')
    G_A2B.load_state_dict(torch.load(G_A2B_path, map_location=lambda storage, loc: storage))
    # G_B2A.load_state_dict(torch.load(G_B2A_path, map_location=lambda storage, loc: storage))

    # Read a batch of testdata
    # test_wavfiles = test_loader.get_batch_test_data(batch_size=8)
    # test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]

    # Read testdata
    test_wavfiles = test_loader.get_test_data()
    # test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]
    print(f"Get {len(test_wavfiles)} test wav files!")

    with torch.no_grad():
        for wav_file_src in tqdm(test_wavfiles):
            # print(len(wav))
            wav_name = basename(wav_file_src)  # 'id.wav'
            wav_id = wav_name.split('.')[0]
            # print(wav_name)

            # ==== target wav ===== #
            wav_file_trg = wav_file_src.replace(test_loader.src_emo, test_loader.trg_emo)
            wav_trg = load_wav(wav_file_trg, sampling_rate)
            f0_trg, _, sp_trg, _ = world_decompose(wav=wav_trg, fs=sampling_rate, frame_period=frame_period)
            _, cont_lf0_lpf_trg = get_cont_lf0(f0_trg)
            coded_sp_trg = world_encode_spectral_envelop(sp=sp_trg, fs=sampling_rate, dim=num_mcep)
            trg_mc_filename = join(current_dir, f"{test_loader.trg_emo}-mc-{wav_id}.npy")
            trg_lf0_filename = join(current_dir, f"{test_loader.trg_emo}-lf0-{wav_id}.npy")
            np.save(file=trg_mc_filename, arr=coded_sp_trg, allow_pickle=False, fix_imports=True)
            np.save(file=trg_lf0_filename, arr=cont_lf0_lpf_trg, allow_pickle=False, fix_imports=True)
            # ===================== #

            # ==== src wav and then convert ===== #
            wav_src = load_wav(wav_file_src, sampling_rate)
            f0, timeaxis, sp, ap = world_decompose(wav=wav_src, fs=sampling_rate, frame_period=frame_period)
            # f0_converted = pitch_conversion(f0=f0, 
            #     mean_log_src=test_loader.logf0s_mean_src, std_log_src=test_loader.logf0s_std_src, 
            #     mean_log_target=test_loader.logf0s_mean_trg, std_log_target=test_loader.logf0s_std_trg)
            uv, cont_lf0_lpf = get_cont_lf0(f0)
            le = get_log_energy(sp)
            lf0_normed = (cont_lf0_lpf - test_loader.lf0_mean_src) / test_loader.lf0_std_src
            le_normed = (le - test_loader.le_mean_src) / test_loader.le_std_src
            lf0_le_cwt = get_lf0_le_cwt(lf0_normed, le_normed)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)

            # print("Before being fed into G: ", coded_sp.shape)
            coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
            mc_lf0_le = np.concatenate((coded_sp_norm, lf0_le_cwt), -1)
            mc_lf0_le_tensor = torch.FloatTensor(mc_lf0_le.T).unsqueeze_(0).unsqueeze_(1).to(device)

            # spk_emo_conds = torch.FloatTensor(test_loader.emo_c_trg).to(device)
            # print(spk_emo_conds.size())
            mc_lf0_le_converted_norm = G_A2B(mc_lf0_le_tensor).data.cpu().numpy()
            mc_lf0_le_converted_norm = np.squeeze(mc_lf0_le_converted_norm).T
            coded_sp_converted = mc_lf0_le_converted_norm[:, :36] * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            lf0_converted = inverse_cwt(mc_lf0_le_converted_norm[:, 36:46].T) * test_loader.lf0_std_trg + test_loader.lf0_mean_trg
            le_converted = inverse_cwt(mc_lf0_le_converted_norm[:, 46:].T) * test_loader.le_std_trg + test_loader.le_mean_trg
            # ------------- low pass filter -------------
            le_converted = low_pass_filter(le_converted)
            # -------------------------------------------
            e_converted = np.exp(le_converted)
            f0_converted = np.squeeze(uv) * np.exp(lf0_converted)
            decoded_sp_converted = world_decode_spectral_envelop(coded_sp_converted, sampling_rate)
            e_sp_converted = np.linalg.norm(decoded_sp_converted, ord=2, axis=-1)
            e_ratio = np.divide(e_converted, e_sp_converted)
            # decoded_sp_converted = decoded_sp_converted * e_ratio[:, None]
            wav_transformed = world_speech_synthesis(f0=f0_converted, sp=decoded_sp_converted, 
                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
            wav_id = wav_name.split('.')[0]
            librosa.output.write_wav(join(current_dir,
                f'{test_loader.src_spk_emo}-vcto-{test_loader.trg_emo}-{wav_id}.wav'), wav_transformed, sampling_rate)

            converted_mc_filename = join(current_dir, f"converted-{test_loader.trg_emo}-mc-{wav_id}.npy")
            converted_lf0_filename = join(current_dir, f"converted-{test_loader.trg_emo}-lf0-{wav_id}.npy")
            np.save(file=converted_mc_filename, arr=coded_sp_converted, allow_pickle=False, fix_imports=True)
            np.save(file=converted_lf0_filename, arr=lf0_converted, allow_pickle=False, fix_imports=True)
            if [True, False][0]:
                wav_cpsyn = world_speech_synthesis(f0=f0, sp=sp, 
                                                ap=ap, fs=sampling_rate, frame_period=frame_period)
                librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters), f'{test_loader.src_spk_emo}-cpsyn-{wav_name}'), wav_cpsyn, sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--target_speaker', type=str, default='liuchang')
    parser.add_argument('--source_emotion', type=str, default='normal')
    parser.add_argument('--target_emotion', type=str, default='sad')
    parser.add_argument('--resume_iters', type=int, default=18000, help='resume training from this step')

    # Directories.
    parser.add_argument('--train_data_dir', type=str, default='/scratch/sxliu/data_exp/CASIA_dataset/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='/scratch/sxliu/data_exp/CASIA_dataset/mc/test')
    parser.add_argument('--src_wav_dir', type=str, default='/scratch/sxliu/data_exp/CASIA_dataset/liuchang_wavs_trimmed')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./logs/1026-0749-28-2018-mc-lf0-le-liuchang-normal-sad/models')
    parser.add_argument('--convert_dir', type=str, default='./logs/1026-0749-28-2018-mc-lf0-le-liuchang-normal-sad/converted')


    config = parser.parse_args()
    print(config)
    test(config)

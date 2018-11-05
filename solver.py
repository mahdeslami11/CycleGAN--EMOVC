from model import Generator, Discriminator, Classifier
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
from tqdm import tqdm
from itertools import chain


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config, log):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # Model configurations.
        self.num_emotions = config.num_emotions
        self.lambda_stl = config.lambda_stl
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp  # may not be used here!

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.c_lr = config.c_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.gan_curriculum = config.gan_curriculum
        self.starting_rate = config.starting_rate # 0.01
        self.default_rate = config.default_rate   # 0.5

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.log = log
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        
        # Build the model and tensorboard.
        self.set_loss_criterion()
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

    def set_loss_criterion(self):
        self.recon_criterion = nn.L1Loss()
        self.gan_criterion = nn.BCELoss()
        self.stl_criterion = nn.BCELoss()


    def build_model(self):
        """Create generators, discriminators and a classifier."""
        self.G_A2B = Generator()
        self.G_B2A = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()
        self.C = Classifier()
        
        gen_params = chain(self.G_A2B.parameters(), self.G_B2A.parameters())
        dis_params = chain(self.D_A.parameters(), self.D_B.parameters())
        clf_params = self.C.parameters()

        self.optim_gen = torch.optim.Adam( gen_params, lr=self.g_lr, betas=(self.beta1, self.beta2), weight_decay=0.00001)
        self.optim_dis = torch.optim.Adam( dis_params, lr=self.d_lr, betas=(self.beta1, self.beta2), weight_decay=0.00001)
        self.optim_clf = torch.optim.Adam( clf_params, lr=self.c_lr, betas=(self.beta1, self.beta2), weight_decay=0.00001)

        self.print_network(self.G_A2B, 'G_A2B')
        self.print_network(self.D_A, 'D_A')
        self.print_network(self.G_B2A, 'G_B2A')
        self.print_network(self.D_B, 'D_B')
        self.print_network(self.C, 'C')
            
        self.G_A2B.to(self.device)
        self.G_B2A.to(self.device)
        self.D_A.to(self.device)
        self.D_B.to(self.device)
        self.C.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.log(model)
        self.log(name)
        self.log("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_A2B_path = os.path.join(self.model_save_dir, '{}-G_A2B.ckpt'.format(resume_iters))
        D_A_path = os.path.join(self.model_save_dir, '{}-D_A.ckpt'.format(resume_iters))
        G_B2A_path = os.path.join(self.model_save_dir, '{}-G_B2A.ckpt'.format(resume_iters))
        D_B_path = os.path.join(self.model_save_dir, '{}-D_B.ckpt'.format(resume_iters))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_iters))
        self.G_A2B.load_state_dict(torch.load(G_A2B_path, map_location=lambda storage, loc: storage))
        self.D_A.load_state_dict(torch.load(D_A_path, map_location=lambda storage, loc: storage))
        self.G_B2A.load_state_dict(torch.load(G_B2A_path, map_location=lambda storage, loc: storage))
        self.D_B.load_state_dict(torch.load(D_B_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, c_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optim_gen.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.optim_dis.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.optim_clf.param_groups:
            param_group['lr'] = c_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optim_gen.zero_grad()
        self.optim_dis.zero_grad()
        self.optim_clf.zero_grad()
    
    def l1_loss(self, x, x_hat):
        return torch.mean(torch.abs(x - x_hat))

    def l2_loss(self, x, x_hat):
        """LS GAN framework"""
        loss_criterion = torch.nn.MSELoss()
        return loss_criterion(x, x_hat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def get_gan_loss(self, dis_real, dis_fake1, dis_fake2, criterion):
        labels_dis_real = torch.ones_like(dis_real).to(self.device)
        labels_dis_fake1 = torch.zeros_like(dis_fake1).to(self.device)
        labels_dis_fake2 = torch.zeros_like(dis_fake2).to(self.device)
        labels_gen1 = torch.ones_like(dis_fake1).to(self.device)
        labels_gen2 = torch.ones_like(dis_fake2).to(self.device)

        dis_loss = criterion( dis_real, labels_dis_real ) * 0.4 + criterion( dis_fake1, labels_dis_fake1 ) * 0.3 + criterion( dis_fake2, labels_dis_fake2 ) * 0.3
        gen_loss = criterion( dis_fake1, labels_gen1 ) * 0.5 + criterion( dis_fake2, labels_gen2) * 0.5

        return dis_loss, gen_loss


    def get_stl_loss(self, A_stl, A1_stl, A2_stl, B_stl, B1_stl, B2_stl, criterion):
        labels_A = torch.ones_like(A_stl, dtype=torch.float).to(self.device)
        labels_A1 = torch.ones_like(A1_stl, dtype=torch.float).to(self.device)
        labels_A2 = torch.ones_like(A2_stl, dtype=torch.float).to(self.device)
        labels_B = torch.zeros_like(B_stl, dtype=torch.float).to(self.device)
        labels_B1 = torch.zeros_like(B1_stl, dtype=torch.float).to(self.device)
        labels_B2 = torch.zeros_like(B2_stl, dtype=torch.float).to(self.device)

        stl_loss_A = criterion( A_stl, labels_A ) * 0.2 + criterion( A1_stl, labels_A1 ) * 0.15 + criterion( A2_stl, labels_A2 ) * 0.15
        stl_loss_B = criterion( B_stl, labels_B ) * 0.2 + criterion( B1_stl, labels_B1 ) * 0.15 + criterion( B2_stl, labels_B2 ) * 0.15
        stl_loss = stl_loss_A + stl_loss_B

        return stl_loss


    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return wav_padding(wav, sr=16000, frame_period=5, multiple = 4)  # TODO

    def train(self):
        """Train StarGAN."""

        log = self.log
        
        # Set data loader.
        train_loader = self.train_loader
        data_iter = iter(train_loader)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            log("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        log('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                
            # =================================================================================== #

            # Fetch real data and labels.
            
            try:
                A, B = next(data_iter)
            except:
                data_iter = iter(train_loader)
                A, B = next(data_iter)
            
            A.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
            B.unsqueeze_(1)

            A = A.to(self.device)                         # Input A.
            B = B.to(self.device)                         # Input B.

            # =================================================================================== #
            #                              2. Train 
            # =================================================================================== #
            # Direction: A -> AB -> ABA
            AB = self.G_A2B(A)
            ABA = self.G_B2A(AB)

            # Direction: B -> BA -> BAB
            BA = self.G_B2A(B)
            BAB = self.G_A2B(BA)

            # Reconstruction loss
            recon_loss_A = self.recon_criterion( ABA, A) * self.lambda_rec
            recon_loss_B = self.recon_criterion( BAB, B) * self.lambda_rec

            # Real/Fake GAN loss (A)
            A_dis = self.D_A(A)
            BA_dis = self.D_A(BA)
            ABA_dis = self.D_A(ABA)

            dis_loss_A, gen_loss_A = self.get_gan_loss(A_dis, BA_dis, ABA_dis, self.gan_criterion)

            # Real/Fake GAN loss (A)
            B_dis = self.D_B(B)
            AB_dis = self.D_B(AB)
            BAB_dis = self.D_B(BAB)

            dis_loss_B, gen_loss_B = self.get_gan_loss(B_dis, AB_dis, BAB_dis, self.gan_criterion)

            # Classification loss
            A_stl = self.C(A)
            B_stl = self.C(B)
            AB_stl = self.C(AB) 
            BA_stl = self.C(BA)
            ABA_stl = self.C(ABA)      
            BAB_stl = self.C(BAB)

            stl_loss = self.get_stl_loss(A_stl, BA_stl, ABA_stl, B_stl, AB_stl, BAB_stl, self.stl_criterion) * self.lambda_stl


            if i < self.gan_curriculum:
                rate = self.starting_rate  # 0.01
            else:
                rate = self.default_rate   # 0.5

            gen_loss_A_total = gen_loss_A * (1.-rate) + recon_loss_A * rate
            gen_loss_B_total = gen_loss_B * (1.-rate) + recon_loss_B * rate
            gen_loss = gen_loss_A_total + gen_loss_B_total + stl_loss
            dis_loss = dis_loss_A + dis_loss_B + stl_loss

            # Update parameters.
            self.reset_grad()
            if (i+1) % self.n_critic == 0:
                gen_loss.backward()
                self.optim_gen.step()
            else:
                dis_loss.backward()
                self.optim_dis.step()
                self.optim_clf.step()


            # Logging.
            loss = {}
            loss['D_loss_real_A'] = dis_loss_A.item()
            loss['D_loss_fake_A'] = gen_loss_A.item()
            loss['loss_cls'] = stl_loss.item()
            loss['recon_loss_A'] = recon_loss_A.item()


            # =================================================================================== #
            #                                 3. Miscellaneous                                    
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                msg = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    msg += ", {}: {:.4f}".format(tag, value)
                log(msg)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            if (i+1) % self.sample_step == 0:
                sampling_rate=16000
                num_mcep=36
                frame_period=5
                with torch.no_grad():
                    for idx, wav in tqdm(enumerate(test_wavs)):
                        wav_name = basename(test_wavfiles[idx])
                        # print(wav_name)
                        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        # f0_converted = pitch_conversion(f0=f0, 
                        #     mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src, 
                        #     mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
                        uv, cont_lf0_lpf = get_cont_lf0(f0)
                        # print('--- 1 ---')
                        le = get_log_energy(sp)
                        lf0_normed = (cont_lf0_lpf - self.test_loader.lf0_mean_src) / self.test_loader.lf0_std_src
                        le_normed = (le - self.test_loader.le_mean_src) / self.test_loader.le_std_src
                        lf0_le_cwt = get_lf0_le_cwt(lf0_normed, le_normed) 
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                        
                        coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        mc_lf0_le = np.concatenate((coded_sp_norm, lf0_le_cwt), -1)
                        mc_lf0_le_tensor = torch.FloatTensor(mc_lf0_le.T).unsqueeze_(0).unsqueeze_(1).to(self.device)
                        # conds = torch.FloatTensor(self.test_loader.emo_c_trg).to(self.device)
                        # print(conds.size())
                        # print('--- 2 ---')
                        mc_lf0_le_converted_norm = self.G_A2B(mc_lf0_le_tensor).data.cpu().numpy()
                        mc_lf0_le_converted_norm = np.squeeze(mc_lf0_le_converted_norm).T
                        coded_sp_converted = mc_lf0_le_converted_norm[:, :36] * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                        lf0_converted = inverse_cwt(mc_lf0_le_converted_norm[:, 36:46].T) * self.test_loader.lf0_std_trg + self.test_loader.lf0_mean_trg
                        le_converted = inverse_cwt(mc_lf0_le_converted_norm[:, 46:].T) * self.test_loader.le_std_trg + self.test_loader.le_mean_trg
                        # print('--- 3 ---')
                        # print(lf0_converted.shape)
                        # print(le_converted.shape)
                        e_converted = np.exp(le_converted)
                        f0_converted = np.squeeze(uv) * np.exp(lf0_converted)
                        # print(coded_sp_converted.shape)
                        decoded_sp_converted = world_decode_spectral_envelop(coded_sp_converted, sampling_rate)
                        e_sp_converted = np.linalg.norm(decoded_sp_converted, ord=2, axis=-1)
                        # print('--- 4 ---')
                        # print(e_sp_converted.shape)
                        # print(decoded_sp_converted.shape)
                        e_ratio = np.divide(e_converted, e_sp_converted)
                        decoded_sp_converted = decoded_sp_converted * e_ratio[:, None]

                        wav_transformed = world_speech_synthesis(f0=f0_converted, sp=decoded_sp_converted, 
                                                                ap=ap, fs=sampling_rate, frame_period=frame_period)
                        
                        librosa.output.write_wav(
                            join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-{}-vcto-{}'.format(self.test_loader.src_spk_emo, self.test_loader.trg_spk_emo)+'.wav'), wav_transformed, sampling_rate)
                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, sp=sp, 
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                            librosa.output.write_wav(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate)
                    cpsyn_flag = False

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_A2B_path = os.path.join(self.model_save_dir, '{}-G_A2B.ckpt'.format(i+1))
                G_B2A_path = os.path.join(self.model_save_dir, '{}-G_B2A.ckpt'.format(i+1))
                D_A_path = os.path.join(self.model_save_dir, '{}-D_A.ckpt'.format(i+1))
                D_B_path = os.path.join(self.model_save_dir, '{}-D_B.ckpt'.format(i+1))
                C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i+1))
                torch.save(self.G_A2B.state_dict(), G_A2B_path)
                torch.save(self.G_B2A.state_dict(), G_B2A_path)
                torch.save(self.D_A.state_dict(), D_A_path)
                torch.save(self.D_B.state_dict(), D_B_path)
                torch.save(self.C.state_dict(), C_path)
                log('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                c_lr -= (self.c_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr, c_lr)
                log('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr, c_lr))




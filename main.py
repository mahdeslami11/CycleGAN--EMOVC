import os
import argparse
from solver import Solver
from data_loader import get_loader, TestDataset
from torch.backends import cudnn
from datetime import datetime
from generic_utils import infolog
from os.path import join, basename

log = infolog.log


def str2bool(v):
    return v.lower() in ('true')

def get_default_logdir(logdir_root='logs', msg=''):
    started_datestring = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, started_datestring + '-' + msg)
    print('Using default logdir: {}'.format(logdir))
    return logdir

def prepare_run(args, default=True):
    if default:
        log_dir = get_default_logdir(args.logdir_root, args.log_msg)
    else:
        log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'terminal_train_log'), run_name=args.log_msg)
    sample_dir = join(log_dir, 'samples')
    model_save_dir = join(log_dir, 'models')
    # Create directories if not exist.
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    return log_dir, sample_dir, model_save_dir

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    train_loader_casia = get_loader(config.train_data_dir_casia, 
                                    config.target_speaker, 
                                    config.source_emotion, 
                                    config.target_emotion, 
                                    config.batch_size, 
                                    'train', 
                                    num_workers=config.num_workers)
    test_loader = TestDataset(config.test_data_dir, 
                              config.src_wav_dir, 
                              config.target_speaker, 
                              config.source_emotion, 
                              config.target_emotion)

    # Solver for training and testing StarGAN.
    solver = Solver(train_loader_casia, test_loader, config, log)

    if config.mode == 'train':    
        solver.train()

    # elif config.mode == 'test':
    #     solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--target_speaker', type=str, default='liuchang')
    parser.add_argument('--source_emotion', type=str, default='normal')
    parser.add_argument('--target_emotion', type=str, default='angry')
    parser.add_argument('--num_emotions', type=int, default=3, help='dimension of emotions labels')
    parser.add_argument('--lambda_stl', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=50000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=10000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--c_lr', type=float, default=0.0001, help='learning rate for C')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step, default: None')
    parser.add_argument('--gan_curriculum', type=int, default=1000, help='Strong GAN loss for certain period at the beginning')
    parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
    parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=25001, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--train_data_dir_casia', type=str, default='/scratch/sxliu/data_exp/CASIA_dataset/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='/scratch/sxliu/data_exp/CASIA_dataset/mc/test')
    parser.add_argument('--src_wav_dir', type=str, default='/scratch/sxliu/data_exp/CASIA_dataset/CASIA')
    parser.add_argument('--logdir_root', type=str, default='./logs')
    parser.add_argument('--log_msg', type=str, default='mc_lf0cwt_lecwt-liuchang-normal-angry')
    parser.add_argument('--log_dir', type=str, default=None)

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    config.log_dir, config.sample_dir, config.model_save_dir = prepare_run(config, default=(config.resume_iters is None))
    log(config)
    main(config)
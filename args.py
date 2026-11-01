import os
import pickle
import time
import argparse


def init_args():
    parser = init_parser()
    args = parser.parse_args()
    return init_sub_args(args)


def init_sub_args(args):
    dataset = args.dataset
    if args.vid_path_train and args.vid_path_test and args.pose_path_train and args.pose_path_test:
        args.vid_path = {'train': args.vid_path_train,
                         'test': args.vid_path_test}

        args.pose_path = {'train': args.pose_path_train,
                          'test': args.pose_path_test}
    else:
        args.vid_path = {'train': os.path.join(args.data_dir, dataset, 'train/images/'),
                         'test':  os.path.join(args.data_dir, dataset, 'test/frames/')}

        args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'pose', 'train/'),
                          'test':  os.path.join(args.data_dir, dataset, 'pose', 'test/')}
    args.pose_path["train_abnormal"] = args.pose_path_train_abnormal
    args.ckpt_dir = None
    
    return args


def init_parser(default_data_dir='data/', default_exp_dir='data/exp_dir'):
    parser = argparse.ArgumentParser(prog="TTM-VAD")
    # General Args
    parser.add_argument('--vid_path_train', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train_abnormal', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train', type=str, default=None, help='Path to training pose')
    parser.add_argument('--vid_path_test', type=str, default=None, help='Path to test vids')
    parser.add_argument('--pose_path_test', type=str, default=None, help='Path to test pose')
    parser.add_argument('--dataset', type=str, default='ShanghaiTech',
                        choices=['ShanghaiTech', 'ShanghaiTech-HR', 'UBnormal', 'c1', 'c2', 'c3', 'c4', 'CHAD', 'HuVAD','combined', 'corridor', 'NWPUC', 'Avenue', 'CPCC0','CPCC1','CPCC2','CPCC3','CPCC4','CPCC5','CPCC6', 'CPCC', 'ABC'], help='Dataset for Eval')
    parser.add_argument('--vid_res', type=str, default='1280x720', help='Video Res')   #I changed it from None to 1280x720
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEV', help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999, help='Random seed, use 999 for random (default: 999)')
    parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1], help='Verbosity [1/0] (default: 1)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR', help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--num_workers', type=int, default=16, metavar='W', help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--plot_vid', type=int, default=0, help='Plot test videos')
    parser.add_argument('--only_test', action='store_true', help='Visualize train/test data')
    parser.add_argument('--mask_root', type=str, default='/home/nrashvan/Token/data/ShanghaiTech/gt/test_frame_mask',  help='Path to the gt masks.')

    # Data Params
    parser.add_argument('--num_transform', type=int, default=1, metavar='T', help='number of transformations to use for augmentation (default: 2)')
    parser.add_argument('--num_kp', type=int, default=18, metavar='T', help='number of keypoints in the skeleton structure (default: 17)')
    parser.add_argument('--headless', action='store_true', help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--Kalman_filter', action='store_true', help='Add Kalman Filter')
    parser.add_argument('--norm_scale', '-ns', type=int, default=0, metavar='NS', choices=[0, 1], help='Scale without keeping proportions [1/0] (default: 0)')
    parser.add_argument('--prop_norm_scale', '-pns', type=int, default=1, metavar='PNS', choices=[0, 1], help='Scale keeping proportions [1/0] (default: 1)')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH', help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--seg_len', type=int, default=24, metavar='SGLEN', help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--seg_stride', type=int, default=12, metavar='SGST', help='Stride for training segment sliding window')
    parser.add_argument('--specific_clip', type=int, default=None, help='Train and Eval on Specific Clip')
    parser.add_argument('--global_pose_segs', action='store_false', help='Use unormalized pose segs')
    parser.add_argument('--relative', action='store_true', help='Adding relative movement to the data')
    parser.add_argument('--traj', action='store_true', help='Adding global trajectory')
    
    
    
    # Model Params
    parser.add_argument('--model_ckpt_dir', type=str, metavar='model', help="Path to a pretrained model")
    parser.add_argument('--model_save_dir', type=str, default= './results', metavar='model', help="Path to a pretrained model")
    parser.add_argument('--batch_size', type=int, default=512,  metavar='B', help='Batch size for train')
    parser.add_argument('--epochs', '-model_e', type=int, default=8, metavar='E', help = 'Number of epochs per cycle')
    parser.add_argument('--model_optimizer', '-model_o', type=str, default='adam', metavar='model_OPT', help = "Optimizer")
    parser.add_argument('--model_lr', type=float, default=1e-4, metavar='LR', help='Optimizer Learning Rate Parameter')  #default=5e-10
    parser.add_argument('--model_weight_decay', '-model_wd', type=float, default=5e-5, metavar='WD', help='Optimizer Weight Decay Parameter')
    parser.add_argument('--model_lr_decay', '-model_ld', type=float, default=0.99, metavar='LD', help='Optimizer Learning Rate Decay Parameter')
    parser.add_argument('--sched', type=str, default='cosine', metavar='AE_SCH', help="Optimization LR scheduler (default: 'tri')")
    parser.add_argument('--save_results', action='store_true', help='save the scores')
    parser.add_argument('--save_results_dir', type=str, help="Path for saving scores")
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    
    
     # ==========================================================
        # Checkpoint / output
        # ==========================================================
    
  
    

    parser.add_argument(
            "--grad_clip",
            type=float,
            default=1.0,
    )
    
    
    parser.add_argument("--ttm_sequence_length",type=int,default=8,help="Number of pose windows / TTM steps per chunk.",)
    parser.add_argument(
        "--ttm_embed_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--ttm_process_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--ttm_memory_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ttm_num_layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--ttm_mlp_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--ttm_num_heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--window_num_layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    
    
     # ==========================================================
    # Static normal memory
    # ==========================================================

    parser.add_argument(
        "--num_static_slots",
        type=int,
        default=16,
        help="Number of static-memory prototypes.",
    )

    parser.add_argument(
        "--static_warmup_epochs",
        type=int,
        default=5,
        help="Epochs for linear static-loss weight warm-up.",
    )

    parser.add_argument(
        "--static_lambda_start",
        type=float,
        default=0.01,
    )
    
    
    # ==========================================================
    # Loss
    # ==========================================================

    parser.add_argument(
        "--lambda_memory",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--lambda_static",
        type=float,
        default=0.1,
        help="Final static-memory loss weight.",
    )
    
    
     # ==========================================================
    # Anomaly score
    # ==========================================================

    parser.add_argument(
        "--memory_score_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--static_score_weight",
        type=float,
        default=1.0,
    )

    # ==========================================================
    # Dynamic-memory gating
    # ==========================================================

    parser.add_argument(
        "--normal_gate_quantile",
        type=float,
        default=0.99,
    )

    

    return parser





def create_exp_dirs(experiment_dir, dirmap=''):
    time_str = time.strftime("%b%d_%H%M")

    experiment_dir = os.path.join(experiment_dir, dirmap, time_str)
    dirs = [experiment_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return experiment_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)


def save_dataset(dataset, fname):
    with open(fname, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(fname):
    with open(fname, 'rb') as file:
        return pickle.load(file)


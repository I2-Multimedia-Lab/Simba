from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

import os
os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'local'

def main():
    # Optimize torch.float32 matrix multiplication precision: [highest, high, medium]
    # 'highest' is fastest but loses precision; 'medium' is balanced
    torch.set_float32_matmul_precision('medium')
    
    # Parse arguments
    args = parser.get_args()

    # 🚫 Auto-inject local_rank if not specified, get from torchrun environment variables
    if not hasattr(args, "local_rank"):
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # CUDA setup
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True 

    if args.launcher == 'none': 
        args.distributed = False
    else:
        args.distributed = True

        # ✅ Initialize process group (cannot set device yet)
        dist_utils.init_dist(args.launcher)

        # ✅ Get current process rank and world_size
        rank, world_size = dist_utils.get_dist_info()
        args.rank = rank
        args.world_size = world_size

        # ✅ Set local_rank and assign GPU
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if args.use_gpu:
            torch.cuda.set_device(args.local_rank)

    # Logger setup - Modified: only create file logger for rank 0, others use console
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if not args.distributed or args.local_rank == 0:
        log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    else:
        log_file = None
    
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    
    # Ensure all processes can output logs to console normally
    if args.distributed and args.local_rank != 0:
        # Set appropriate log level for non-main processes
        logger.setLevel(logging.INFO)

    # Define tensorboard writer
    if not args.test:
        if not args.distributed or args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    else:
        train_writer = val_writer = None

    # Load configuration
    config = get_config(args, logger=logger)

    # Batch size configuration
    if args.distributed:
        assert config.total_bs % args.world_size == 0
        config.dataset.train.others.bs = config.total_bs // args.world_size
    else:
        config.dataset.train.others.bs = config.total_bs

    # Log arguments and configuration
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    logger.info(f'Distributed training: {args.distributed}')

    # Set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)

    # Run training or testing
    if args.test:
        test_net(args, config)
    else:
        run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
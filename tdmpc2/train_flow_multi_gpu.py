import os
# os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
# from common.seed import set_seed
from grl.utils import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2, TDMPC2_Flow, TDMPC2_Flow_MultiGPU
from trainer.offline_trainer import MultiGPUOfflineTrainer
from trainer.online_trainer import MultiGPUOnlineTrainer
from common.logger import Logger
from accelerate import Accelerator, DistributedDataParallelKwargs

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config_flow', config_path='configs')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	#ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
	#accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])

	accelerator = Accelerator()
	set_seed(cfg.seed + accelerator.process_index)

	trainer_cls = MultiGPUOfflineTrainer if cfg.multitask else MultiGPUOnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2_Flow_MultiGPU(cfg, accelerator),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train(accelerator)
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()

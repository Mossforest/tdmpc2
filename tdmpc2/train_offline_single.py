import xml.etree.ElementTree as ET
import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


def train(cfg: dict, friction):
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
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer_cls = OfflineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


@hydra.main(config_name='config_train_offline_single', config_path='configs')
def train_single(cfg: dict):

    # XML文件路径
    xml_file_path = '/mnt/nfs/chenxinyan/Metaworld/metaworld/envs/assets_v2/objects/assets/xyz_base.xml'
    frictions = [str(fric)+" 0.1 0.002" for fric in cfg.friction]
    for idx, friction in enumerate(frictions):
        # 解析XML文件
        # tree = ET.parse(xml_file_path)
        # root = tree.getroot()
        # for geom in root.findall('.//geom'):
        #     # 检查geom元素是否有name属性且值为'leftpad_geom'
        #     if 'name' in geom.attrib and geom.get('name') in ['leftpad_geom', 'rightpad_geom']:
        #         # 如果friction属性存在，则更改它
        #         if 'friction' in geom.attrib:
        #             geom.set('friction', friction)
        # # 保存修改后的XML文件
        # tree.write(xml_file_path)
        
        train(cfg, cfg.friction[idx])
        print(f'\n\n ========== done: friction {cfg.friction[idx]} ==========')


if __name__ == '__main__':
	train_single()

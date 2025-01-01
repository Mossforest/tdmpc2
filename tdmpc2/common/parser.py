import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from common import TASK_SET


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except:
            pass

    # Convenience
    cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / cfg.exp_name
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

    return cfg

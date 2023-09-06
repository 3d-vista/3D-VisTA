from torch.utils.tensorboard import SummaryWriter
from pipeline.registry import registry
import os
import wandb
import collections.abc

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

@registry.register_utils("tensorboard_logger")
class TensorboardLogger(object):
    def __init__(self, cfg):
        log_dir = cfg['logger']['args']['log_dir']
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir)
    
    def log(self, log_dict, step=None):
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, step)

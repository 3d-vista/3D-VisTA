import os

import torch

from pipeline.registry import registry


@registry.register_utils("model_saver")
class ModelSaver(object):
    def __init__(self, save_dir=None, save_name=None, load_dir=None, load_name=None):
        if save_dir and not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if save_dir:
            self.save_path = os.path.join(save_dir, save_name)
        else:
            self.save_path = None
        if load_dir is not None:
            self.load_path = os.path.join(load_dir, load_name)
        else:
            self.load_path = self.save_path
        
    def save_model(self, model):
        torch.save(model.state_dict(), self.save_path)
    
    def restore_model(self, model):
        model.load_state_dict(torch.load(self.load_path))
        
    def save_dict(self, state_dict):
        torch.save(state_dict, self.save_path)
        
    def restore_dict(self):
        return torch.load(self.load_path)
    
if __name__ == '__main__':
    pass
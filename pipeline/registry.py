import glob
import os
import importlib

class Registry:
    mapping = {
        "dataset": {},
        "vision_model": {},
        "language_model": {},
        "other_model": {},
        "model_assembler": {},
        "optimizer": {},
        "scheduler": {},
        "optimize_assembler": {},
        "utils" : {}
        
    }
    
    @classmethod
    def register_dataset(cls, name):
        def wrap(trainer_cls):
            cls.mapping["dataset"][name] = trainer_cls
            return trainer_cls
        
        return wrap
        
    @classmethod
    def register_vision_model(cls, name):
        def wrap(trainer_cls):
            cls.mapping["vision_model"][name] = trainer_cls
            return trainer_cls
        
        return wrap
    
    @classmethod
    def register_language_model(cls, name):
        def wrap(trainer_cls):
            cls.mapping["language_model"][name] = trainer_cls
            return trainer_cls
        
        return wrap
    
    @classmethod
    def register_other_model(cls, name):
        def wrap(trainer_cls):
            cls.mapping["other_model"][name] = trainer_cls
            return trainer_cls
        
        return wrap
    
    @classmethod
    def register_model_assembler(cls, name):
        def wrap(trainer_cls):
            cls.mapping["model_assembler"][name] = trainer_cls
            return trainer_cls
        
        return wrap
    
    @classmethod
    def register_optimizer(cls, name):
        def wrap(trainer_cls):
            cls.mapping["optimizer"][name] = trainer_cls
            return trainer_cls
        
        return wrap
    
    @classmethod
    def register_scheduler(cls, name):
        def wrap(trainer_cls):
            cls.mapping["scheduler"][name] = trainer_cls
            return trainer_cls
        
        return wrap
        
    @classmethod
    def register_optimize_assembler(cls, name):
        def wrap(trainer_cls):
            cls.mapping["optimize_assembler"][name] = trainer_cls
            return trainer_cls
        
        return wrap

    @classmethod
    def register_utils(cls, name):
        def wrap(trainer_cls):
            cls.mapping["utils"][name] = trainer_cls
            return trainer_cls
        
        return wrap
        
    @classmethod
    def get_dataset(cls, name):
        return cls.mapping["dataset"][name]

    @classmethod
    def get_vision_model(cls, name):
        return cls.mapping["vision_model"][name]
    
    @classmethod
    def get_language_model(cls, name):
        return cls.mapping["language_model"][name]
     
    @classmethod
    def get_other_model(cls, name):
        return cls.mapping["other_model"][name]
    
    @classmethod
    def get_model_assembler(cls, name):
        return cls.mapping["model_assembler"][name]
        
    @classmethod
    def get_optimizer(cls, name):
        return cls.mapping["optimizer"][name]

    @classmethod
    def get_scheduler(cls, name):
        return cls.mapping["scheduler"][name]
     
    @classmethod
    def get_optimize_assembler(cls, name):
        return cls.mapping["optimize_assembler"][name]
    
    @classmethod
    def get_utils(cls, name):
        return cls.mapping["utils"][name]
    
registry = Registry()

def setup_imports(base_folder="./"):
    # folders to import
    folder_list = ["dataset", "model", "optimization", "pipeline", "utils"]
    files = sum([glob.glob(os.path.join(base_folder, folder) + "/**", recursive=True) for folder in folder_list], [])
    for f in files:
        if f.endswith(".py") and not "setup.py" in f:
            #f = os.path.realpath(f)
            splits = f.split(os.sep)[1:]
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(splits[0:-1] + [module_name])
            importlib.import_module(module)
    

if __name__ == "__main__":
    setup_imports()
    
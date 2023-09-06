from pipeline.registry import registry
from pipeline.pipeline import *
import yaml

class PipelineFactory(object):
    def __init__(self):
        pass
    
    @classmethod
    def create_pipelines_from_yml(cls, yml_path):
        with open(yml_path, "r") as stream:
            try:
                yml_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)
            
            pipeline_list = []
            for pipeline_id in yml_file:
                name = yml_file[pipeline_id]['name']
                if name == "optimus_prime":
                    pipeline = cls.get_optimus_prime(yml_file[pipeline_id])
                else:
                    raise ValueError("pipline {} does not exist".format(name))

                pipeline_list.append(pipeline)
            
            return pipeline_list
                    
    @staticmethod
    def get_optimus_prime(cfg):
        return OptimusPrimePipeline(cfg)

pipeline_factory = PipelineFactory()

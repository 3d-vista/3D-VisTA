from pipeline.registry import registry, setup_imports
from pipeline.pipeline_factory import pipeline_factory
import argparse

def get_arg_parse():
    parser = argparse.ArgumentParser(description='config path')
    parser.add_argument('--config', type=str, default=None, help='path of cfg')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arg_parse()
    setup_imports()
    pipelines = pipeline_factory.create_pipelines_from_yml(args.config)
    for pipeline in pipelines:
        pipeline.run_all()
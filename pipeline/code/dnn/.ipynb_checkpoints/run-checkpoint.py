# run.py
from config.args import get_args
from config import Config
from pipeline import CorticalPipeline

if __name__ == '__main__':

    args_ = get_args()

    params = TrainingParameters(raw_args)

    pipeline = DnnPipeline(params)
    
    pipeline.execute()
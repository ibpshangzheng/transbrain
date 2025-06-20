# pipeline.py
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import TrainingParameters
from dataloader import generate_human_matrix

class DnnPipeline(object):

    def __init__(self, params: TrainingParameters):
        self.params = params
        self._setup_directories()

    def _setup_directories(self):

        os.makedirs(f'{self.params.model_s_path}', exist_ok=True)

    def execute(self):

        if self.params.train:
            self.train()
        if self.params.trans:
            self.transform()

    def train(self):

        for data_id, data_path in enumerate(self.config.mix_data_path_list):
            self._process_single_dataset(data_path, data_id)

    def _process_single_dataset(self, mix_path, data_id):
        """处理单个数据集"""
        X, y, df, genes = generate_human_matrix(mix_path, self.params)
        dataset = np.concatenate((X, y), axis=1)
        
        for i in tqdm(range(self.params.repeat_n)):
            train, valid = train_test_split(dataset, test_size=self.params.valid_ratio)
            # 调用训练模块
            self._train_single_model(train, valid, mix_path, i)

    def _train_single_model(self, train_data, valid_data, mix_path, iteration):
        """单个模型的训练流程"""
        # 调用 train.py 中的训练函数
        train_model(
            params=self.params,
            train_data=train_data,
            valid_data=valid_data,
            mix_path=mix_path,
            iteration=iteration
        )
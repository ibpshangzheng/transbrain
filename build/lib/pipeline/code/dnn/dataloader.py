import anndata
import pandas as pd
from functools import wraps
from pipeline.code.dnn.config.logger import get_logger
import torch.utils.data as torch_data
from skorch.helper import DataFrameTransformer

class MyDataset(torch_data.Dataset):
    def __init__(self, table):
        self.table = table

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        data = self.table[idx, 0:-1]
        label = self.table[idx, -1]
        return data, label

def create_dataloader(dataset, batch_size, shuffle,drop_last) -> torch_data.DataLoader:
    return torch_data.DataLoader(
        MyDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last
    )

def extract_sample_list(original_list):

    sample_ids = []
    
    for item in original_list:
        parts = item.split('_')
        if len(parts) < 2 or not parts[1].startswith('sample'):
            return None
        sample_ids.append(f"{parts[0]}_{parts[1]}")

    return sample_ids

def generate_net_input(mession=None):
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            df,sample_id,all_id = func(*args, **kwargs)

            y_label = df['index'].values
        
            label_df = df[['index']].astype('category')
            label_transformer = DataFrameTransformer()
            y = label_transformer.fit_transform(label_df)['index'].reshape(-1, 1)

            feature_df = df.drop(columns=['index'], errors='ignore').astype(float)
            feature_transformer = DataFrameTransformer()
            X = feature_transformer.fit_transform(feature_df)['X']

            return X, y, y_label,sample_id,all_id
        
        return wrapper
    
    return decorator 

@ generate_net_input() 
def generate_matrix(anndata_path,params):

    logger=get_logger()

    try:

        anndata_ex = anndata.read_h5ad(anndata_path)

        region_id = [i.split('_')[0] for i in anndata_ex.obs['region_index'].values.tolist()]
        sample_id = extract_sample_list(anndata_ex.obs['region_index'].values.tolist())
        all_id = anndata_ex.obs['region_index'].values.tolist()

        expression = pd.DataFrame(anndata_ex.X,index=region_id,
                                  columns=anndata_ex.var['Gene'].values.tolist())
        
        expression = expression.drop(index="other", errors="ignore")

        common_gene = pd.read_csv(params.data_files['gene_path'])
        common_gene = common_gene['genes'].to_list()
        expression = expression[common_gene]

        expression.rename_axis('index',inplace=True)
        expression.reset_index(inplace=True)

    except Exception as e:
            logger.error(e)
            raise
    
    return expression,sample_id,all_id
import anndata
import pandas as pd
from functools import wraps
from config.logger import get_logger
from skorch.helper import DataFrameTransformer

logger = get_logger()

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

    try:

        anndata_ex = anndata.read_h5ad(anndata_path)

        region_id = [i.split('_')[0] for i in anndata_ex.obs['brain_region'].values.tolist()]
        sample_id = extract_sample_list(anndata_ex.obs['brain_region'].values.tolist())
        all_id = anndata_ex.obs['brain_region'].values.tolist()

        expression = pd.DataFrame(anndata_ex.X,index=region_id,
                                  columns=anndata_ex.var['gene'].values.tolist())
        
        expression.rename_axis('index',inplace=True)
        expression.reset_index(inplace=True)

        common_gene = pd.read_csv(params.data_files['gene_path'])
        common_gene = common_gene['genes'].to_list()
        expression = expression[common_gene]

    except Exception as e:
            logger.error(e)
            raise
    
    return expression,sample_id,all_id
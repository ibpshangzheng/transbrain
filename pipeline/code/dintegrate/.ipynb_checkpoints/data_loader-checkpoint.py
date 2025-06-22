import pandas as pd
from typing import Tuple
from config import Config

def load_data(config: Config) -> Tuple[pd.DataFrame, ...]:
    """
    Load all required files
    
    Args:
        config: Pipeline configuration object
        
    Returns:
        Tuple of (align_csv,ahba_mean_ex,sample_ex,h19_data,thre_df,gene_filter)

    """

    align_csv = pd.read_csv(config.data_files['alignment_file'], index_col=0)   
    ahba_mean_ex = pd.read_csv(config.data_files['mean_path'], index_col=0)
    sample_ex = pd.read_csv(config.data_files['sample_path'], index_col=0)
    h19_data = pd.read_feather(config.data_files['h19_path'], index_col=0)
    thre_df = pd.read_csv(config.data_files['thre_file'], index_col=0)
    gene_filter = pd.read_csv(config.data_files['gene_file'])['genes'].tolist()
    
    if config.pool_type != 'all':
        is_cortex = h19_data.index.str.contains('Cerebral cortex')
        h19_data = h19_data[is_cortex] if config.pool_type == 'cortex' else h19_data[~is_cortex]
        h19_data['cell_id'] = range(len(h19_data))
    
    ahba_mean_ex = ahba_mean_ex[gene_filter]
    sample_ex = sample_ex[gene_filter]
    h19_data = h19_data[gene_filter]
    
    return align_csv, ahba_mean_ex, sample_ex, h19_data, thre_df, gene_filter
import pandas as pd
from typing import Tuple
from pipeline.code.dintegrate.integrate_config import Config,parse_arguments

def load_data() -> Tuple[pd.DataFrame, ...]:
    """
    Load and preprocess all required datasets for AHBA and single-nucleus integration.

    This function parses command-line arguments, updates the configuration parameters. See in config.py.

    Returns
    -------
    align_csv : pandas.DataFrame
        Regional alignment between AHBA and single-nucleus data.
        Required format: columns include 'brain_region' and 'sn_region'.
    
    ahba_mean_ex : pandas.DataFrame
        AHBA regional mean expression matrix.
        Required format: index = ROIs, columns = genes.

    sample_ex : pandas.DataFrame
        AHBA sample-level expression matrix.
        Required format: index = ROIs of sample, columns = genes.

    sn_data : pandas.DataFrame
        Single-nucleus RNA-seq expression matrix.
        Required format: index = ROIs of smooth_cells, columns = genes.

    thre_df : pandas.DataFrame
        Similarity threshold file used to guide pool construction.
        Required format: index = ROIs, column = 'correlation'.

    gene_filter : list of str
        List of stable gene names used to filter all datasets.

    config : Config
        Configuration object containing all parsed parameters and resolved paths.
    """

    args = parse_arguments()

    config = Config()
    
    config.update_parameters(
        pool = args.pool,
        integrate = args.integrate,
        downsample = args.downsample,
        zscore = args.zscore,
        pool_type = args.pool_type,
        ahba_weight = args.ahba_weight,
        sn_weight = args.sn_weight,
        jobs = args.jobs,
        iterations = args.iterations,
        downsample_n = args.downsample_n,
        corr_threshold = args.corr_threshold,
        screening_thre = args.screening_thre,
        max_value = args.max_value
    )

    config.update_paths(
        mean_path = args.mean_path,
        sample_path = args.sample_path,
        h19_path = args.h19_path,
        alignment_file = args.alignment_file,
        gene_file = args.gene_file,
        thre_file = args.thre_file,
        pool_s_path = args.pool_s_path,
        integrated_s_path = args.integrated_s_path 
    )
    
    align_csv = pd.read_csv(config.data_files['alignment_file'], index_col=0)   
    ahba_mean_ex = pd.read_csv(config.data_files['mean_path'], index_col=0)
    sample_ex = pd.read_csv(config.data_files['sample_path'], index_col=0)
    sn_data = pd.read_feather(config.data_files['h19_path'])
    sn_data.set_index('index',inplace=True,drop=True)
    thre_df = pd.read_csv(config.data_files['thre_file'], index_col=0)
    gene_filter = pd.read_csv(config.data_files['gene_file'])['genes'].tolist()
    
    if config.pool_type != 'all':
        is_cortex = sn_data.index.str.contains('Cerebral cortex')
        sn_data = sn_data[is_cortex] if config.pool_type == 'cortex' else sn_data[~is_cortex]

    ahba_mean_ex = ahba_mean_ex[gene_filter]
    sample_ex = sample_ex[gene_filter]
    sn_data = sn_data[gene_filter]

    sn_data['cell_id'] = range(len(sn_data))
    
    return align_csv, ahba_mean_ex, sample_ex, sn_data, thre_df, gene_filter,config
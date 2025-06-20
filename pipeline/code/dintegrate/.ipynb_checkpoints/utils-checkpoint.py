import math
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats


def dataframe_mean(dataframe: pd.DataFrame, region_: list) -> pd.DataFrame:
    """Calculate mean expression per region and filter for specified regions.
    Args:
        dataframe_scaled: Input DataFrame with expression data
        region_: List of regions to include
        
    Returns:
        DataFrame with mean expression for specified regions
    """
    return (
        dataframe
        .groupby(dataframe.index)
        .mean()
        .dropna(axis=1)
        .T[region_]
        .T
        .reset_index()
        .set_index('index', drop=True)
    )


def stable_gene_filter(AHBA_mean: pd.DataFrame,sn_ex: pd.DataFrame, region_: list, common_genes: list) -> pd.DataFrame:
    """
    Identifies genes with consistent spatial patterns between AHBA and snRNA-seq data. 
    Args:
        AHBA_mean: AHBA regional mean expression (regions x genes)
        sn_ex: Processed snRNA-seq expression (smooth_cells x genes) 
        region_: Brain regions to analyze
        common_genes: gene features
    Returns:
        DataFrame with genes and their cross-modality spatial correlation scores
    """
    AHBA_mean = AHBA_mean[common_genes]
    sn_ex = sn_ex[common_genes]
    AHBA_mean = AHBA_mean.T[region_].T
    sn_mean = dataframe_mean(sn_ex, region_)
    
    correlations = []
    for gene in tqdm(common_genes, desc="Processing genes..."):
        r, _ = stats.spearmanr(
            AHBA_mean[gene].values,
            sn_mean[gene].values
        )
        if not math.isnan(r):
            correlations.append(r)
        else:
            common_genes = [i for i in common_genes if i!=gene]
    return pd.DataFrame({'correlation': correlations},index=common_genes)


def calculate_thresholds(ahba_mean: pd.DataFrame,h19_data: pd.DataFrame,align_csv: pd.DataFrame,percentile: float = 0.1) -> dict:
    """
    Calculate regional correlation thresholds between AHBA and single-nucleus data.
    
    Args:
        ahba_mean: AHBA mean expression (regions x genes)
        h19_data: Single-nucleus expression data (smooth_cells x genes)
        align_csv: Alignment table with columns ['brain_region', 'sn_region']
        percentile: Top percentile threshold (default: 0.1 for top 10%)
    
    Returns:
        Dictionary mapping regions to (correlation, p-value) thresholds
    """
    threshold_dict = {}
    
    for _, row in tqdm(align_csv.iterrows(), total=len(align_csv), desc="Processing regions..."):
        
        allen_region = row['brain_region']
        sc_regions = row['sn_region'].split('\\')
        
        # Skip if region not in AHBA data
        if allen_region not in ahba_mean.index:
            continue

        ahba_expr = ahba_mean.loc[allen_region].values
        h19_expr = h19_data[h19_data.index.isin(sc_regions)].values
    
        correlations = [
            stats.pearsonr(ahba_expr, cell_expr)
            for cell_expr in h19_expr
        ]
        
        sorted_corrs = sorted(correlations, key=lambda x: x[0], reverse=True)
        try:
            threshold_idx = int(len(sorted_corrs) * percentile)
            threshold_dict[allen_region] = sorted_corrs[threshold_idx]
            
        except IndexError:
            continue
            
    return threshold_dict


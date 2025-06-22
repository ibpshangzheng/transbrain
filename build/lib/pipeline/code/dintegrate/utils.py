import math
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats


def dataframe_mean(dataframe: pd.DataFrame, region_: list) -> pd.DataFrame:
    """
    Calculate mean expression per region and filter for specified regions.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input expression matrix with samples as row indices and genes as columns.

    region_ : list of str
        List of brain region names (index values) to retain in the final output.

    Returns
    -------
    pd.DataFrame
        A DataFrame with mean expression values for the specified regions.
        Rows are brain regions, columns are genes.
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

    Parameters
    ----------
    AHBA_mean : pd.DataFrame
        Regional mean expression matrix from AHBA data.  (regions x genes)

    sn_ex : pd.DataFrame
        Processed snRNA-seq expression (smooth_cells x genes) 

    region_ : list of str
        List of brain region names to include in the analysis.

    common_genes : list of str
        List of gene names present in both AHBA and snRNA-seq datasets.

    Returns
    -------
    pd.DataFrame
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


def calculate_thresholds(ahba_mean: pd.DataFrame,sn_data: pd.DataFrame,align_csv: pd.DataFrame,percentile: float = 0.1) -> dict:
    """
    Calculate regional correlation thresholds between AHBA and single-nucleus data.

    Parameters
    ----------
    ahba_mean : pd.DataFrame
        AHBA mean expression matrix (regions x genes), where each row corresponds 
        to a brain region and each column to a gene.

    sn_data : pd.DataFrame
        Single-nucleus expression data (smooth_cells x genes), with cell-level expression 
        profiles aligned to brain regions.

    align_csv : pd.DataFrame
        Alignment table containing two columns: 
        'brain_region' for AHBA region names and 'sn_region' for corresponding snRNA-seq region labels.

    percentile : float, optional
        Top percentile used to define the correlation threshold (default is 0.1, i.e., top 10%).

    Returns
    -------
    dict
        A dictionary mapping each AHBA region (str) to a tuple (correlation, p-value), 
        representing the correlation threshold at the given percentile.
    """

    threshold_dict = {}
    
    for _, row in tqdm(align_csv.iterrows(), total=len(align_csv), desc="Processing regions..."):
        
        allen_region = row['brain_region']
        sc_regions = row['sn_region'].split('\\')
        
        # Skip if region not in AHBA data
        if allen_region not in ahba_mean.index:
            continue

        ahba_expr = ahba_mean.loc[allen_region].values
        sn_expr = sn_data[sn_data.index.isin(sc_regions)].values
    
        correlations = [
            stats.pearsonr(ahba_expr, cell_expr)
            for cell_expr in sn_expr
        ]
        
        sorted_corrs = sorted(correlations, key=lambda x: x[0], reverse=True)
        try:
            threshold_idx = int(len(sorted_corrs) * percentile)
            threshold_dict[allen_region] = sorted_corrs[threshold_idx]
            
        except IndexError:
            continue
            
    return threshold_dict


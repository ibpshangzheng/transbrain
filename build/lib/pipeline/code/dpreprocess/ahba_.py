import warnings
warnings.filterwarnings('ignore')
import abagen
import pandas as pd
from nilearn import image

import os
import itertools
import anndata
import scanpy as sc

micro_path='../../datasets/microarray' # AHBA microarray

def extract_AHBA_data(atlas_path: str,atlas_info_path: str,lr_mirror='bidirectional',gene_norm='srs',sample_norm='srs',return_report=True,
                      return_counts=True,return_donors=True,norm_matched=False,ibf_threshold=0,region_agg=None) -> pd.DataFrame:
    
    """
    Extracts Allen Human Brain Atlas (AHBA) gene expression data based on a given brain atlas.
    For more information, please ref https://abagen.readthedocs.io/en/stable/generated/abagen.get_expression_data.html

    Parameters
    ----------
    atlas_path : str
        Path to the human brain atlas NIfTI file.
    atlas_info_path : str
        Path to a CSV file containing atlas region information with columns:
        'Anatomical Name' and 'Atlas Index'.
    lr_mirror : {'left', 'right', 'bidirectional'}, optional
        How to mirror samples across hemispheres. Default is 'bidirectional'.
    gene_norm : str, optional
        Method by which to normalize microarray expression values for each donor. Default is 'srs'.
    sample_norm : str, optional
        Method by which to normalize microarray expression values for each sample. Default is 'srs'.
    return_report : bool, optional
        Whether to return a string containing longform text describing the processing procedures used to generate the expression DataFrames returned by this function. Default is True.
    return_counts : bool, optional
        Whether to return dataframe containing information on how many samples were assigned to each parcel in atlas for each donor. Default is True.
    return_donors : bool, optional
        Whether to return donor-level expression arrays instead of aggregating expression across donors with provided agg_metric. Default is True.
    norm_matched : bool, optional
        Whether to perform gene normalization (gene_norm) across only those samples matched to regions in atlas instead of all available samples. Default is False.
    ibf_threshold : float, optional
        Intensity-based filtering threshold. Default is 0.
    region_agg : str or None, optional
        Mechanism by which to reduce sample-level expression data into region-level expression.

    Returns
    -------
    dict[str, pandas.DataFrame] or pandas.DataFrame
        If `return_donors=True`, returns a dictionary where keys are donor IDs and values are gene expression DataFrames (regions × genes).
        If `return_donors=False`, returns a single aggregated DataFrame.

    Examples
    --------
    >>> expr_data = extract_AHBA_data(
    ...     '/path/to/atlas.nii.gz',
    ...     '/path/to/atlas_info.csv'
    ... ) 
    """
    
    # Load atlas and region information
    atlas = image.load_img(atlas_path)
    atlas_info = pd.read_csv(atlas_info_path)

    atlas_dict = (
        atlas_info
        .set_index('Atlas Index')['Anatomical Name']
        .to_dict()
    )
    atlas_dict.setdefault(0, 'other')

    # Get expression data from abagen
    expr_raw = abagen.get_expression_data(
        atlas, data_dir=micro_path, lr_mirror=lr_mirror, gene_norm=gene_norm,
        sample_norm=sample_norm, return_report=return_report, return_counts=return_counts,
        return_donors=return_donors, norm_matched=norm_matched, ibf_threshold=ibf_threshold,
        region_agg=region_agg
    )

    # Process each donor's data
    processed_data = {}
    for donor_id, expr_data in expr_raw.items():
        index_values = (
            expr_data.index.get_level_values(0)
            if isinstance(expr_data.index, pd.MultiIndex)
            else expr_data.index
        )

        region_labels = index_values.map(atlas_dict).fillna('other')
        processed_data[donor_id] = expr_data.set_index(region_labels.values)

    return processed_data
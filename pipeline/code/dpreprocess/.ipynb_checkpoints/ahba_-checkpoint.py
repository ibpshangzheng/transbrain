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
        Extract AHBA expression data from specific atlas.
    
    Args:
        atlas_path: Path to the human brain atlas NIfTI file
        atlas_info_path: Path to CSV file containing atlas region information: |Anatomical Name|Atlas Index|
        
    Returns:
        A concatenated DataFrame of expression data from all donors,
        with anatomical region names as indices
        
    Example:
        >>> expr_data = extract_AHBA_data(
                '/path/to/atlas.nii.gz',
                '/path/to/atlas_info.csv'
            )
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
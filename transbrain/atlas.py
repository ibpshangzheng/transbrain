import os
import pandas as pd
import numpy as np
from nilearn import image
from transbrain.config import Config
from typing import Literal

AtlasType = Literal['bn', 'dk', 'aal']
RegionType = Literal['cortex', 'subcortex', 'all']

def fetch_human_atlas(atlas_type: AtlasType = 'bn', region_type: RegionType = 'all'):
    """
    Fetch a human brain atlas image and its region information.

    This function loads a labeled brain atlas image (e.g., Brainnetome, Desikan-Killiany, or AAL) 
    along with its corresponding ROI (region of interest) metadata table.

    Parameters
    ----------
    atlas_type : {'bn', 'dk', 'aal'}, optional
        The type of atlas to load. Must be one of:
        - 'bn'  : Brainnetome Atlas
        - 'dk'  : Desikan-Killiany Atlas
        - 'aal' : Automated Anatomical Labeling (AAL) Atlas
        Default is 'bn'.
    
    region_type : {'cortex', 'subcortex', 'all'}, optional
        Which regions to include in returned region names and info table: cortical, subcortical, or all. Default is 'all'.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - 'atlas' : nibabel.Nifti1Image
            The loaded NIfTI image of the atlas.
        - 'atlas_data' : np.ndarray
            The atlas volume data as a NumPy array.
        - 'region_info' : np.ndarray
            A list of anatomical region names, extracted from the 'Anatomical Name' column of the label file.
        - 'info_table' : pandas.DataFrame
            The full region-of-interest (ROI) information table including additional metadata.

    Raises
    ------
    FileNotFoundError
        If the atlas image file or the ROI label file is not found.
    """

    # Load the Human atlas image
    human_atlas_path,human_labels_path = Config.atlas_resources[atlas_type]

    if not os.path.exists(human_atlas_path):
        raise FileNotFoundError(f"Human atlas file not found at {human_atlas_path}")
    human_brain_atlas = image.load_img(human_atlas_path)

    # Load the ROI information table
    if not os.path.exists(human_labels_path):
        raise FileNotFoundError(f"ROI information file not found at {human_labels_path}")
    
    human_brain_info_all = pd.read_csv(human_labels_path,index_col=0)
    human_brain_atlas_data = np.asarray(human_brain_atlas.dataobj).astype(np.float32)
    
    # Extract region information based on region_type
    cortex_region_list = Config.region_resources[atlas_type][0]
    subcortex_region_list = Config.region_resources[atlas_type][1]

    if region_type == 'cortex':
        human_brain_info = human_brain_info_all.loc[human_brain_info_all['Anatomical Name'].isin(cortex_region_list)]
        human_brain_info = human_brain_info.reset_index(drop=True)
        region_info = human_brain_info['Anatomical Name'].values  # Cortex regions
    elif region_type == 'subcortex':
        human_brain_info = human_brain_info_all.loc[human_brain_info_all['Anatomical Name'].isin(subcortex_region_list)]
        human_brain_info = human_brain_info.reset_index(drop=True)
        region_info = human_brain_info['Anatomical Name'].values  # Cortex regions
    elif region_type == 'all':
        human_brain_info = human_brain_info_all
        region_info = human_brain_info['Anatomical Name'].values 
    else:
        raise ValueError(f"Invalid region_type: {region_type}")


    # Return the results as a dictionary
    return {
        'atlas': human_brain_atlas,
        'atlas_data': human_brain_atlas_data,
        'region_info': region_info,
        'info_table': human_brain_info
    }


def fetch_mouse_atlas(region_type: RegionType = 'all'):
    """
    Fetch the mouse atlas data and related information.

    This function loads the labeled mouse brain atlas image along with its corresponding ROI (region of interest) metadata table.

    Parameters
    ----------
    region_type : {'cortex', 'subcortex', 'all'}, optional
        Which regions to include in returned region names and info table: cortical, subcortical, or all.
        Default is 'all'.

    Returns
    -------
    dict
        A dictionary with the following keys:
        
        - 'atlas' : nibabel.Nifti1Image
            The loaded NIfTI image of the mouse atlas.
        - 'atlas_data' : np.ndarray
            The atlas volume data as a NumPy array.
        - 'region_info' : np.ndarray
            A list of anatomical region names, extracted from the 'Anatomical Name' column of the label file.
        - 'info_table' : pandas.DataFrame
            The full region-of-interest (ROI) information table including additional metadata.

    Raises
    ------
    FileNotFoundError
        If the atlas image file or the ROI label file is not found.
    """
    mouse_atlas_path,mouse_labels_path = Config.atlas_resources['mouse']

    if not os.path.exists(mouse_atlas_path):
        raise FileNotFoundError(f"Mouse atlas file not found at {mouse_atlas_path}")
    mouse_brain_atlas = image.load_img(mouse_atlas_path)

    # Load the ROI information table
    if not os.path.exists(mouse_labels_path):
        raise FileNotFoundError(f"ROI information file not found at {mouse_labels_path}")
    mouse_brain_info_all = pd.read_csv(mouse_labels_path,index_col=0)

    # Extract region information based on region_type
    cortex_region_list = Config.region_resources['mouse'][0]
    subcortex_region_list = Config.region_resources['mouse'][1]

    if region_type == 'cortex':
        mouse_brain_info = mouse_brain_info_all.loc[mouse_brain_info_all['Anatomical Name'].isin(cortex_region_list)]
        mouse_brain_info = mouse_brain_info.reset_index(drop=True)
        region_info = mouse_brain_info['Anatomical Name'].values  # Cortex regions
    elif region_type == 'subcortex':
        mouse_brain_info = mouse_brain_info_all.loc[mouse_brain_info_all['Anatomical Name'].isin(subcortex_region_list)]
        mouse_brain_info = mouse_brain_info.reset_index(drop=True)
        region_info = mouse_brain_info['Anatomical Name'].values  # Cortex regions
    elif region_type == 'all':
        mouse_brain_info = mouse_brain_info_all
        region_info = mouse_brain_info['Anatomical Name'].values 
    else:
        raise ValueError(f"Invalid region_type: {region_type}")


    # Convert the atlas data to a numpy array
    mouse_atlas_data = np.asarray(mouse_brain_atlas.dataobj).astype(np.float32)

    # Return the results as a dictionary
    return {
        'atlas': mouse_brain_atlas,
        'atlas_data': mouse_atlas_data,
        'region_info': region_info,
        'info_table': mouse_brain_info
    }
import os
import pandas as pd
import numpy as np
from nilearn import image



def fetch_human_atlas(region_type='cortex'):
    """
    Fetch the Human BN atlas data and related information.

    Parameters:
        region_type (str): Type of regions to fetch. Options: 'cortex', 'subcortex', 'all'. Default is 'cortex'.

    Returns:
        dict: A dictionary containing the following keys:
              - 'atlas': The loaded Human 127 atlas image.
              - 'atlas_data': The atlas data as a numpy array.
              - 'region_info': A list of anatomical names for the specified regions.
              - 'info_table': The full ROI information table.
    """

    if region_type not in ['cortex', 'subcortex', 'all']:
        raise ValueError("Invalid region_type. Choose in 'cortex', 'subcortex', or 'all'.")

    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    atlas_folder = os.path.join(dir_path, 'Atlas')

    human_127_atlas_path = os.path.join(atlas_folder, 'Human_127atlas_2mm_symmetry.nii.gz')
    human_127_info_path = os.path.join(atlas_folder, 'Table1 ROI of human atlas.xlsx')

    # Load the Human 127 atlas image
    if not os.path.exists(human_127_atlas_path):
        raise FileNotFoundError(f"Human 127 atlas file not found at {human_127_atlas_path}")
    human_127_atlas = image.load_img(human_127_atlas_path)

    # Load the ROI information table
    if not os.path.exists(human_127_info_path):
        raise FileNotFoundError(f"ROI information file not found at {human_127_info_path}")
    human_127_info = pd.read_excel(human_127_info_path)

    if region_type == 'cortex':
        human_info = human_127_info.iloc[:105,:]
        region_info = human_info['Anatomical Name'].values  # Cortex regions
    elif region_type == 'subcortex':
        human_info = human_127_info.iloc[105:,:]
        region_info = human_info['Anatomical Name'].values  # Subcortex regions
    elif region_type == 'all':
        human_info = human_127_info
        region_info = human_127_info['Anatomical Name'].values  # All regions

    human_127_atlas_data = np.asarray(human_127_atlas.dataobj).astype(np.float32)

    # Return the results as a dictionary
    return {
        'atlas': human_127_atlas,
        'atlas_data': human_127_atlas_data,
        'region_info': region_info,
        'info_table': human_info
    }




def fetch_mouse_atlas(region_type='cortex'):
    """
    Fetch the Mouse atlas data and related information.

    Parameters:
        region_type (str): Type of regions to fetch. Options: 'cortex', 'subcortex', 'all'. Default is 'cortex'.

    Returns:
        dict: A dictionary containing the following keys:
              - 'atlas': The loaded Mouse atlas image.
              - 'atlas_data': The atlas data as a numpy array.
              - 'region_info': A list of anatomical names for the specified regions.
              - 'info_table': The full ROI information table.
    """

    if region_type not in ['cortex', 'subcortex', 'all']:
        raise ValueError("Invalid region_type. Choose in 'cortex', 'subcortex', or 'all'.")


    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    atlas_folder = os.path.join(dir_path, 'Atlas')


    mouse_atlas_path = os.path.join(atlas_folder, 'Mouse_atlas.nii.gz')
    mouse_info_path = os.path.join(atlas_folder, 'Table2 ROI of mouse atlas.xlsx')

    # Load the Mouse atlas image
    if not os.path.exists(mouse_atlas_path):
        raise FileNotFoundError(f"Mouse atlas file not found at {mouse_atlas_path}")
    mouse_atlas = image.load_img(mouse_atlas_path)

    # Load the ROI information table
    if not os.path.exists(mouse_info_path):
        raise FileNotFoundError(f"ROI information file not found at {mouse_info_path}")
    mouse_info = pd.read_excel(mouse_info_path)

    # Extract region information based on region_type
    if region_type == 'cortex':
        mouse_info = mouse_info.loc[mouse_info['Brain Region']=='Isocortex']
        region_info = mouse_info['Anatomical Name'].values # Cortex regions
    elif region_type == 'subcortex':
        mouse_info = mouse_info.loc[mouse_info['Brain Region']!='Isocortex']
        region_info = mouse_info['Anatomical Name'].values  # Subcortex regions
    elif region_type == 'all':
        region_info = mouse_info['Anatomical Name'].values  # All regions

    # Convert the atlas data to a numpy array
    mouse_atlas_data = np.asarray(mouse_atlas.dataobj).astype(np.float32)

    # Return the results as a dictionary
    return {
        'atlas': mouse_atlas,
        'atlas_data': mouse_atlas_data,
        'region_info': region_info,
        'info_table': mouse_info
    }



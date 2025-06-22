import os
import numpy as np
import pandas as pd
from scipy import ndimage
from nilearn import image
from nilearn.image import resample_to_img

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
atlas_folder = os.path.join(dir_path, 'Atlas')

human_127_atlas_path = os.path.join(atlas_folder, 'Human_127atlas_2mm_symmetry.nii.gz')
human_127_info_path = os.path.join(atlas_folder, 'Table1 ROI of human atlas.xlsx')

mouse_atlas_path = os.path.join(atlas_folder, 'Mouse_atlas.nii.gz')
mouse_info_path = os.path.join(atlas_folder, 'Table2 ROI of mouse atlas.xlsx')


def get_region_phenotypes(
    phenotype_nii_path,
    label_column='Atlas Index',
    region_column='Anatomical Name',
    method='mean',
    resample = True,
    species = 'Human'
):
    """
    Calculate region-wise phenotypes based on an input NIfTI file and an atlas NIfTI file.

    Parameters:
    
        phenotype_nii_path (str): Path to the phenotype NIfTI file.

        Human nii in MNI space, Mouse nii in Allen CCFv3 space.
        
        info_df (pd.DataFrame): DataFrame containing atlas region information.
        label_column (str): Column in info_df that represents the label values in the atlas. 
        region_column (str): Column in info_df that represents the region names. 
        statistic (str): Statistic to calculate for each region. Options: 'mean', 'sum'. Default is 'mean'.
        resample(bool): Whether to resample the phenotype image to match the atlas image. Default is True.
        speceis: Human or Mouse

    Returns:
        pd.DataFrame: A DataFrame containing region-wise statistics, sorted by the order in info_df.
    """
    # Load the phenotype and atlas NIfTI files
    phenotype_img = image.load_img(phenotype_nii_path)
    
    if species == 'Human':
        atlas_img = image.load_img(human_127_atlas_path)
        info_df = pd.read_excel(human_127_info_path)
    elif species =='Mouse':
        atlas_img = image.load_img(mouse_atlas_path)
        info_df = pd.read_excel(mouse_info_path)

    if resample and phenotype_img.shape != atlas_img.shape:
        print("Resampling phenotype image to match atlas image resolution...")
        atlas_img = resample_to_img(atlas_img,phenotype_img,interpolation='nearest')

    phenotype_data = np.asarray(phenotype_img.dataobj)
    atlas_data = np.asarray(atlas_img.dataobj)

    if phenotype_data.shape != atlas_data.shape:
        raise ValueError("The phenotype and atlas NIfTI files must have the same shape after resampling.")

    # Extract labels from the info DataFrame
    labels = info_df[label_column].values

    # Calculate the region phenotype for each label
    if method == 'mean':
        region_values = ndimage.mean(phenotype_data, labels=atlas_data, index=labels)
    elif method == 'sum':
        region_values = ndimage.sum(phenotype_data, labels=atlas_data, index=labels)
    else:
        raise ValueError(f"Unsupported statistic method: {method}. Choose from 'mean', 'sum'.")

    # Create a DataFrame for the results
    result_df = info_df.copy()
    result_df['Phenotype'] = region_values

    # Sort the DataFrame by the order in info_df
    result_df = result_df[[region_column,'Phenotype']]
    result_df.columns = ['Region name','Phenotype']
    result_df = result_df.set_index('Region name')

    return result_df


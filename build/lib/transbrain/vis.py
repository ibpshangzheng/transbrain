import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image, masking
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import os
import nibabel as nib
import matplotlib.colors as mcolors


file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
atlas_folder = os.path.join(dir_path, 'atlas')


def load_background_image(contour_img, reference_file):
    """
    Load the background image and compute the mask.
    
    Parameters
    ----------
    contour_img : str
        Path to the contour image (NIfTI file).
    reference_file : str
        Path to the reference NIfTI file for background image.

    Returns
    -------
    new_bg_img : NIfTI image
        The background image with applied mask.
    """
    dsure_img = image.load_img(contour_img)
    dsure_bg_img = masking.compute_background_mask(dsure_img)
    new_bg_img = masking.unmask(masking.apply_mask(reference_file, dsure_bg_img), dsure_bg_img)
    return new_bg_img

def adjust_color(color, factor):
    """
    Adjust the saturation and brightness of the color using HSV space.
    
    Parameters
    ----------
    color : array-like
        A color in RGBA format (values between 0 and 1).
    factor : float
        A factor to adjust the color. Values greater than 1 enhance the color, and values less than 1 reduce the color intensity.
    
    Returns
    -------
    adjusted_color : numpy.ndarray
        The adjusted color in RGBA format.
    """
    r, g, b, a = color
    hsv = rgb_to_hsv([r, g, b])
    hsv[1] = min(1, hsv[1] * factor)  # Adjust saturation
    hsv[2] = min(1, hsv[2] * factor)  # Adjust brightness
    adjusted_rgb = hsv_to_rgb(hsv)
    return np.array([adjusted_rgb[0], adjusted_rgb[1], adjusted_rgb[2], a])

def create_custom_cmap(cmap_name="coolwarm", red_adjustment=1.5, blue_adjustment=1.5):
    """
    Create a custom colormap by adjusting the red and blue components.
    
    Parameters
    ----------
    cmap_name : str, optional
        The name of the base colormap (default is 'coolwarm').
    red_adjustment : float, optional
        Factor to adjust the red channel (default is 1.5).
    blue_adjustment : float, optional
        Factor to adjust the blue channel (default is 1.5).
    
    Returns
    -------
    custom_cmap : LinearSegmentedColormap
        The custom colormap with adjusted red and blue components.
    """
    original_cmap = plt.get_cmap(cmap_name)
    colors = original_cmap(np.linspace(0, 1, 256))
    
    for i in range(len(colors)):
        if colors[i, 2] > 0.5:  # Blue part
            colors[i] = adjust_color(colors[i], blue_adjustment)
        elif colors[i, 0] > 0.5:  # Red part
            colors[i] = adjust_color(colors[i], red_adjustment)
    
    custom_cmap = LinearSegmentedColormap.from_list(f"{cmap_name}_adjusted", colors)
    return custom_cmap

def brighten_cmap(cmap, factor=1.2):

    new_colors = []
    for c in cmap(np.linspace(0, 1, cmap.N)):
        new_color = mcolors.rgb_to_hsv(c[:3])
        new_color[2] = min(new_color[2] * factor, 1.0)
        new_colors.append(mcolors.hsv_to_rgb(new_color))
    return LinearSegmentedColormap.from_list("bright_plasma", new_colors)

def zscore_nii(nii_img):
    """
    Perform z-score normalization on the non-zero regions of the NIfTI image.
    
    Parameters
    ----------
    nii_img : NIfTI image
        The input NIfTI image to normalize.
    
    Returns
    -------
    zscore_img : NIfTI image
        The z-score normalized NIfTI image.
    """
    data = nii_img.get_fdata()
    non_zero_mask = data != 0
    non_zero_data = data[non_zero_mask]
    mean = np.mean(non_zero_data)
    std = np.std(non_zero_data)

    data_zscore = np.copy(data)
    data_zscore[non_zero_mask] = (data[non_zero_mask] - mean) / std

    zscore_img = image.new_img_like(nii_img, data_zscore)
    return zscore_img


def map_phenotype_to_nifti(phenotype_df, atlas_dict):
    """
    Map phenotype values to a mouse atlas and create a NIfTI image.

    Parameters:
        phenotype_df (pd.DataFrame): DataFrame with 'Phenotype' column and anatomical names as index.
        atlas_dict (dict): Dictionary with 'info_table' (mapping anatomical names to atlas indices) 
                           and 'atlas' (NIfTI image of the mouse brain atlas).

    Returns:
        phenotype_img (nib.Nifti1Image): NIfTI image with phenotype values mapped to atlas regions.
    """
    phenotype_df = phenotype_df.reset_index()
    phenotype_df.columns = ['Anatomical Name', 'Phenotype']

    info_table = atlas_dict['info_table']
    label_img = atlas_dict['atlas']
    label_data = label_img.get_fdata().astype(int)
    new_header = label_img.header.copy()
    new_header.set_data_dtype(np.float32)
    # Create mapping from ROI name to atlas index
    label_map = dict(zip(info_table['Anatomical Name'], info_table['Atlas Index']))
    # Create empty array and fill with phenotype values
    phenotype_arr = np.zeros_like(label_data, dtype=np.float32)
    for _, row in phenotype_df.iterrows():
        roi_name = row['Anatomical Name']
        value = row['Phenotype']
        if roi_name in label_map:
            roi_idx = label_map[roi_name]
            phenotype_arr[label_data == roi_idx] = value

    # Construct NIfTI image
    phenotype_img = nib.Nifti1Image(phenotype_arr, affine=label_img.affine, header=new_header)
    return phenotype_img



def plot_mouse_phenotype(phenotype_img, coor = [1.85, 0.85, -1.65, -2.15, -3.15], normalize_img=False, symmetric_cbar=True,vmax=2, threshold=0):
    """
    Visualize the phenotype map and overlay contours.
    
    Parameters
    ----------
    phenotype_img : NIfTI image
        The phenotype image to visualize. This is a 3D NIfTI image containing the data to be plotted.

    coor : List, optional
        Coordinates for display
        
    normalize_img : bool, optional
            If True, the phenotype image will be normalized using z-score transformation. Default is False.
                    
    symmetric_cbar : bool, optional
        Whether to use a symmetric colorbar (True) or not (False). Default is True.
        
    vmax : float, optional
        Maximum value for colormap scaling. Default is 2. This parameter defines the upper bound for the colormap range.
        
    threshold : float, optional
        Threshold value for masking the image. Default is 0. Any values below this threshold will be masked (set to 0).
        
    Returns
    -------
    None
        This function does not return any value. It generates and displays a plot of the phenotype image with contours.
    
    """
    template_labels = os.path.join(atlas_folder,'p56_annotation.nii.gz')
    reference_file=reference_file = os.path.join(atlas_folder,'p56_atlas.nii.gz')
    contour_img=os.path.join(atlas_folder,'p56_annotation.nii.gz')

    # Load background image and compute mask
    new_bg_img = load_background_image(contour_img, reference_file)
    
    # Create custom colormap
    red_adjustment = 1.5
    blue_adjustment = 1.5
    custom_cmap = create_custom_cmap('coolwarm', red_adjustment, blue_adjustment)
    bright_cmap = brighten_cmap(custom_cmap, factor=6)

    # Apply z-score normalization to the img
    if normalize_img:
        phenotype_img = zscore_nii(phenotype_img)
        

    levels=[0, 6, 11, 18, 23, 24, 29, 44, 50, 51, 57, 65, 71, 72, 78, 79, 85, 
        86, 92, 93, 99, 100, 106, 107, 113, 114, 120, 122, 128, 136, 142, 
        143, 149, 150, 156, 164, 170, 171, 177, 178, 184, 185, 191, 199, 205, 
        206, 212, 226, 231, 232, 237, 238, 244, 245, 251, 258, 263, 264, 270, 272,
        277, 279, 284, 285, 290, 291, 296, 298, 303, 325, 331, 332, 338, 346, 352,
        353, 359, 457, 461, 462, 466, 467, 472, 473, 489, 490, 491, 527, 530, 535,
        543, 544, 552, 553, 554, 557, 558, 560, 572, 573, 574, 583, 585, 591, 609, 611, 
        612, 614, 615, 619, 620, 638, 643, 651, 652, 654, 655, 656, 657, 665, 667, 673, 
        674, 682, 683, 690, 691, 695, 696, 702, 703, 704, 710, 711, 714, 821, 864]

    # Set cut coordinates for display
    y_coor = coor
    
    # Create the plot
    fig = plt.figure(figsize=(10, 1), dpi=300, facecolor='white')
    disp = plotting.plot_stat_map(phenotype_img, display_mode='y', vmax=vmax,
                                  cut_coords=y_coor, symmetric_cbar=symmetric_cbar, threshold=threshold, 
                                  bg_img=new_bg_img, annotate=False, black_bg=False, figure=fig, cmap=bright_cmap)
    
    # Set colorbar
    colorbar = disp._cbar
    colorbar.set_ticks(np.linspace(-vmax, vmax, num=3))
    
    # Add contours
    disp.add_contours(template_labels, colors="black", linewidths=0.5, levels=levels[::5])
        
    return None



def plot_human_phenotype(phenotype_img, normalize_img = False, cut_coords=range(0, 50, 10), vmax=2, symmetric_cbar = True):
    """
    Plots the human phenotype data on the MNI152 template.

    Parameters
    ----------
    phenotype_img : nib.Nifti1Image
        A NIfTI image containing the phenotype data to be visualized. This image will be overlaid on the MNI152 template.
    
    normalize_img : bool, optional, default=False
        If True, the phenotype image will be normalized using z-score transformation before visualization.

    cut_coords : range, optional, default=range(0, 50, 10)
        Slice coordinates at which the phenotype data will be displayed. The default is from 0 to 50 in steps of 10.

    vmax : float, optional, default=2
        The maximum value for color intensity in the plot. This parameter controls the upper bound for the colormap range.
    
    symmetric_cbar : bool, optional, default=True
        If True, the colorbar will be symmetric, with zero at the center. If False, the colorbar will scale based on the provided data.

    Returns
    -------
    None

    """
     # Apply z-score normalization to the img
    if normalize_img:
        phenotype_img = zscore_nii(phenotype_img)

    # Load background image
    bg_img_path = os.path.join(atlas_folder,'mni152_t1_1mm_brain.nii.gz')
    bg_img = nib.load(bg_img_path)
    bg_data = bg_img.get_fdata()


    brightness_factor = 0.5
    bg_data_adjusted = bg_data * brightness_factor

    # Create new adjusted background image
    adjusted_bg_img = nib.Nifti1Image(bg_data_adjusted, bg_img.affine, bg_img.header)

    # Create custom colormap
    red_adjustment = 1.5
    blue_adjustment = 1.5
    custom_cmap = create_custom_cmap('coolwarm', red_adjustment, blue_adjustment)
    bright_cmap = brighten_cmap(custom_cmap, factor=6)

    # Plot the phenotype data on the adjusted background image
    fig = plt.figure(figsize=(10, 1), dpi=300)
    plotting.plot_stat_map(phenotype_img,
                           draw_cross=True,
                           display_mode='x',
                           annotate=False,
                           cut_coords=cut_coords,
                           threshold=0,
                           symmetric_cbar=symmetric_cbar,
                           cmap=bright_cmap,
                           vmax=vmax,
                           bg_img=adjusted_bg_img,
                           black_bg=False)
    plt.show()


import scipy.io as sio
import numpy as np
import os
import nibabel as nib 
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps
import glob
import nibabel as nib
import nibabel.gifti
import nilearn
from nilearn import datasets, plotting
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
mpl.rcParams['svg.fonttype'] = 'none'
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import logging
import warnings
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

logging.getLogger('nibabel').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
atlas_folder = os.path.join(dir_path, 'atlas')
surf_folder = os.path.join(dir_path, 'vis_file')


### The code is modified based on the visualization functions of Nilearn.


def get_colorbar_and_data_ranges(
    stat_map_data,
    vmin = None,
    vmax=None,
    symmetric_cbar=True,
    force_min_stat_map_value=None
):


    # avoid dealing with masked_array:
    if hasattr(stat_map_data, "_mask"):
        stat_map_data = np.asarray(
            stat_map_data[np.logical_not(stat_map_data._mask)]
        )

    if force_min_stat_map_value is None:
        stat_map_min = np.nanmin(stat_map_data)
    else:
        stat_map_min = force_min_stat_map_value
    stat_map_max = np.nanmax(stat_map_data)

    if symmetric_cbar == "auto":
        if (vmin is None) or (vmax is None):
            symmetric_cbar = stat_map_min < 0 and stat_map_max > 0
        else:
            symmetric_cbar = np.isclose(vmin, -vmax)

    # check compatibility between vmin, vmax and symmetric_cbar
    if symmetric_cbar:
        if vmin is None and vmax is None:
            vmax = max(-stat_map_min, stat_map_max)
            vmin = -vmax
        elif vmin is None:
            vmin = -vmax
        elif vmax is None:
            vmax = -vmin
        elif not np.isclose(vmin, -vmax):
            raise ValueError(
                "vmin must be equal to -vmax unless symmetric_cbar is False."
            )
        cbar_vmin = vmin
        cbar_vmax = vmax

    # set colorbar limits
    else:
        negative_range = stat_map_max <= 0
        positive_range = stat_map_min >= 0
        if positive_range:
            if vmin is None:
                cbar_vmin = 0
            else:
                cbar_vmin = vmin
            cbar_vmax = vmax
        elif negative_range:
            if vmax is None:
                cbar_vmax = 0
            else:
                cbar_vmax = vmax
            cbar_vmin = vmin
        else:
            # limit colorbar to plotted values
            cbar_vmin = vmin
            cbar_vmax = vmax

    # set vmin/vmax based on data if they are not already set
    if vmin is None:
        vmin = stat_map_min
    if vmax is None:
        vmax = stat_map_max

    return cbar_vmin, cbar_vmax, vmin, vmax


def plot_surf_stat_map(coords, faces, stat_map=None,
        elev=0, azim=0,
        cmap='jet',
        threshold=None, bg_map=None,
        mask=None,
        bg_on_stat=False,
        alpha='auto',darkness=0.5,vmin = None,
        vmax=None, symmetric_cbar="auto", returnAx=False,
        figsize=(14,11), label=None, lenient=None,
        **kwargs):

    ''' Visualize results on cortical surface using matplotlib'''


    # load mesh and derive axes limits
    faces = np.array(faces, dtype=int)
    limits = [coords.min(), coords.max()]

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if cmap is given as string, translate to matplotlib cmap
    if type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)

    # initiate figure and 3d axes
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
    # ax1._axis3don = False
    ax1.grid(False)
    ax1.set_axis_off()
    ax1.w_zaxis.line.set_lw(0.)
    ax1.set_zticks([])
    ax1.view_init(elev=elev, azim=azim)
    
    # plot mesh without data
    p3dcollec = ax1.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                triangles=faces, linewidth=0.,
                                antialiased=False,
                                color='white')

    
    if mask is not None:
        cmask = np.zeros(len(coords))
        cmask[mask] = 1
        cutoff = 2
        if lenient:
            cutoff = 0
        fmask = np.where(cmask[faces].sum(axis=1) > cutoff)[0]
        
    # If depth_map and/or stat_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or stat_map is not None:

        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = .5*face_colors[:, :3]

        if bg_map is not None:
            bg_data = bg_map
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_data[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            face_colors = plt.cm.gray_r(bg_faces*darkness)


        # modify alpha values of background
        face_colors[:, 3] = alpha*face_colors[:, 3]

        if stat_map is not None:
            stat_map_data = stat_map
            stat_map_faces = np.mean(stat_map_data[faces], axis=1)
            if label:
                stat_map_faces = np.median(stat_map_data[faces], axis=1)

            # Call _get_plot_stat_map_params to derive symmetric vmin and vmax
            # And colorbar limits depending on symmetric_cbar settings
            cbar_vmin, cbar_vmax, vmin, vmax = \
                get_colorbar_and_data_ranges(stat_map_faces, vmin,vmax,symmetric_cbar)
  
            if threshold is not None:
                kept_indices = np.where(abs(stat_map_faces) >= threshold)[0]
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices]) * face_colors[kept_indices]
                else:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
            else:
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    if mask is not None:
                        face_colors[fmask,:] = cmap(stat_map_faces)[fmask,:] * face_colors[fmask,:]
                    else:
                        face_colors = cmap(stat_map_faces) * face_colors
                else:
                    face_colors = cmap(stat_map_faces)

        p3dcollec.set_facecolors(face_colors)
    
    if returnAx == True:
        return fig, ax1
    else:
        return fig,vmin,vmax


def showSurf(input_data, surf, sulc, cort,dpi, showall=None, output_file=None, cmap='jet', symmetric_cbar=True,vmin = None,vmax=None, darkness=0.5,threshold=None, boundary=False, boundary_color='#626262'):    

    """
    Visualize surface statistical maps using the provided input data and surface mesh information.

    Parameters:
    - input_data (numpy.ndarray): The statistical data to be mapped onto the surface.
    - surf (list): A list containing surface mesh information. The first element is the array of vertex coordinates, and the second element is the array of faces (triangles).
    - sulc (numpy.ndarray): Sulcal depth values for the surface vertices.
    - cort (numpy.ndarray): A mask indicating the cortical vertices.
    - dpi (int): The resolution of the output image.
    - showall (bool, optional): If True, show views from multiple angles.
    - output_file (str, optional): The base name of the output files. If provided, images will be saved.
    - cmap (str, optional): The colormap to be used for the statistical map. Default is 'jet'.
    - symmetric_cbar (bool, optional): Whether to make the colorbar symmetric. Default is True.
    - vmin (float, optional): The minimum value for the colormap. If None, it will be set automatically.
    - vmax (float, optional): The maximum value for the colormap. If None, it will be set automatically.
    - darkness (float, optional): The darkness of the background surface. Default is 0.5.
    - threshold (float, optional): The threshold to apply to the statistical map.
    - boundary (numpy.ndarray, optional): The boundary information for the surface.
    - boundary_color (str, optional): The color to be used for the boundary. Default is '#626262'.
    """

    single_color_cmap = LinearSegmentedColormap.from_list("single_color", [boundary_color, boundary_color], N=256)
    
    f,vmin_,vmax_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=0, cmap=cmap,vmin = vmin,vmax=vmax,
                          symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
    if boundary:
        plotting.plot_surf_contours([surf[0], surf[1]], input_data, figure=f, cmap=single_color_cmap)
    #plt.show()
    plt.close()

    if output_file:
        count = 0
        f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
        plt.tight_layout()
        count += 1
    f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=180, cmap=cmap,vmin = vmin,vmax=vmax,
                          symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
    if boundary:
        plotting.plot_surf_contours([surf[0], surf[1]], input_data, figure=f, cmap=single_color_cmap)
    #plt.show()
    plt.close()

    if output_file:
        f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
        count += 1
    if showall:
        f,vmin_,vmax_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=90, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1
        f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=270, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1
        f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, elev=90, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1
        f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, elev=270, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1

    return vmin_,vmax_



def imageCrop(filename):

    from PIL import Image

    i1 = Image.open(filename)
    i2 = np.array(i1)
    i2[i2.sum(axis=2) == 255*4,:] = 0
    i3 = i2.sum(axis=2)
    x = np.where((i3.sum(axis=1) != 0) * 1)[0]
    y = np.where((i3.sum(axis=0) != 0) * 1)[0]

    result = Image.fromarray(i2[x.squeeze()][:,y.squeeze()])
    result.save(filename)
    
    
path = surf_folder
path_global = surf_folder


surfmL = nib.load(os.path.join(path, 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii')).darrays
surfiL = nib.load(os.path.join(path, 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii')).darrays
surfL = []
surfL.append(np.array(surfmL[0].data*0.3 + surfiL[0].data*0.7))
surfL.append(np.array(surfmL[1].data))

surfmR = nib.load(os.path.join(path, 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii')).darrays
surfiR = nib.load(os.path.join(path, 'S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii')).darrays
surfR = []
surfR.append(np.array(surfmR[0].data*0.3 + surfiR[0].data*0.7))
surfR.append(np.array(surfmR[1].data))
                                      
res = nib.load(os.path.join(path_global, 'L.atlasroi.32k_fs_LR.shape.gii'))
res = res.darrays[0].data
cortL = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))
                                      
res = nib.load(os.path.join(path_global, 'R.atlasroi.32k_fs_LR.shape.gii'))
res = res.darrays[0].data
cortR = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))
cortLen = len(cortL) + len(cortR)
del res

sulcL = np.zeros(len(surfL[0]))
sulcR = np.zeros(len(surfR[0]))
sulcL[cortL] = -1 * np.array(nib.load(os.path.join(path, 'S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii')).dataobj)[0, :len(cortL)]
sulcR[cortR] = -1 * np.array(nib.load(os.path.join(path, 'S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii')).dataobj)[0, len(cortL)::]
sulcL[np.setdiff1d(range(32492),cortL)] = -1 * np.array(nib.load(surf_folder + '/Q1-Q6_R440.sulc.32k_fs_LR.dscalar.nii').dataobj).squeeze()[np.setdiff1d(range(32492),cortL)]
sulcR[np.setdiff1d(range(32492),cortR)] = -1 * np.array(nib.load(surf_folder + '/Q1-Q6_R440.sulc.32k_fs_LR.dscalar.nii').dataobj).squeeze()[32492+np.setdiff1d(range(32492),cortR)]


def PNGWhiteTrim(input_png):
    image=Image.open(input_png)
    image.load()
    imageSize = image.size

    # remove alpha channel
    invert_im = image.convert("RGB") 

    # invert image (so that white is 0)
    invert_im = ImageOps.invert(invert_im)
    imageBox = invert_im.getbbox()

    cropped=image.crop(imageBox)
    return cropped


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('nipy_spectral')
new_cmap = truncate_colormap(cmap, 0.2, 0.95)

#colors1 = plt.cm.YlGnBu(np.linspace(0, 1, 128))
first = int((128*2)-np.round(255*(1.-0.90)))
second = (256-first)
#colors2 = new_cmap(np.linspace(0, 1, first))
colors2 = plt.cm.viridis(np.linspace(0.1, .98, first))
colors3 = plt.cm.YlOrBr(np.linspace(0.25, 1, second))
colors4 = plt.cm.PuBu(np.linspace(0., 0.5, second))
#colors4 = plt.cm.pink(np.linspace(0.9, 1., second))
# combine them and build a new colormap
cols = np.vstack((colors2,colors3))
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)


def visualize_surface_32k_fs_LR(save_path, name, dpi, mymap=mymap, Data=None, dataL=None, dataR=None, vmax=None, threshold=None, darkness =None, boundary=False, Sym = False,boundary_color='#626262'):
    """
    Function to visualize and save surface brain maps with optional boundary overlays and colorbars.

    Parameters:
    - save_path (str): The path where the resulting image will be saved.
    - name (str): The title of the resulting image.
    - dpi (int): The resolution of the saved image in dots per inch.
    - mymap (colormap): The colormap used for the surface plots. Default is 'mymap'.
    - Data (array-like, optional): The combined data for both left and right hemispheres.
    - dataL (array-like, optional): Data specific to the left hemisphere.
    - dataR (array-like, optional): Data specific to the right hemisphere.
    - vmax (float, optional): The maximum value for the color scale. If None, it's calculated based on the 95th percentile.
    - threshold (float, optional): The value below which data is not displayed.
    - darkness (float, optional): The darkness level for the background surface.
    - boundary (bool, optional): If True, a boundary overlay is added to the surface plots.
    - Sym (bool, optional): If True, the color scale is symmetric around zero.
    - boundary_color (str, optional): The color of the boundary overlay. Default is '#626262'.

    """
        
    if Data is not None:
       
        dataL = np.zeros(len(surfL[0]))
        dataL[cortL] = Data[0:len(cortL)]

        dataR = np.zeros(len(surfR[0]))
        dataR[cortR] = Data[len(cortL):cortLen]

    if vmax==None:
        vmax = np.mean(np.percentile(dataL,95)+ np.percentile(dataR,95))

    vmin_left,vmax_left = showSurf(dataL, surfL, sulcL, cortL, dpi,showall=None, output_file=os.path.join(save_path, 'fig.hcp.embed.L'), symmetric_cbar = Sym,vmax =vmax,
             cmap=mymap, threshold=threshold, darkness = darkness,boundary=boundary, boundary_color=boundary_color)
    vmin_right,vmax_right = showSurf(dataR, surfR, sulcR, cortR, dpi,showall=None, output_file=os.path.join(save_path, 'fig.hcp.embed.R'), symmetric_cbar = Sym,vmin = vmin_left,vmax=vmax_left,
             cmap=mymap, threshold=threshold, darkness = darkness,boundary=boundary, boundary_color=boundary_color)   
    
    gap = 150
    i1 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.L.0.png'))
    i2 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.L.1.png'))
    #i4 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.R.0.png'))
    #i3 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.R.1.png'))
    result = Image.new("RGBA", (gap*2+np.shape(i1)[1]+gap+np.shape(i2)[1]+int(gap*4.5), int(3*gap)+np.shape(i1)[0]+gap), (255, 255, 255, 255))
    result.paste(i2, (2*gap, int(3*gap)))
    result.paste(i1, (np.shape(i2)[1]+3*gap, int(3*gap)))
    #result.paste(i3, (np.shape(i1)[1]+3*gap+np.shape(i2)[1]+gap + int(gap*0.5), int(3*gap)))
    #result.paste(i4, (np.shape(i1)[1]+3*gap+np.shape(i2)[1]+gap+np.shape(i3)[1]+gap + int(gap*0.5), int(3*gap)))


    fig, ax = plt.subplots(figsize=(result.width / 90, 2)) 
    fig.patch.set_alpha(0)
    #ax.text(0.5, 0.7, name, ha='center', va='center', fontsize=100)
    #ax.text(0, 0, 'L', ha='center', va='center', fontsize=80)
    #ax.text(1, 0, 'R', ha='center', va='center', fontsize=80)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    plt.close(fig)
    title_image = Image.open(buf)
    result.paste(title_image, (int(1.2*gap), gap))


    fig, ax = plt.subplots(figsize=(1, (result.height / 100) * 0.8)) 
    ax.axis('off')
    fig.patch.set_alpha(0)  

    norm = plt.Normalize(vmin=vmin_left, vmax=vmax_left)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(mymap), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=1, pad=0,aspect=10)
    cbar.ax.spines['top'].set_linewidth(2)
    cbar.ax.spines['right'].set_linewidth(2)
    cbar.ax.spines['bottom'].set_linewidth(2)
    cbar.ax.spines['left'].set_linewidth(2)


    ticks = np.linspace(vmin_left, vmax_left, num=3) 
    ticklabels = [f'{tick: .1f}' for tick in ticks] 
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=60, length=30, width=3)


    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    colorbar_image = Image.open(buf)
    plt.close(fig)

    result.paste(colorbar_image, (result.width - int(2.6 * gap), int(2.8 * gap)))

    right = result.width
    lower = result.height
    result = result.crop((0,  result.height*0.1, right, lower))

    result = result.convert("RGB") 
    result.save(os.path.join(save_path, name+'.png'))

    return result


from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import matplotlib.colors as mcolors

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




def BN_region_to_vertex(phenotype_df):
    """
    Map phenotype data to Brainnetome Atlas vertices.

    Parameters
    ----------
    phenotype_df : pandas.DataFrame
        A DataFrame containing phenotype data with anatomical names as the index. The DataFrame should have at least
        one column with the phenotype data values to be mapped to the Brainnetome Atlas regions.

    Returns
    -------
    BN_data_vis : numpy.ndarray
        A 1D array of size matching the Brainnetome Atlas regions, where the phenotype data is mapped to the corresponding
        Brainnetome Atlas regions.

    """
    
    # Reading region data from the Brainnetome Atlas Excel file
    roi_data = pd.read_excel(os.path.join(atlas_folder,'roi_of_bn_atlas.xlsx'))
    roi_data = roi_data.iloc[0:105].sort_values(by='Left Index')

    # Sorting the phenotype data based on the anatomical names
    phenotype_sorted = phenotype_df.loc[roi_data['Anatomical Name'].values]

    # Concatenating the phenotype data for both hemispheres
    vis_Data = np.squeeze(np.concatenate([phenotype_sorted.values, phenotype_sorted.values], axis=0))

    # Loading Brainnetome Atlas data
    BN_Atlas = nib.load(os.path.join(surf_folder,'100307.BN_Atlas.32k_fs_LR.dlabel.nii'))
    BN_Atlas_data = np.array(BN_Atlas.dataobj).flatten()

    # Adjusting the Brainnetome Atlas data to ensure proper indexing
    for index_ in np.unique(BN_Atlas_data):
        if (index_ > 0) & (index_ % 2 == 0):
            BN_Atlas_data[BN_Atlas_data == index_] = index_ - 210
    
    # Initializing the result array
    BN_data_vis = np.zeros_like(BN_Atlas_data)

    # Mapping phenotype data to the Brainnetome Atlas regions
    for i in range(105):
        BN_data_vis[BN_Atlas_data == 2 * i + 1] = vis_Data[i]
        BN_data_vis[BN_Atlas_data == 2 * i + 2] = vis_Data[105:][i]
    
    return BN_data_vis



from nilearn import image
from IPython.display import display
import shutil



def plot_surface_phenotype(vertex_data, vmax=2, thresh=0, darkness=0.6, show_boundary=True, sym_bar=True):
    """
    Visualize the surface phenotype data by generating a surface map with the given input data.

    Parameters
    ----------
    vertex_data : array-like
        The input vertex data representing the surface phenotype values.
    vmax : float, optional, default=2
        The maximum value for the color scale.
    thresh : float, optional, default=0
        The threshold value below which data is not displayed.
    darkness : float, optional, default=0.6
        The darkness level for the background surface. Ranges between 0 (light) and 1 (dark).
    show_boundary : bool, optional, default=True
        Whether to show the boundary overlay on the surface map.
    sym_bar : bool, optional, default=True
        Whether to make the colorbar symmetric around zero.

    Returns
    -------
    visualize : PIL.Image.Image 
        The generated figure containing the surface phenotype visualization.

    """
    
    # Create custom colormap
    red_adjustment = 1.5
    blue_adjustment = 1.5
    custom_cmap = create_custom_cmap('coolwarm', red_adjustment, blue_adjustment)
    bright_cmap = brighten_cmap(custom_cmap, factor=6)

    # Create temporary directory for saving the figure
    fig_dir = './temp_figure'
    os.makedirs(fig_dir, exist_ok=True)

    # Visualize the surface
    visualize = visualize_surface_32k_fs_LR(save_path=fig_dir, name='Surface Visualization', dpi=300, 
                                             mymap=bright_cmap, Data=vertex_data, vmax=vmax, threshold=thresh, 
                                             darkness=darkness, boundary=show_boundary, Sym=sym_bar)


    # Clean up by removing the temporary directory
    shutil.rmtree(fig_dir)

    return visualize



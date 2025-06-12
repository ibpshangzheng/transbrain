import streamlit as st
import pandas as pd
import os
import sys
import transbrain as tb
import numpy as np
import nibabel as nib
from nilearn import image,plotting
import io
import zipfile

os.environ["WATCHFILES_DISABLE"] = "true"

st.set_page_config(
    page_title="TransBrain",
    layout="centered",
    page_icon="🧠"
)
mapping_done_flag = False


# Step 1: Sidebar - Mapping direction
st.sidebar.markdown(
    '<div style="font-size:2rem; font-weight:bold; color:#2980b9; margin-bottom:20px;">Mapping Options</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div style="font-size: 1em; font-weight: bold; margin-bottom: 0px;">Step 1: Select Mapping Direction</div>',
    unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="font-size: 0.9em; color: gray; margin-top: 6px; margin-bottom: -40px;">
        Select the direction of mapping from human to mouse, or from mouse to human.
    </div>
    """,unsafe_allow_html=True)
direction = st.sidebar.selectbox(
    label="", 
    options=["human to mouse", "mouse to human"]
)


# Step 2: Select region type
st.sidebar.markdown(
    '<div style="font-size: 1em; font-weight: bold; margin-bottom: 0px;">Step 2:  Select Region</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <div style="font-size: 0.9em; color: gray; margin-top: 6px; margin-bottom: -40px;">
        Select the brain region that you want to transform ('all' means the entire brain).
    </div>
    """,unsafe_allow_html=True)
region_type = st.sidebar.selectbox("", ["cortex", "subcortex", "all"])



# Step 3: Select atlas
if direction == "human to mouse":
    mouse_atlas = 'CCF'
    st.sidebar.markdown(
        '<div style="font-size: 1em; font-weight: bold; margin-bottom: 0px;">Step 3:  Select Human Atlas</div>',
        unsafe_allow_html=True)

    st.sidebar.markdown(
    """
    <div style="font-size: 0.9em; color: gray; margin-top: 6px; margin-bottom: -40px;">
        Select the atlas you used when extracting human phenotypes.
    </div>
    """,unsafe_allow_html=True)
    human_atlas_choose = st.sidebar.selectbox("", ["BN", "DK", "AAL"])
    st.sidebar.markdown("**Mouse Atlas:** CCF V3")


else:# mouse to human, choose target atlas
    mouse_atlas = "CCF V3"
    st.sidebar.markdown(
        '<div style="font-size: 1em; font-weight: bold; margin-bottom: 0px;">Step 3:  Select Huamn Atlas:</div>',
        unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div style="font-size: 0.9em; color: gray; margin-top: 6px; margin-bottom: -40px;">
            Select the target human atlas you want to map to.
        </div>
        """,unsafe_allow_html=True)
    human_atlas_choose = st.sidebar.selectbox("", ["BN", "DK", "AAL"])
    st.sidebar.markdown("**Mouse Atlas:** CCF V3")


# Step 4: Data upload format option
st.sidebar.markdown(
    '<div style="font-size: 1em; font-weight: bold; margin-bottom: 0px;">Step 4:  Select Data Upload Format</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <div style="font-size: 0.9em; color: gray; margin-top: 6px; margin-bottom: -40px;">
        Select the data format you use. CSV tables and NII images are supported.
    </div>
    """,unsafe_allow_html=True)
data_type = st.sidebar.selectbox("", ["Table", "Image"])


region_map = {
    'all': 'all',
    'cortex': 'cortical',
    'subcortex': 'subcortical'
}

atlas_type_map = {
    'BN': 'bn',
    'DK': 'dk',
    'AAL': 'aal'    
}

atlas_flag = atlas_type_map.get(human_atlas_choose, None) 
region_flag = region_map.get(region_type, None) 

# Step 5: Download Example CSV or image
if data_type == 'Table':
    st.sidebar.markdown(
        '<div style="font-size: 1em; font-weight: bold; margin-bottom: 0px;">Step 5:  Download Example Data</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        <div style="font-size: 0.9em; color: gray; margin-top: 6px; margin-bottom: 20px;">
            Please download the provided template CSV file and refill it with your data according to the brain region indices.
        </div>
        """,unsafe_allow_html=True)

    if direction == 'mouse to human':
        TEMPLATE_DIR = "./transbrain/exampledata/mouse"
        filename = f"{'mouse'}_{region_flag}_example_data.csv"
        template_path = os.path.join(TEMPLATE_DIR, filename)
    else:
        TEMPLATE_DIR = "./transbrain/exampledata/human"
        filename = f"{'human'}_{atlas_flag}_{region_flag}_example_data.csv"
        template_path = os.path.join(TEMPLATE_DIR,atlas_flag,filename)
    
    if os.path.exists(template_path):
        df = pd.read_csv(template_path)
        csv_bytes = df.to_csv(index=False)
        filename_new = filename
        st.sidebar.download_button("Download Template", csv_bytes, file_name=filename_new)
    else:
        st.sidebar.error("Template file not found.")

elif data_type == 'Image':
    st.sidebar.markdown(
        '<div style="font-size: 1em; font-weight: bold; margin-bottom: 0px;">Step 5:  Download Example Data and Template</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        <div style="font-size: 0.9em; color: gray; margin-top: 6px; margin-bottom: 20px;">
            Please download the provided brain template and example image. Please make sure that your data is registered to the template like the example data.
        </div>
        """,unsafe_allow_html=True)

    if direction == 'mouse to human':
        DATA_DIR = "./transbrain/exampledata/mouse"
        filename = "mouse_example_phenotype_data.nii.gz"
        data_path = os.path.join(DATA_DIR, filename)
        template_path = "./transbrain/atlas/p56_atlas.nii.gz"

    else:
        DATA_DIR = "./transbrain/exampledata/human"
        filename = "human_example_phenotype_data.nii.gz"
        data_path = os.path.join(DATA_DIR, filename)
        template_path = "./transbrain/atlas/mni152nlin6_res-2x2x2_t1w.nii.gz"


    # zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w') as zf:
        zf.write(data_path, arcname=os.path.basename(data_path))
        zf.write(template_path, arcname=os.path.basename(template_path))
    zip_buffer.seek(0)

    st.sidebar.download_button(
        label="Download Example Files",
        data=zip_buffer,
        file_name="Example_data.zip",
        mime="application/zip"
    )

import io
import tempfile

# Step 6: Upload data
#st.sidebar.markdown("#### Step 6:  Upload Your Data")
st.sidebar.markdown(
    '<div style="font-size: 1em; font-weight: bold; margin-bottom: 6px;">Step 6: Upload Your Data</div>',
    unsafe_allow_html=True
)

if data_type == 'Table':
    uploaded_file = st.sidebar.file_uploader("Please upload your data, and we will validate it.", type="csv")
    data_df = None
    if uploaded_file is not None:
        try:
            data_df = pd.read_csv(uploaded_file)
            #st.sidebar.write(f"Uploaded data shape: {data_df.shape}")
            #st.sidebar.write(f"Template data shape: {df.shape}")
            # Load local template for validation
            assert data_df.shape == df.shape, "Shape mismatch with template!"
            assert (data_df.columns == df.columns).all(), "Column name mismatch!"
            assert (data_df.index == df.index).all(), "Index mismatch!"
            assert not data_df.isnull().values.any(), "Uploaded data contains NaN values!"
            st.sidebar.success("Data uploaded and validated successfully.")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

elif data_type == 'Image':
    uploaded_file = st.sidebar.file_uploader("Please upload your phenotype image (.nii or .nii.gz)", type=["nii", "nii.gz"])
    img = None
    if uploaded_file is not None:
        try:

            file_bytes = uploaded_file.getvalue()

            suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name

            uploaded_img = nib.load(tmp_file_path)
            uploaded_img_data = uploaded_img.get_fdata()
            template_img = nib.load(template_path)
            template_img_data = template_img.get_fdata()

            if uploaded_img_data.shape != template_img_data.shape:
                st.sidebar.error(f"Shape mismatch! Uploaded image shape {uploaded_img_data.shape} != template shape {template_img_data.shape}")

            elif not (uploaded_img.affine == template_img.affine).all():
                st.sidebar.error("Affine matrix mismatch between uploaded image and template image.")
            else:
                st.sidebar.success(f"Image uploaded successfully and matches template shape and affine.")
                img = uploaded_img 

        except Exception as e:
            st.sidebar.error(f"Failed to load or validate NIfTI image: {str(e)}")


# Step 7: Prefix input
st.sidebar.markdown(
    '<div style="font-size: 1em; font-weight: bold; margin-bottom: -30px;">Step 7: Enter Save Prefix</div>',
    unsafe_allow_html=True
)
prefix = st.sidebar.text_input("", "Transformed_phenotype")

# Step 8: Optional flags
st.sidebar.markdown(
    '<div style="font-size: 1em; font-weight: bold; margin-bottom: 6px;">Step 8: Normalization Option</div>',
    unsafe_allow_html=True
)
normalize_input = st.sidebar.checkbox("Normalize Input", value=True)
#restore_output = st.sidebar.checkbox("Normalize Output", value=True)

# Step 9: Start mapping
import io
from scipy.stats import zscore

plot_flag = False
if st.sidebar.button("9. Start Mapping"):
    try:
        #table input
        if data_type == 'Table':
            data_df = data_df.set_index('Anatomical Name')
        elif data_type =='Image':
            if direction == "human to mouse":
                atlas_dict = tb.atlas.fetch_human_atlas(atlas_type=atlas_flag,region_type=region_type)
            else:
                atlas_dict = tb.atlas.fetch_mouse_atlas(region_type=region_type)
            data_df = tb.base.get_region_phenotypes(img, atlas_dict = atlas_dict)

        #Initialize TransBrain
        Transformer = tb.trans.SpeciesTrans(atlas_flag)
        if direction == "human to mouse":
            result_df = Transformer.human_to_mouse(
                data_df,
                region_type=region_type,
                normalize=normalize_input
            )
            suffix = f"human_{human_atlas_choose}_{region_type}_to_mouse_CCF"
        else:
            result_df = Transformer.mouse_to_human(
                data_df,
                region_type=region_type,
                normalize=normalize_input
            )
            suffix = f"mouse_CCF_{region_type}_to_human_{human_atlas_choose}"

        output_filename = f"{prefix}_{suffix}.csv"

        #zscore results
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        result_df[numeric_cols] = result_df[numeric_cols].apply(zscore)

        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer)
        csv_bytes = csv_buffer.getvalue()
        st.sidebar.success("✅ Mapping completed.")
        #st.download_button(
        #        label="📥 Download results",
        #        data=csv_bytes,
        #        file_name=output_filename)
        st.sidebar.markdown("Please download the results in main page.")
        mapping_done_flag = True

    except Exception as e:
        st.sidebar.error(f"❌ Mapping failed: {str(e)}")



# Main content: image and introduction
st.image("./web/figure/web_title.png", use_container_width=True)

st.title("TransBrain: translating brain-wide phenotypes between humans and mice")
#需要替换网址
st.markdown("""
🧠 **TransBrain** is an integrated computational framework for bidirectional translation of brain-wide phenotypes between humans and mice. Specifically, TransBrain provides a systematic approach for cross-species quantitative comparison and mechanistic investigation of both normal and pathological brain functions.
""")

st.image("./figure/transbrain_fig1.jpg", caption="TransBrain framework")


st.markdown("""
### What can TransBrain do?

1. Exploring the similarity relationships at the transcriptional level.
2. Inferring the conservation of whole-brain phenotypes.
3. Transforming and annotating whole-brain functional circuits.
4. Linking specific mouse models with human diseases.       
...  
---
            
### Usage

- This website provides a code-free demonstration platform for TransBrain, enabling users to directly upload your data for online mapping and visualization. 
- Supported data formats include ``CSV`` tables and ``NII`` images. For the latter, TransBrain will extract regional phenotypes based on the atlas selected by the user.
- You can follow the step-by-step instructions from in the left sidebar to complete the mapping process.
- If your data has already been normalized, you can uncheck the `Normalize` option.
- When the sidebar shows "Mapping Completed", the results will be visualized at the bottom of the main page. Rendering process may take a few minutes, when it's finshed, you can download the mapped `CSV` table and `nii.gz` file in MNI152 volume space.
- If you are interested in our approach or wish to explore the tool further, please visit the GitHub repository or documentation links below for detailed instructions on setting up your TransBrain environment and accessing detailed tutorials.
        
---
#### ✅ Supported Atlases

##### Human Brain Atlases:
                   
We currently provide the following the options. The naming of regions in the Brainnetome (BN) atlas are defined based on the anatomical locations from [Brodmann atlas](https://en.wikipedia.org/wiki/Brodmann_area). You can check the correspondence in the BN website or in this [table](https://github.com/ibpshangzheng/Transbrain/blob/main/TransBrain/Atlas/BNA_subregions.xlsx) to help understand. For subcortical regions, we adopted a [hybrid approach (22 ROIs)](https://github.com/ibpshangzheng/Transbrain/tree/main/Tutorials/Atlas) that integrates the Brainnetome Atlas, the [Allen Brain Atlas](https://community.brain-map.org/t/allen-human-reference-atlas-3d-2020-new/405), and [public manual delineations](https://www.sciencedirect.com/science/article/abs/pii/S1053811913001237?via%3Dihub).

- [BN (Brainnetome Atlas)](https://atlas.brainnetome.org/)
- [DK (Desikan–Killiany Atlas)](https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation)
- [AAL (Automated Anatomical Labeling)](https://www.gin.cnrs.fr/en/tools/aal/)

              
##### Mouse Brain Atlas:
- [CCFv3 (Allen Mouse Common Coordinate Framework v3)](https://atlas.brain-map.org/)
            

⚠️ **Notice:**

- If you upload volumetric data in ``.nii`` or ``.nii.gz`` format, make sure that it has been aligned to the atlas space required by TransBrain. (Download template and example data to check)      
- If you upload tables, please **strictly follow** the format and region order in the provided template file. Mapping will **fail** if the structure is incorrect.
- For detailed atlases information, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1) and [transbrain/atlas](https://github.com/ibpshangzheng/Transbrain/tree/main/transbrain/atlas)
- Support for additional atlases will be expanded in future updates.
""")
#st.markdown('##### <span style="color:#00BBFF;">Here is a demonstration of how to map your own data using this website.</span>', unsafe_allow_html=True)


st.markdown("""
---
### Futher reading
        
- 🌐 [GitHub Repository](https://github.com/ibpshangzheng/transbrain)
- 📦 [Install from PyPI](https://pypi.org/project/transbrain/)  
- 📄 [Documentation](http://192.168.193.179:10088/index.html#)
- 📜 [Our Paper](https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1)
- 📧 For questions, contact the author: Shangzheng Huang (huangshangzheng@ibp.ac.cn)
""")


from io import BytesIO
import matplotlib.pyplot as plt
import zipfile
import tempfile
from nilearn.image import resample_img

def map_phenotype_to_nifti_raw(mouse_phenotype_df, mouse_atlas_dict):
    """
    Maps ROI-level phenotype values to a mouse atlas label image.

    Parameters:
        mouse_phenotype_df (pd.DataFrame): DataFrame with ['Anatomical Name', 'Phenotype'] columns.
        mouse_atlas_dict (dict): Dictionary containing 'info_table' and 'atlas' keys.

    Returns:
        phenotype_img (nib.Nifti1Image): NIfTI image with mapped phenotype values.
    """
    info_table = mouse_atlas_dict['info_table']
    label_img = mouse_atlas_dict['atlas']
    label_data = label_img.get_fdata().astype(int)
    new_header = label_img.header.copy()
    new_header.set_data_dtype(np.float32)
    # Create mapping from ROI name to atlas index
    label_map = dict(zip(info_table['Anatomical Name'], info_table['Atlas Index']))
    # Create empty array and fill with phenotype values
    phenotype_arr = np.zeros_like(label_data, dtype=np.float32)
    for _, row in mouse_phenotype_df.iterrows():
        roi_name = row['Anatomical Name']
        value = row['Phenotype']
        if roi_name in label_map:
            roi_idx = label_map[roi_name]
            phenotype_arr[label_data == roi_idx] = value

    # Construct NIfTI image
    phenotype_img = nib.Nifti1Image(phenotype_arr, affine=label_img.affine, header=new_header)
    return phenotype_img


def map_phenotype_to_nifti(mouse_phenotype_df, mouse_atlas_dict):
    """
    Fast version: fully vectorized.
    """

    info_table = mouse_atlas_dict['info_table']
    label_img = mouse_atlas_dict['atlas']
    label_data = label_img.get_fdata().astype(int)
    new_header = label_img.header.copy()
    new_header.set_data_dtype(np.float32)

    label_map = dict(zip(info_table['Anatomical Name'], info_table['Atlas Index']))

    phenotype_arr = np.zeros_like(label_data, dtype=np.float32)

    # Flatten the label data to 1D for fast indexing
    label_flat = label_data.ravel()
    phenotype_flat = phenotype_arr.ravel()

    # Build a reverse mapping: Atlas Index → Phenotype Value
    roi_value_map = {label_map[row['Anatomical Name']]: row['Phenotype']
                     for _, row in mouse_phenotype_df.iterrows()
                     if row['Anatomical Name'] in label_map}

    # Convert dict to array assignment
    unique_roi_indices = np.array(list(roi_value_map.keys()))
    unique_values = np.array(list(roi_value_map.values()))

    for roi_idx, value in zip(unique_roi_indices, unique_values):
        mask = (label_flat == roi_idx)
        phenotype_flat[mask] = value

    # Reshape back to original 3D
    phenotype_arr = phenotype_flat.reshape(label_data.shape)
    phenotype_img = nib.Nifti1Image(phenotype_arr, affine=label_img.affine, header=new_header)

    
    return phenotype_img


def get_nii_download_button(img, filename):
    buf = BytesIO()
    img.to_filename('/tmp/tmpfile.nii.gz')  # 先保存到临时文件（必须）
    with open('/tmp/tmpfile.nii.gz', 'rb') as f:
        buf.write(f.read())
    buf.seek(0)
    st.download_button(
        label="Download NIfTI",
        data=buf,
        file_name=filename,
        mime='application/gzip'
    )

def pad_nifti_image(img, pad_width=20):
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    padded_data = np.pad(data,
                         pad_width=((pad_width, pad_width),  
                                    (pad_width, pad_width),  
                                    (pad_width, pad_width)), 
                         mode='constant', constant_values=0)

    new_affine = affine.copy()
    new_affine[:3, 3] -= pad_width * affine[:3, :3].dot([1, 1, 1])

    padded_img = nib.Nifti1Image(padded_data, new_affine, header=header)
    return padded_img

def downsample_img_by_factor(img, factor=2):
    """Downsamples a NIfTI image by the given factor."""
    affine = img.affine.copy()
    affine[:3, :3] *= factor 
    new_shape = np.ceil(np.array(img.shape) / factor).astype(int)
    
    return resample_img(img, target_affine=affine, target_shape=new_shape, interpolation='linear')


if mapping_done_flag:
    st.markdown("""
        ---
        """)

    mouse_atlas = tb.atlas.fetch_mouse_atlas(region_type=region_type)
    human_atlas = tb.atlas.fetch_human_atlas(atlas_type=atlas_flag,region_type = region_type)
    mouse_template = image.load_img('./transbrain/atlas/p56_atlas.nii.gz')
    human_template = image.load_img('./transbrain/atlas/mni152nlin6_res-2x2x2_t1w.nii.gz')

    if direction == 'mouse to human':
        st.markdown("### Visualization of Input Data")
        st.markdown("You can drag the image to display the slice you want.")
        with st.spinner('Rendering...'):
            data_df.reset_index(inplace=True)
            data_df.columns = [['Anatomical Name','Phenotype']]
            source_img = map_phenotype_to_nifti(data_df, mouse_atlas)
            
            #buf = BytesIO()
            #display = plotting.plot_stat_map(
            #    source_img, bg_img=mouse_template, display_mode='y',
            #    cut_coords=range(-4, 3, 1), cmap='coolwarm',
            #    draw_cross=False, annotate=True)
            #plt.savefig(buf, format='png', bbox_inches='tight')
            #plt.close()
            #buf.seek(0)

            padded_source_img = pad_nifti_image(source_img, pad_width=30)
            padded_bg_img = pad_nifti_image(mouse_template, pad_width=30)
            #downsample image to accelerate rendering
            padded_source_img = downsample_img_by_factor(padded_source_img, factor=3)
            padded_bg_img = downsample_img_by_factor(padded_bg_img, factor=3)
            html_view = plotting.view_img(
                padded_source_img, bg_img=padded_bg_img,
                cmap='coolwarm', draw_cross=False, annotate=True, symmetric_cmap=True)
        st.components.v1.html(html_view._repr_html_(),height=260,width=800)
        #st.image(buf, caption="Phenotype in Mouse Space",  use_container_width = True)
        #get_nii_download_button(source_img, "Source_Data.nii.gz")

        st.markdown("### Visualization of Output Data")
        with st.spinner('Rendering...'):
            result_df.reset_index(inplace=True)
            result_df.columns = [['Anatomical Name','Phenotype']]
            target_img = map_phenotype_to_nifti(result_df, human_atlas)
            
            #buf2 = BytesIO()
            #display2 = plotting.plot_stat_map(target_img,cmap='coolwarm',cut_coords=(-20,-10,10))
            #plt.savefig(buf2, format='png', bbox_inches='tight')
            #plt.close()
            #buf2.seek(0)"""
            html_view = plotting.view_img(
                target_img,cmap='coolwarm',draw_cross=False, annotate=True,symmetric_cmap=True)
        st.components.v1.html(html_view._repr_html_(),height=260,width=800)
        #st.image(buf2, caption="Phenotype in Human Space",  use_container_width = True)
        #get_nii_download_button(target_img, "Target_Data.nii.gz")
        plot_flag = True
    else:
        st.markdown("### Visualization of Input Data")
        st.markdown("You can drag the image to display the slice you want.")
        with st.spinner('Rendering...'):
            data_df.reset_index(inplace=True)
            data_df.columns = [['Anatomical Name','Phenotype']]
            source_img = map_phenotype_to_nifti(data_df, human_atlas)
            #buf2 = BytesIO()
            #display2 = plotting.plot_stat_map(source_img,cmap='coolwarm',cut_coords=(-20,-10,10))
            #plt.savefig(buf2, format='png', bbox_inches='tight')
            #plt.close()
            #buf2.seek(0)
            html_view = plotting.view_img(
                source_img,cmap='coolwarm',draw_cross=False, annotate=True,symmetric_cmap=True)
        st.components.v1.html(html_view._repr_html_(),height=260,width=800)
        #st.image(buf2, caption="Phenotype in Human Space",  use_container_width = True)
        #get_nii_download_button(source_img, "Source_Data.nii.gz")

        st.markdown("### Visualization of Output Data")
        with st.spinner('Rendering...'):
            result_df.reset_index(inplace=True)
            result_df.columns = [['Anatomical Name','Phenotype']]
            target_img = map_phenotype_to_nifti(result_df, mouse_atlas)
            #buf = BytesIO()
            #display = plotting.plot_stat_map(
            #    target_img, bg_img=mouse_template, display_mode='y',
            #    cut_coords=range(-4, 3, 1), cmap='coolwarm',
            #    draw_cross=False, annotate=True)
            #plt.savefig(buf, format='png', bbox_inches='tight')
            #plt.close()
            #buf.seek(0)
            padded_target_img = pad_nifti_image(target_img, pad_width=30)
            padded_bg_img = pad_nifti_image(mouse_template, pad_width=30)
            padded_source_img = downsample_img_by_factor(padded_target_img, factor=3)
            padded_bg_img = downsample_img_by_factor(padded_bg_img, factor=3)

            html_view = plotting.view_img(
                padded_target_img, bg_img=padded_bg_img,
                cmap='coolwarm', draw_cross=False, annotate=True, symmetric_cmap=True)
        st.components.v1.html(html_view._repr_html_(),height=260,width=800)
        #st.image(buf, caption="Phenotype in Mouse Space",  use_container_width = True)
        #get_nii_download_button(target_img, "Target_Data.nii.gz")
        plot_flag = True


if plot_flag==True:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp1:
            nib.save(source_img, tmp1.name)
            zip_file.write(tmp1.name, arcname="Source_Data.nii.gz")

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp2:
            nib.save(target_img, tmp2.name)
            zip_file.write(tmp2.name, arcname="Target_Data.nii.gz")

        zip_file.writestr(output_filename, csv_bytes)

    zip_buffer.seek(0)

    st.download_button(
    label="📦 Download all results as ZIP",
    data=zip_buffer,
    file_name="Mapping_Results.zip",
    mime="application/zip")
import os

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
atlas_folder = os.path.join(dir_path, 'atlas')
graph_embeddings_folder = os.path.join(dir_path, 'graphembeddings')


class Config:
    """Centralized configuration for TransBrain with minimal structure"""
    
    # Atlas paths
    bnatlas_path = os.path.join(atlas_folder, 'bn_atlas_2mm_symmetry.nii.gz')
    bnatlas_label_path = os.path.join(atlas_folder, 'bn_atlas.csv')
    dkatlas_path = os.path.join(atlas_folder, 'dk_atlas_2mm_symmetry.nii.gz')
    dkatlas_label_path = os.path.join(atlas_folder, 'dk_atlas.csv')
    aalatlas_path = os.path.join(atlas_folder, 'aal_atlas_2mm_symmetry.nii.gz')
    aalatlas_label_path = os.path.join(atlas_folder, 'aal_atlas.csv')
    mouseatlas_path = os.path.join(atlas_folder, 'mouse_atlas.nii.gz')
    mouseatlas_label_path = os.path.join(atlas_folder, 'mouse_atlas.csv')

    atlas_resources = {
        'bn': (bnatlas_path, bnatlas_label_path),
        'dk': (dkatlas_path, dkatlas_label_path),
        'aal': (aalatlas_path, aalatlas_label_path),
        'mouse': (mouseatlas_path, mouseatlas_label_path)
    }
    
    # Graph embeddings paths
    bn_embeddings = os.path.join(graph_embeddings_folder, 'bn_p0.01_q0.1_graph_embeddings.pkl')
    dk_embeddings = os.path.join(graph_embeddings_folder, 'dk_p0.01_q0.1_graph_embeddings.pkl')
    aal_embeddings = os.path.join(graph_embeddings_folder, 'aal_p0.01_q0.1_graph_embeddings.pkl')

    embeddings_resources = {
        'bn': bn_embeddings,
        'dk': dk_embeddings,
        'aal': aal_embeddings,
    }
    
    # Region definitions
    BN_CORTICAL = (
        'A8m', 'A8dl', 'A9l','A6dl', 'A6m', 'A9m', 'A10m', 'A9/46d', 'IFJ', 'A46', 'A9/46v', 
        'A8vl', 'A6vl','A10l', 'A44d', 'IFS', 'A45c', 'A45r', 'A44op', 'A44v','A14m', 'A12/47o', 
        'A11l','A11m', 'A13', 'A12/47l','A32p','A32sg','A24cd','A24rv','A4hf', 'A6cdl', 'A4ul', 
        'A4t', 'A4tl', 'A6cvl','A1/2/3ll', 'A4ll','A1/2/3ulhf', 'A1/2/3tonIa', 'A2','A1/2/3tru',
        'A7r', 'A7c', 'A5l', 'A7pc', 'A7ip', 'A39c', 'A39rd', 'A40rd', 'A40c', 'A39rv','A40rv', 
        'A7m', 'A5m', 'dmPOS','A31','A23d','A23c','A23v','cLinG', 'rCunG','cCunG', 'rLinG', 
        'vmPOS', 'mOccG', 'V5/MT+', 'OPC', 'iOccG', 'msOccG', 'lsOccG', 'G', 'vIa', 'dIa', 
        'vId/vIg', 'dIg', 'dId','A38m', 'A41/42', 'TE1.0 and TE1.2', 'A22c', 'A38l', 'A22r', 
        'A21c','A21r', 'A37dl', 'aSTS', 'A20iv', 'A37elv', 'A20r', 'A20il', 'A37vl', 'A20cl',
        'A20cv', 'A20rv', 'A37mv', 'A37lv', 'A35/36r', 'A35/36c', 'lateral PPHC', 'A28/34', 
        'TH','TI','rpSTS','cpSTS'
    )
    
    BN_SUBCORTICAL = (
        'mAmyg', 'lAmyg', 'CA1', 'CA4DG', 'CA2CA3', 'subiculum','Claustrum', 
        'head of caudate', 'body of caudate', 'Putamen','posterovemtral putamen', 
        'nucleus accumbens','external segment of globus pallidus',
        'internal segment of globus pallidus', 'mPMtha', 'Stha','cTtha', 'Otha',
        'mPFtha','lPFtha','rTtha', 'PPtha'
    )
    
    DK_CORTICAL = (
        'caudalanteriorcingulate','rostralanteriorcingulate','caudalmiddlefrontal',
        'rostralmiddlefrontal','superiorfrontal','lateralorbitofrontal','medialorbitofrontal',
        'parsopercularis','parsorbitalis','parstriangularis','precentral','postcentral',
        'inferiorparietal','superiorparietal','supramarginal','paracentral','posteriorcingulate',
        'isthmuscingulate','pericalcarine','cuneus','lingual','lateraloccipital','insula',
        'transversetemporal','superiortemporal','middletemporal','inferiortemporal','temporalpole',
        'parahippocampal','entorhinal','fusiform'
    )
    
    DK_SUBCORTICAL = (
        'Amygdala','Hippocampus','Accumbens-area','Putamen','Caudate','Pallidum','Thalamus-Proper'
    )
    
    AAL_CORTICAL = (
        'CingulateAnt', 'CingulateMid', 'FrontalSupMedial','FrontalSup2','FrontalMid2','OFCmed',
        'OFCant','OFCpost','OFClat','FrontalMedOrb','FrontalInfOrb2','FrontalInfOper',
        'FrontalInfTri','RolandicOper','SuppMotorArea','Precentral','Postcentral','ParietalSup',
        'ParietalInf','SupraMarginal','Angular','ParacentralLobule','Precuneus','CingulatePost',
        'Calcarine','Cuneus','Lingual','OccipitalSup','OccipitalMid','OccipitalInf','Insula',
        'Heschl','TemporalSup','TemporalMid','TemporalInf','TemporalPoleSup','TemporalPoleMid',
        'ParaHippocampal','Fusiform'
    )
    
    AAL_SUBCORTICAL = ('Amygdala','Hippocampus','Caudate','Putamen','Pallidum','Thalamus')
    
    MOUSE_CORTICAL = (
        'ACAd', 'ACAv', 'PL','ILA', 'ORBl', 'ORBm', 'ORBvl','MOp','MOs','SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m',
        'SSp-ul', 'SSp-tr', 'SSp-un','SSs','PTLp','RSPagl','RSPd', 'RSPv','VISpm','VISp','VISal','VISam','VISl',
        'VISpl','AId','AIp','AIv','GU','VISC','TEa', 'PERI', 'ECT','AUDd', 'AUDp','AUDpo', 'AUDv'
    )
    
    MOUSE_SUBCORTICAL = (
        'LA', 'BLA', 'BMA', 'PA','CA1', 'CA2', 'CA3', 'DG', 'SUB', 'ACB', 'CP', 'FS', 'SF', 'SH','sAMY', 
        'PAL', 'VENT', 'SPF', 'SPA', 'PP', 'GENd', 'LAT', 'ATN','MED', 'MTN', 'ILM', 'GENv', 'EPI', 'RT'
    )
    
    region_resources = {
        'bn': (list(BN_CORTICAL), list(BN_SUBCORTICAL)),
        'dk': (list(DK_CORTICAL), list(DK_SUBCORTICAL)),
        'aal': (list(AAL_CORTICAL), list(AAL_SUBCORTICAL)),
        'mouse': (list(MOUSE_CORTICAL), list(MOUSE_SUBCORTICAL))}
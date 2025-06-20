import os
import torch
import numpy as np
import pandas as pd
import logging
from logging import NullHandler
from pathlib import Path
from collections import defaultdict
from models import ClassifierModule,create_dataloader
from dataloader import generate_matrix
from config.logger import get_logger

from sklearn.metrics import confusion_matrix

try:
    logger = get_logger() 
except:  
    logger = logging.getLogger(__name__)
    logger.addHandler(NullHandler())  
    logger.propagate = False  

def train_test_split(dataset, valid_size, sample_id, all_id):
    """
    Stratified train-validation split based on brain regions
    
    Parameters:
        dataset : numpy array of shape (n_samples, n_features+1)
                  Features matrix concatenated with labels in the last column
        valid_size : float (0.0-1.0)
                  Proportion of data to allocate for validation
        sample_id : set
                  Collection of sample identifiers in 'region_sampleX' format
        all_id : list
                  Full identifiers in 'region_sampleX_scellY' format
    
    Returns:
        X_train, X_val, y_train, y_val : split arrays matching sklearn's train_test_split format
    """

    brain_sample_map = defaultdict(set)  
    sample_allid_map = defaultdict(list)  
    
    for idx, full_id in enumerate(all_id):
        parts = full_id.split('_')
        if len(parts) < 2:
            continue  
        
        brain_region = parts[0]
        sample_key = f"{parts[0]}_{parts[1]}"
        
        if sample_key in sample_id:
            brain_sample_map[brain_region].add(sample_key)
            sample_allid_map[sample_key].append(idx)
    
    val_indices = []
    for brain_region, samples in brain_sample_map.items():
        sample_list = list(samples)
        n_samples = len(sample_list)

        n_val = max(1, int(np.round(n_samples * valid_size)))

        selected_samples = np.random.choice(sample_list, size=n_val, replace=False)
        
        for sample in selected_samples:
            val_indices.extend(sample_allid_map[sample])
    
    mask = np.zeros(dataset.shape[0], dtype=bool)
    mask[val_indices] = True

    X = dataset[:, :-1]  
    y = dataset[:, -1]   
    
    X_train = X[~mask] 
    X_val = X[mask]     
    y_train = y[~mask]  
    y_val = y[mask]
    
    return X_train, X_val, y_train, y_val

def generate_average_data(data_path,params):
    """
    Generate and process transcriptional dataset with region averaging
    """
    features, labels, region_ids, _ = generate_matrix(data_path, params)
        
    combined_data = np.hstack([features, labels.reshape(-1,1)])
        
    unique_regions = np.unique(combined_data[:, -1])
    averaged_data = np.array([
        np.append( 
            np.mean(combined_data[combined_data[:,-1] == region, :-1], axis=0),
            region
        ) for region in unique_regions])
        
    return averaged_data, np.unique(region_ids)

def extract_embeddings(trans_data, nn_model):
    """
    Extract deep learning embeddings from transform data
    """
    data_loader = create_dataloader(dataset=trans_data,batch_size=1,shuffle=False)
        
    embeddings = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.cuda().float()
            _, latent_emb = nn_model(inputs)
            embeddings.append(latent_emb.cpu().numpy())
                
    return np.concatenate(embeddings)

def create_aligned_dataframe(embeddings, region_ids):
    """Create region-aligned feature dataframe"""
    return (
            pd.DataFrame(embeddings, columns=map(str, range(1, embeddings.shape[1]+1)))
            .assign(region_id=region_ids)
            .set_index('region_id')
            .loc[region_ids] 
        )

def calculate_homology_rank(df_corr,mouse_regions, human_regions):

    region_pairs = list(zip(mouse_regions, human_regions))

    ranks = []
    for mouse_reg, human_reg in region_pairs:
        sorted_series = df_corr[mouse_reg].sort_values(ascending=False)
        rank = sorted_series.index.get_loc(human_reg) + 1
        ranks.append(rank)

    return np.mean(ranks)

def calculate_rank(params=None, model=None):
    """
    Calculate cross-species embedding correlation ranks between human and mouse embeddings.
    
    Args:
        params: Configuration parameters
        model: Trained neural network model for embedding extraction
    
    Returns:
        float: Average homology rank across all region pairs
    """
    
    human_data, human_regions = generate_average_data(params.data_files['human_trans_path'])
    mouse_data, mouse_regions = generate_average_data(params.data_files['mouse_trans_path'])

    human_embeddings = extract_embeddings(human_data, model)
    mouse_embeddings = extract_embeddings(mouse_data, model)

    human_df = create_aligned_dataframe(human_embeddings, human_regions)
    mouse_df = create_aligned_dataframe(mouse_embeddings, mouse_regions)

    homology_map = pd.read_csv(params.data_files['human_mouse_path'])
    correlation_matrix = np.corrcoef(human_df, mouse_df)[len(human_df):, :len(human_df)]
    correlation_df=pd.DataFrame(correlation_matrix.T,index=homology_map['human_region'],columns=homology_map['mouse_region'])

    return calculate_homology_rank(correlation_df,mouse_regions=homology_map['mouse_region'],human_regions=homology_map['human_region'])

def transform_human_mouse(params) -> None:
    """
    Transform human and mouse biological data using trained embedding models.
    
    Args:
        params: Configuration object containing data paths and experiment parameters
    """

    human_data, human_regions = generate_average_data(params.data_files['human_trans_path'])
    mouse_data, mouse_regions = generate_average_data(params.data_files['mouse_trans_path'])

    for data_id in range(len(params.data_files['data_path'])):

        data_path = params.data_files['data_path'][data_id]
        data_label = Path(data_path).stem.split("_")[-1]

        for i in range(params.repeat_n):

            net = ClassifierModule(input_units = params.input_units,
                        output_units = params.output_units,
                        hidden_units1 = params.hidden_units1,
                        hidden_units2 = params.hidden_units2,
                        hidden_units3 = params.hidden_units3,
                        mode='Transform').cuda()
            
            model_dir = Path(params.data_files["save_path"]) / f"Data_{data_label}/Repeat_{i}"
            
            if params.use_bestrank == True:
                modelpath = os.path.join(model_dir,'Best_rank_epoch.pth')
            elif params.use_bestvalid == True:
                modelpath = os.path.join(model_dir,'Best_validloss_epoch.pth')
            else:
                modelpath = os.path.join(model_dir,'Last_epoch.pth')

            if os.path.exists(modelpath):
                logger.info(f"Transform model: {model_dir}")
                net.load_state_dict(torch.load(modelpath))
            else:
                logger.warning(f"Transform model: {modelpath} not exist")
                continue

            human_embeddings = extract_embeddings(human_data, net.eval())
            mouse_embeddings = extract_embeddings(mouse_data, net.eval())

            human_df = create_aligned_dataframe(human_embeddings, human_regions)
            mouse_df = create_aligned_dataframe(mouse_embeddings, mouse_regions)

            save_path = os.path.join(model_dir,'Results')
            os.makedirs(save_path,exist_ok=True)
            human_df.to_csv(os.path.join(save_path,'human_embedding.csv'))
            mouse_df.to_csv(os.path.join(save_path,'mouse_embedding.csv'))

def independent_test(params) -> None:
    """
    Execute model evaluation on independent test datasets.
    
    Args:
        params: Configuration object containing data paths and experiment parameters
    """

    X, y, labels, _ = generate_matrix(params.data_files['independent_test_path'], params)
    test_loader = create_dataloader(np.concatenate((X, y), axis=1), params.batch_size, shuffle=False)

    for data_path in params.data_files['independent_data_path']:
        data_label = Path(data_path).stem.split("_")[-1]
        
        for repeat in range(params.repeat_n):

            model_dir = Path(params.data_files["indepedent_s_path"]) / f"Data_{data_label}/Repeat_{repeat}"
            save_path = model_dir / 'Results'
            model_path = model_dir / ('Best_rank_epoch.pth' if params.use_bestrank else 
                                    'Best_validloss_epoch.pth' if params.use_bestvalid else 
                                    'Last_epoch.pth')

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
                
            model = ClassifierModule(
                input_units=params.input_units,
                output_units=params.output_units,
                hidden_units1=params.hidden_units1,
                hidden_units2=params.hidden_units2,
                hidden_units3=params.hidden_units3,
                mode='Transform'
            ).cuda().eval()
            model.load_state_dict(torch.load(model_path))

            gt, pred = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs.cuda().float())
                    pred.extend(torch.argmax(outputs.softmax(1), 1).cpu().numpy())
                    gt.extend(targets.cpu().numpy())

            save_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'gt': gt, 'pred': pred}).to_csv(save_path/'predictions.csv', index=False)

            conf_matrix = confusion_matrix(gt, pred)
            pd.DataFrame(conf_matrix/conf_matrix.sum(1, keepdims=True),
                        index=labels,
                        columns=labels).to_csv(save_path/'confusion_matrix.csv')


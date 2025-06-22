import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from pipeline.code.dnn.dataloader import create_dataloader
from pipeline.code.dnn.dataloader import generate_matrix
from pipeline.code.dnn.config.logger import get_logger
from pipeline.code.dnn.models import ClassifierModule
from sklearn.metrics import confusion_matrix


def train_test_split(dataset, valid_size, sample_id, all_id):
    """
    Stratified train-validation split based on brain regions.

    This function performs a stratified train-validation split by brain region.

    Parameters
    ----------
    dataset : ndarray of shape (n_samples, n_features + 1)
        Feature matrix concatenated with the target labels in the last column.
    valid_size : float
        Proportion (between 0 and 1) of samples per brain region to allocate to the validation set.
    sample_id : set of str
        Collection of sample identifiers in the format 'region_sampleX'.
    all_id : list of str
        Full identifiers in the format 'region_sampleX_scellY', corresponding to each row in `dataset`.

    Returns
    -------
    train : ndarray
        Training set after splitting.
    val : ndarray
        Validation set after splitting.
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

    train = dataset[~mask] 
    val = dataset[mask]
    
    return train, val

def generate_average_data(data_path,params):
    """
    Generate and process transcriptional dataset with region averaging.

    This function loads transcriptional data and averages the features across
    brain regions. Each region's data is collapsed into a single feature vector
    representing the mean across all samples from that region.

    Parameters
    ----------
    data_path : str
        Path to the dataset file containing transcriptional features and metadata.
    params : object
        Configuration object containing options for data loading and preprocessing.

    Returns
    -------
    averaged_data : ndarray of shape (n_regions, n_features + 1)
        Matrix where each row represents a brain region. The last column contains
        the region label, and preceding columns are the averaged features.
    unique_regions : ndarray
        Array of unique region identifiers.
    """
    features, labels, region_ids, _, _ = generate_matrix(data_path, params)

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
    Extract DNN embeddings.

    This function passes the data through a trained neural network model
    to extract latent feature embeddings.

    Parameters
    ----------
    trans_data : numpy.ndarray 
        Processed data to extract embeddings from.
    nn_model : torch.nn.Module
        A trained neural network model used to extract embeddings. 

    Returns
    -------
    embeddings : ndarray of shape (n_samples, embedding_dim)
        Concatenated latent embeddings extracted from the model.
    """
    data_loader = create_dataloader(dataset=trans_data,batch_size=1,shuffle=False,drop_last=True)
        
    embeddings = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.cuda().float()
            try:
                _,latent_emb = nn_model(inputs)
                embeddings.append(latent_emb.cpu().numpy())
            except:
                latent_emb = nn_model(inputs)
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
    """
    Calculate average rank of true homologous human regions in mouse-to-human correlation.

    For each mouse region, the function retrieves the correlation values with all human regions
    from the correlation DataFrame `df_corr`, ranks them in descending order, and finds the
    rank position of the true homologous human region. It returns the average of these ranks.

    Parameters
    ----------
    df_corr : pandas.DataFrame
        A square DataFrame of shape (n_human_regions, n_mouse_regions), where each column
        corresponds to a mouse region and each row to a human region. The values represent
        similarity or correlation scores.
    mouse_regions : list of str
        List of mouse region names corresponding to the columns of `df_corr`.
    human_regions : list of str
        List of prior homologous human region names, same length as `mouse_regions`.
        Each entry corresponds to the homolog of the mouse region at the same index.

    Returns
    -------
    float
        The mean rank of the prior homologous human regions in descending correlation order
        for each mouse region. A lower value indicates better alignment.
    """
    region_pairs = list(zip(mouse_regions, human_regions))

    ranks = []
    for mouse_reg, human_reg in region_pairs:
        sorted_series = df_corr[mouse_reg].sort_values(ascending=False)
        rank = sorted_series.index.get_loc(human_reg) + 1
        ranks.append(rank)

    return np.mean(ranks)

def calculate_rank(params=None, model=None,epoch=0,model_dir=None):
    """
    Calculate cross-species embedding correlation ranks between human and mouse regions.

    This function extracts embeddings for human and mouse transcriptional data using
    a trained neural network model, computes the correlation matrix between region-level
    embeddings, and calculates the average rank of prior homologous human regions for each
    mouse region.

    Parameters
    ----------
    params : object
        Configuration object containing paths to human/mouse transcriptional data and
        homology mapping file, typically accessed via `params.data_files`.
    model : torch.nn.Module
        Trained deep learning model used to extract embeddings from the input data.
    epoch : int, optional
        Epoch index used.
    model_dir : str, optional
        Directory path to save the embedding results.

    Returns
    -------
    float
        The average rank of prior homologous human regions for each mouse region in terms
        of correlation with the extracted embeddings. Lower values indicate better cross-species
        alignment.
    """
    
    human_data, human_regions = generate_average_data(params.data_files['human_trans_path'],params)
    mouse_data, mouse_regions = generate_average_data(params.data_files['mouse_trans_path'],params)

    human_embeddings = extract_embeddings(human_data, model.eval())
    mouse_embeddings = extract_embeddings(mouse_data, model.eval())

    human_df = create_aligned_dataframe(human_embeddings, human_regions)
    mouse_df = create_aligned_dataframe(mouse_embeddings, mouse_regions)

    save_path = os.path.join(model_dir,'Results')
    os.makedirs(save_path,exist_ok=True)
    human_df.to_csv(os.path.join(save_path,'human_embedding_epoch{}.csv'.format(epoch)))
    mouse_df.to_csv(os.path.join(save_path,'mouse_embedding_epoch{}.csv'.format(epoch)))

    homology_map = pd.read_csv(params.data_files['human_mouse_path'])
    correlation_matrix = np.corrcoef(human_df, mouse_df)[len(human_df):, :len(human_df)]
    correlation_df=pd.DataFrame(correlation_matrix.T,index=human_df.index,columns=mouse_df.index)

    return calculate_homology_rank(correlation_df,mouse_regions=homology_map['mouse_region'],human_regions=homology_map['human_region'])


def transform_human_mouse(params) -> None:
    """
    Transform human and mouse data using trained embedding models.

    This function loads trained neural network models for a number of repeated experiments, 
    transforms human and mouse transcriptional data into embeddings, and saves them as CSV files.

    Parameters
    ----------
    params : object
        Configuration object containing:
            - data_files: dictionary with paths to human/mouse data and model save directories
            - repeat_n: number of repeated models
            - model hyperparameters: input_units, hidden_units1/2/3, output_units
            - use_bestrank / use_bestvalid: flags for model selection
    Returns
    -------
    None
        Saves the generated embedding CSV files.
    """
    logger = get_logger()
    human_data, human_regions = generate_average_data(params.data_files['human_trans_path'],params)
    mouse_data, mouse_regions = generate_average_data(params.data_files['mouse_trans_path'],params)

    data_label = Path(params.data_files['data_path']).stem.split("_")[-1]

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
    Evaluate the trained model on an independent test dataset.

    Loads the saved models from repeated training runs, runs inference on the test data,
    saves predictions and confusion matrix to CSV files.

    Parameters
    ----------
    params : object

        Configuration object with the following attributes:
        
        - data_files : dict
            Paths to independent test data and saved model weights.
            Expected keys include 'independent_test_path', 'independent_data_path', 'independent_s_path'.
        - repeat_n : int
            Number of repeated training runs/models to evaluate.
        - input_units, hidden_units1, hidden_units2, hidden_units3, output_units : int
            Model architecture hyperparameters.
        - use_bestrank, use_bestvalid : bool
            Flags to select which checkpoint to load for evaluation.

    Returns
    -------
    None
        Saves prediction results and confusion matrices as CSV files.
    """
    logger = get_logger()
    X, y, labels, _ ,_ = generate_matrix(params.data_files['independent_test_path'], params)
    test_loader = create_dataloader(np.concatenate((X, y), axis=1), 1, shuffle=False, drop_last=True)

    data_label = Path(params.data_files['independent_data_path']).stem.split("_")[-1]
        
    for repeat in range(params.repeat_n):

        model_dir = Path(params.data_files["independent_s_path"]) / f"Data_{data_label}/Repeat_{repeat}"
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
            mode='Test'
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
                    index=np.unique(labels),
                    columns=np.unique(labels)).to_csv(save_path/'confusion_matrix.csv')


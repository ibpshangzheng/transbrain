import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import os
import logging
logging.basicConfig(level=logging.INFO)

class SpeciesTrans:
    def __init__(self):
        """
        Initialize the SpeciesTrans class with human and mouse brain regions.
        """
        # Define ROIs of humans and mice
        self.human_select_cortex_region = ['A8m', 'A8dl', 'A9l','A6dl', 'A6m', 'A9m', 'A10m', 'A9/46d', 'IFJ', 'A46', 'A9/46v', 'A8vl', 'A6vl','A10l', 'A44d', 'IFS', 'A45c', 'A45r', 'A44op', 'A44v','A14m', 'A12/47o', 'A11l','A11m', 'A13', 'A12/47l','A32p','A32sg',
                'A24cd','A24rv','A4hf', 'A6cdl', 'A4ul', 'A4t', 'A4tl', 'A6cvl','A1/2/3ll', 'A4ll','A1/2/3ulhf', 'A1/2/3tonIa', 'A2','A1/2/3tru','A7r', 'A7c', 'A5l', 'A7pc', 'A7ip', 'A39c', 'A39rd', 'A40rd', 'A40c', 'A39rv','A40rv', 'A7m', 'A5m', 'dmPOS',
                'A31','A23d','A23c','A23v','cLinG', 'rCunG','cCunG', 'rLinG', 'vmPOS', 'mOccG', 'V5/MT+', 'OPC', 'iOccG', 'msOccG', 'lsOccG', 'G', 'vIa', 'dIa', 'vId/vIg', 'dIg', 'dId','A38m', 'A41/42', 'TE1.0 and TE1.2', 'A22c', 'A38l', 'A22r', 'A21c',
                'A21r', 'A37dl', 'aSTS', 'A20iv', 'A37elv', 'A20r', 'A20il', 'A37vl', 'A20cl','A20cv', 'A20rv', 'A37mv', 'A37lv', 'A35/36r', 'A35/36c', 'lateral PPHC', 'A28/34', 'TH','TI','rpSTS','cpSTS']
        
        self.mouse_select_cortex_region = ['ACAd', 'ACAv', 'PL','ILA', 'ORBl', 'ORBm', 'ORBvl','MOp','SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m',
       'SSp-ul', 'SSp-tr', 'SSp-un','SSs','PTLp','RSPagl','RSPd', 'RSPv','VISpm','VISp','VISal','VISl','VISpl','AId','AIp','AIv','GU','VISC','TEa', 'PERI', 'ECT','AUDd', 'AUDp',
       'AUDpo', 'AUDv']
        
        self.human_select_subcortex_region = ['mAmyg', 'lAmyg', 'CA1', 'CA4DG', 'CA2CA3', 'subiculum','Claustrum', 'head of caudate', 'body of caudate', 'Putamen',
       'posterovemtral putamen', 'nucleus accumbens','external segment of globus pallidus','internal segment of globus pallidus', 'mPMtha', 'Stha','cTtha', 'Otha',
        'mPFtha','lPFtha','rTtha', 'PPtha']
        
        self.mouse_select_subcortex_region = ['LA', 'BLA', 'BMA', 'PA','CA1', 'CA2', 'CA3', 'DG', 'SUB', 'ACB', 'CP', 'FS', 'SF', 'SH','sAMY', 'PAL', 'VENT', 'SPF', 'SPA', 'PP', 'GENd', 'LAT', 'ATN',
       'MED', 'MTN', 'ILM', 'GENv', 'EPI', 'RT']
        
        # Combine cortex and subcortex regions
        self.Human_select_region = self.human_select_cortex_region + self.human_select_subcortex_region
        self.Mouse_select_region = self.mouse_select_cortex_region + self.mouse_select_subcortex_region
        
        # Load graph embeddings
        logging.info('SpeciesTrans initialized')
        self.Human_Mouse_embedding = self._load_graph_embeddings()
        



    def _load_graph_embeddings(self):
        """
        Load the graph embeddings from the specified file.

        Returns:
            numpy.ndarray: The graph embeddings.
        """
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        logging.info('Loading graph embeddings.')
        with open(os.path.join(dir_path, 'Graphembeddings', 'Human_Mouse_p0.01_q0.1_graph_embeddings.pkl'), 'rb') as f:
            return pickle.load(f)



    def dual_mapping_data(self, pheno_data, source_matrix, target_matrix, normalize_input=False, restore_output=False):
        """
        Perform dual mapping between source and target domain matrix using linear regression.

        Parameters:
        
            pheno_data (numpy.ndarray): Specific phenotype data. Each row represents a region.
            source_matrix (pandas.DataFrame): Source species graph embedding matrix.
            target_matrix (pandas.DataFrame): Target species graph embedding matrix.
            normalize_input (bool): Whether to apply normalization to the input data. Default is False.
            restore_output (bool): Whether to restore the output predictions to the original scale. Default is False.

        Returns:
            numpy.ndarray: Phenotypes transformed to target species.
        """
        if restore_output and not normalize_input:
            raise ValueError("Unknown raw distribution, restore_output can only be set True when normalize_input is True")
        
        source_matrix = source_matrix.values
        target_matrix = target_matrix.values

        # Reshape and normalize the phenotype data
        min_max_scaler = MinMaxScaler()
        pheno_y = pheno_data.reshape(-1, 1)

        if normalize_input:
            pheno_y_normalized = min_max_scaler.fit_transform(pheno_y)
        else:
            pheno_y_normalized = pheno_y
        pheno_y_normalized = pheno_y_normalized.ravel()
        
        # Train a linear regression model
        model = LinearRegression()
        model.fit(source_matrix, pheno_y_normalized)
        
        # Predict the target data
        y_target = model.predict(target_matrix)
        
        # Restore the predicted values if needed
        if restore_output and normalize_input:
            y_target = min_max_scaler.inverse_transform(y_target.reshape(-1, 1)).ravel()
        
        return y_target



    def mouse_to_human(self, mouse_phenotype, region_type='cortex', normalize_input=True, restore_output=False):
        """
        Translate mouse phenotype to human phenotype.

        Parameters:
        
            mouse_phenotype (pandas.DataFrame): DataFrame containing mouse phenotype data.
            
            index|phenotype1|phenotype2|...
            
            region_type (str): Type of regions to trans. Options: 'cortex', 'subcortex', 'all'. Default is 'cortex'.
            normalize_input (bool): Whether to apply normalization to the input data. Default is False.
            restore_output (bool): Whether to restore the output predictions to the original scale. Default is False.

        Returns:
            pandas.DataFrame: DataFrame containing human phenotype predictions.
        """
        if region_type not in ['cortex', 'subcortex', 'all']:
            raise ValueError("Invalid region_type. Choose in 'cortex', 'subcortex', or 'all'.")

        if region_type == 'cortex':
            mouse_phenotype = mouse_phenotype.T[self.mouse_select_cortex_region].T
        elif region_type == 'subcortex':
            mouse_phenotype = mouse_phenotype.T[self.mouse_select_subcortex_region].T
        elif region_type == 'all':
            mouse_phenotype = mouse_phenotype.T[self.Mouse_select_region].T

        graph_trans_dict = {}
        for key in mouse_phenotype.columns:
            mouse_array = np.asarray([mouse_phenotype[key].values])
            if region_type == 'cortex':
                expanded_array = np.pad(mouse_array, ((0, 0), (0, len(self.mouse_select_subcortex_region))), mode='constant', constant_values=0)
            elif region_type == 'subcortex':
                expanded_array = np.pad(mouse_array, ((0, 0), (len(self.mouse_select_cortex_region), 0)), mode='constant', constant_values=0)
            elif region_type == 'all':
                expanded_array = mouse_array
            else:
                raise ValueError("Invalid region_type. Must be 'cortex', 'subcortex', or 'all'.")

            graph_trans = []
            for i in range(self.Human_Mouse_embedding.shape[0]):
                Human_matrix = self.Human_Mouse_embedding[i][:len(self.Human_select_region)]
                Mouse_matrix = self.Human_Mouse_embedding[i][len(self.Human_select_region):]
                Human_matrix = pd.DataFrame(Human_matrix)
                Mouse_matrix = pd.DataFrame(Mouse_matrix)
                
                transed_data = self.dual_mapping_data(
                    expanded_array, 
                    Mouse_matrix, 
                    Human_matrix, 
                    normalize_input=normalize_input, 
                    restore_output=restore_output
                )
                graph_trans.append(transed_data)
            
            graph_trans_mean = np.mean(np.asarray(graph_trans), axis=0)
            graph_trans_dict[key] = graph_trans_mean

        if region_type == 'cortex':
            human_phenotype = np.asarray([graph_trans_dict[key][:len(self.human_select_cortex_region)] for key in graph_trans_dict.keys()])
            human_phenotype = pd.DataFrame(human_phenotype.T, index=self.human_select_cortex_region, columns=mouse_phenotype.columns)
        elif region_type == 'subcortex':
            human_phenotype = np.asarray([graph_trans_dict[key][len(self.human_select_cortex_region):] for key in graph_trans_dict.keys()])
            human_phenotype = pd.DataFrame(human_phenotype.T, index=self.human_select_subcortex_region, columns=mouse_phenotype.columns)
        elif region_type == 'all':
            human_phenotype = np.asarray([graph_trans_dict[key] for key in graph_trans_dict.keys()])
            human_phenotype = pd.DataFrame(human_phenotype.T, index=self.Human_select_region, columns=mouse_phenotype.columns)

        logging.info('Mouse '+ region_type + ' phenotypes have been successfully transformed to human!')

        return human_phenotype



    def human_to_mouse(self, human_phenotype, region_type='cortex', normalize_input=True, restore_output=False):
        """
        Translate human phenotype to mouse phenotype.

        Parameters:
        
            human_phenotype (pandas.DataFrame): DataFrame containing human phenotype data.
            
            index|phenotype1|phenotype2|...
            
            region_type (str): Type of regions to trans. Options: 'cortex', 'subcortex', 'all'. Default is 'cortex'.
            normalize_input (bool): Whether to apply normalization to the input data. Default is False.
            restore_output (bool): Whether to restore the output predictions to the original scale. Default is False.

        Returns:
            pandas.DataFrame: DataFrame containing mouse phenotype predictions.
        """
        if region_type not in ['cortex', 'subcortex', 'all']:
            raise ValueError("Invalid region_type. Choose in 'cortex', 'subcortex', or 'all'.")

        if region_type == 'cortex':
            human_phenotype = human_phenotype.T[self.human_select_cortex_region].T
        elif region_type == 'subcortex':
            human_phenotype = human_phenotype.T[self.human_select_subcortex_region].T
        elif region_type == 'all':
            human_phenotype = human_phenotype.T[self.Human_select_region].T
        
        graph_trans_dict = {}
        for key in human_phenotype.columns:
            human_array = np.asarray([human_phenotype[key].values])
            if region_type == 'cortex':
                expanded_array = np.pad(human_array, ((0, 0), (0, len(self.human_select_subcortex_region))), mode='constant', constant_values=0)
            elif region_type == 'subcortex':
                expanded_array = np.pad(human_array, ((0, 0), (len(self.human_select_cortex_region), 0)), mode='constant', constant_values=0)
            elif region_type == 'all':
                expanded_array = human_array
            else:
                raise ValueError("Invalid region_type. Must be 'cortex', 'subcortex', or 'all'.")

            graph_trans = []
            for i in range(self.Human_Mouse_embedding.shape[0]):
                Human_matrix = self.Human_Mouse_embedding[i][:len(self.Human_select_region)]
                Mouse_matrix = self.Human_Mouse_embedding[i][len(self.Human_select_region):]
                Human_matrix = pd.DataFrame(Human_matrix)
                Mouse_matrix = pd.DataFrame(Mouse_matrix)
                
                transed_data = self.dual_mapping_data(
                    expanded_array, 
                    Human_matrix, 
                    Mouse_matrix, 
                    normalize_input=normalize_input, 
                    restore_output=restore_output
                )
                graph_trans.append(transed_data)
            
            graph_trans_mean = np.mean(np.asarray(graph_trans), axis=0)
            graph_trans_dict[key] = graph_trans_mean

        if region_type == 'cortex':
            mouse_phenotype = np.asarray([graph_trans_dict[key][:len(self.mouse_select_cortex_region)] for key in graph_trans_dict.keys()])
            mouse_phenotype = pd.DataFrame(mouse_phenotype.T, index=self.mouse_select_cortex_region, columns=human_phenotype.columns)
        elif region_type == 'subcortex':
            mouse_phenotype = np.asarray([graph_trans_dict[key][len(self.mouse_select_cortex_region):] for key in graph_trans_dict.keys()])
            mouse_phenotype = pd.DataFrame(mouse_phenotype.T, index=self.mouse_select_subcortex_region, columns=human_phenotype.columns)
        elif region_type == 'all':
            mouse_phenotype = np.asarray([graph_trans_dict[key] for key in graph_trans_dict.keys()])
            mouse_phenotype = pd.DataFrame(mouse_phenotype.T, index=self.Mouse_select_region, columns=human_phenotype.columns)

        logging.info('Human '+ region_type + ' phenotypes have been successfully transformed to mouse!')

        return mouse_phenotype

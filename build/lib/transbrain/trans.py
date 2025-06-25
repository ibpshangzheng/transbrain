import numpy as np
import pandas as pd
import pickle
import logging
from typing import Literal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from transbrain.config import Config
import copy

logging.basicConfig(level=logging.INFO)
RegionType = Literal['cortex', 'subcortex', 'all']

class SpeciesTrans:
    """
    Transfer phenotypes between species using graph embeddings.

    Parameters
    ----------
    atlas_type : {'bn', 'dk', 'aal'}, optional
        The type of human atlas used for initialization.
        
        - 'bn'  : Brainnetome Atlas
        - 'dk'  : Desikan-Killiany Atlas
        - 'aal' : Automated Anatomical Labeling (AAL) Atlas

        Default is 'bn'.

    Attributes
    ----------
    atlas_type : str
        The selected atlas type.
    regions : dict
        Dictionary containing human and mouse brain regions (cortex, subcortex, all).
    embeddings : np.ndarray
        Loaded graph embeddings used for phenotype translation.
    """
    
    def __init__(self, atlas_type: str = 'bn'):
        self.atlas_type = atlas_type
        self.regions = self._load_region_data()
        self.embeddings = self._load_embeddings()
        logging.info(f'Initialized for {atlas_type} atlas.')

    def _load_region_data(self) -> dict:
        h_cortex, h_subcortex = Config.region_resources[self.atlas_type]
        m_cortex, m_subcortex = Config.region_resources['mouse']
        return {
            'human': {'cortex': h_cortex, 'subcortex': h_subcortex, 'all': h_cortex + h_subcortex},
            'mouse': {'cortex': m_cortex, 'subcortex': m_subcortex, 'all': m_cortex + m_subcortex}
        }
    
    def _fill_region_values(self, df):
        """
        Fill dropped regions using the mean value of regions from the same hierarchical level.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to fill.

        Returns
        -------
        pd.DataFrame
            DataFrame with values filled.
        """
        regions_to_fill = {
            'MOs': ['MOp', 'SSp-ll'],
            'VISam': ['VISal', 'VISpm']
        }

        for region_name in ['MOs','VISam']:
            region_mean = df.loc[regions_to_fill[region_name]].mean(axis=0)
            df.loc[region_name] = region_mean
            
        return df

    def _load_embeddings(self) -> np.ndarray:
        """
        Load graph embeddings for phenotype translation.

        The function loads precomputed embeddings from a binary file (pickle format)
        based on the selected atlas type. These embeddings are used to map phenotypes
        between species.

        Returns
        -------
        np.ndarray
            A NumPy array containing the loaded embeddings corresponding to the
            specified atlas type.
        """
        with open(Config.embeddings_resources[self.atlas_type], 'rb') as f:
            return pickle.load(f)
    
    def _restore_values(self, prediction, phenotype_name, phenotype_data):
        """
        Restore the values to the original scale.

        Parameters
        ----------
        prediction : np.ndarray
            The predicted phenotype values after translation.
        phenotype_name : str
            The name of the phenotype type.
        phenotype_data : pd.DataFrame
            The original phenotype data used for scaling.

        Returns
        -------
        np.ndarray
            The restored phenotype values in the original scale.
        """
        # Get the original min and max values of the phenotype data
        original_min = phenotype_data[phenotype_name].min()
        original_max = phenotype_data[phenotype_name].max()

        # First, map the prediction to [0, 1] range
        prediction_min = prediction.min()
        prediction_max = prediction.max()
        normalized_prediction = (prediction - prediction_min) / (prediction_max - prediction_min)
        
        # Then, reverse the normalization to the original scale [original_min, original_max]
        restored_prediction = normalized_prediction * (original_max - original_min) + original_min

        return restored_prediction

    def _dual_mapping(self, pheno_data: np.ndarray, source_matrix: np.ndarray, 
                     target_matrix: np.ndarray, normalize: bool = False) -> np.ndarray:

        """
        Map phenotype data from source to target space using dual regression.

        Parameters
        ----------
        pheno_data : np.ndarray
            An array of phenotype values (regions,) in the source species.
        source_matrix : np.ndarray
            The embedding matrix for the source species.
        target_matrix : np.ndarray
            The embedding matrix for the target species.
        normalize : bool, optional
            Whether to normalize the phenotype values before regression. Default is False.

        Returns
        -------
        np.ndarray
            An array of predicted phenotype values in the target species.
        """
            
        y = pheno_data.reshape(-1, 1)
        scaler = MinMaxScaler() if normalize else None
        
        if normalize:
            y = scaler.fit_transform(y)
        model = LinearRegression().fit(source_matrix, y.ravel())
        prediction = model.predict(target_matrix)

        return prediction

    def _translate(self, phenotype: pd.DataFrame, direction: str, region_type: RegionType = 'cortex',
                  normalize: bool = True,restore: bool = False) -> pd.DataFrame:
        """
        Unified translation method for both directions.

        Parameters
        ----------
        phenotype : pd.DataFrame
            A DataFrame where rows are brain regions and columns are phenotype types.
        direction : {'human_to_mouse', 'mouse_to_human'}
            The translation direction.
        region_type : {'cortex', 'subcortex', 'all'}, optional
            The region subset to use for translation. Default is 'cortex'.
        normalize : bool, optional
            Whether to normalize phenotype values before translation. Default is True.
        restore : bool, optional
            Whether to inverse-transform values back to original scale. Only used if normalize is True.

        Returns
        -------
        pd.DataFrame
            Translated phenotype values in the target species, indexed by brain region name.
        """
        
        if restore and not normalize:
            raise ValueError("Restore requires normalized input.")

        if direction not in ['human_to_mouse', 'mouse_to_human']:
            raise ValueError("Invalid translation direction.")
        
        if direction == 'mouse_to_human':
            source_species, target_species = direction.split('_to_')
            source_regions = copy.deepcopy(self.regions[source_species])

            for dict_region_type, dict_regions in source_regions.items():
                #Remove the dropped regions during the graph stage due to clearly incorrect similarity.
                source_regions[dict_region_type] = [region for region in dict_regions if region not in ['MOs', 'VISam']]

            region_data = source_regions[region_type] 
            target_regions = copy.deepcopy(self.regions[target_species])
        else:
            source_species, target_species = direction.split('_to_')
            source_regions = copy.deepcopy(self.regions[source_species])
            region_data = source_regions[region_type]
            target_regions = copy.deepcopy(self.regions[target_species])

            for dict_region_type, dict_regions in target_regions.items():
                target_regions[dict_region_type] = [region for region in dict_regions if region not in ['MOs', 'VISam']] 


        phenotype = phenotype.T[region_data].T
        n_cortex = len(source_regions['cortex'])
        n_subcortex = len(source_regions['subcortex'])
        
        if region_type == 'cortex':
            padding = ((0, 0), (0, n_subcortex))
        elif region_type == 'subcortex':
            padding = ((0, 0), (n_cortex, 0))
        else:
            padding = None
        
        n_human = len(self.regions['human']['all'])
        embed_slices = {
            'mouse_to_human': (slice(n_human, None), slice(0, n_human)),
            'human_to_mouse': (slice(0, n_human), slice(n_human, None))
        }
        src_slice, tgt_slice = embed_slices[direction]
        
        results = {}
        for phenotype_name, values in phenotype.items():
            arr = np.pad(values.values[None, :], padding) if padding else values.values[None, :]
            predictions = []
            
            for emb in self.embeddings:
                src_mat = emb[src_slice]
                tgt_mat = emb[tgt_slice]
                pred = self._dual_mapping(arr, src_mat, tgt_mat, normalize)
                predictions.append(pred)
                
            results[phenotype_name] = np.mean(predictions, axis=0)
            # Restore the values if needed
            if restore and normalize:
                results[phenotype_name] = self._restore_values(results[phenotype_name], phenotype_name, phenotype)
    
        
        if region_type == 'cortex':
            results = {k: v[:len(target_regions['cortex'])] for k, v in results.items()}
            index = target_regions['cortex']
        elif region_type == 'subcortex':
            results = {k: v[-len(target_regions['subcortex']):] for k, v in results.items()}
            index = target_regions['subcortex']
        else:
            index = target_regions['all']
        
        logging.info(f'Successfully translated {source_species} {region_type} phenotypes to {target_species}.')

        return pd.DataFrame(results, index=index)


    def mouse_to_human(self, phenotype: pd.DataFrame, region_type: RegionType = 'cortex',
                      normalize: bool = True,restore: bool = False) -> pd.DataFrame:
        
        """
        Translate mouse phenotype to human.

        Parameters
        ----------
        phenotype : pd.DataFrame
            Mouse phenotype DataFrame (regions × phenotypes).
        region_type : {'cortex', 'subcortex', 'all'}, optional
            The brain region type to translate. Default is 'cortex'.
        normalize : bool, optional
            Whether to normalize data before translation. Default is True.
        restore : bool, optional
            Whether to restore values back to original scale after translation. Only used if normalize is True.

        Returns
        -------
        pd.DataFrame
            Translated human phenotype DataFrame (regions × phenotypes).
        """
        result_df = self._translate(phenotype, 'mouse_to_human', region_type, normalize, restore)

        return result_df

    def human_to_mouse(self, phenotype: pd.DataFrame, region_type: RegionType = 'cortex',
                      normalize: bool = True,restore: bool = False) -> pd.DataFrame:

        """
        Translate human phenotype to mouse.

        Parameters
        ----------
        phenotype : pd.DataFrame
            Human phenotype DataFrame (regions × phenotypes).
        region_type : {'cortex', 'subcortex', 'all'}, optional
            The brain region type to translate. Default is 'cortex'.
        normalize : bool, optional
            Whether to normalize data before translation. Default is True.
        restore : bool, optional
            Whether to restore values back to original scale after translation. Only used if normalize is True.

        Returns
        -------
        pd.DataFrame
            Translated mouse phenotype DataFrame (regions × phenotypes).
        """
        result_df = self._translate(phenotype, 'human_to_mouse', region_type, normalize, restore)
        
        if region_type != 'subcortex':
            #Fill regions using brain areas from the same hierarchical level.
            target_regions = self.regions['mouse']
            index_all = target_regions['cortex'] if region_type == 'cortex' else target_regions['subcortex'] if region_type == 'subcortex' else target_regions['all']
            result_df = result_df.reindex(index_all, fill_value=0)
            result_df = self._fill_region_values(result_df)

        return result_df
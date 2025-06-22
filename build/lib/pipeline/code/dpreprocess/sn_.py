# -*- encoding: utf-8 -*-
'''
@File    :   preprocess_sn.py
@Author  :   shangzhengii
@Version :   2.0
@Contact :   huangshangzheng@ibp.ac.cn
'''

import warnings
warnings.filterwarnings('ignore')

import os
import anndata
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from datetime import datetime
from scipy.sparse import issparse

def setup_logger():

    logger = logging.getLogger('sn_preprocess')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    os.makedirs('./logs', exist_ok=True)
    fh = logging.FileHandler(f'./logs/preprocess_sn_{datetime.now().strftime("%Y%m%d")}.log',mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

logger = setup_logger()

class Preprocess_Sn(object):
    """
    Preprocessing pipeline for single-nucleus RNA-seq data based on the Scanpy framework.

    Implements quality control, normalization, batch correction, gene selection,
    intra-regional smoothing, and scaling, with optional saving of the processed data.

    Parameters
    ----------
    adata : AnnData, optional
        Initial AnnData object containing expression matrix.
    min_genes : int, optional
        Minimum number of genes expressed per cell (default is 200).
    min_cells : int, optional
        Minimum number of cells a gene must be expressed in (default is 3).
    total_UMIs : int, optional
        Minimum total UMI count required per cell (default is 800).
    log_base : float, optional
        Base of logarithm used in log transformation (default is 2).
    target_sum : float, optional
        Target total count after normalization (default is 1e4).
    exclude_highly_expressed : bool, optional
        Whether to exclude highly expressed genes during normalization (default is False).
    max_fraction : float, optional
        Maximum expression fraction to define 'highly expressed' genes (default is 0.05).
    n_jobs : int, optional
        Number of threads for parallel computation (default is 60).
    regress_out : bool, optional
        Whether to regress out technical covariates (e.g., total UMIs) (default is False).
    combat : bool, optional
        Whether to perform batch effect correction using ComBat (default is False).
    num_samples : int, optional
        Number of resampling iterations for intra-region smoothing (default is None).
    max_value : float, optional
        Maximum value for scaling (clipping) of gene expression (default is 10).
    """

    def __init__(self,adata=None,min_genes=200,min_cells=3,total_UMIs=800,log_base=2,target_sum=1e4,exclude_highly_expressed=True,max_fraction=0.05,n_jobs=60,
                 regress_out=True,combat=False,num_samples=100,max_value=10):
        
        self.adata = adata
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.total_UMIs = total_UMIs
        self.log_base = log_base
        self.target_sum = target_sum
        self.exclude_highly_expressed = exclude_highly_expressed
        self.max_fraction = max_fraction
        self.n_jobs = n_jobs
        self.regress_out = regress_out
        self.combat = combat
        self.num_samples = num_samples
        self.max_value = max_value
        
    def load_data(self,file_path):

        """
        Load an `.h5ad` file into the `adata` attribute.

        Parameters
        ----------
        file_path : str
            Path to the h5ad file.

        Returns
        -------
        self
            Returns the instance to allow method chaining.
        """

        logger.info(f"Loading data from {file_path}")
        self.adata = anndata.read_h5ad(file_path)
        logger.info(f"Loaded data with shape: {self.adata.shape}")
        return self

    def dataset_qc(self):
        """
        Perform quality control filtering on the dataset.

        This function filters cells and genes based on quality metrics:
        
            - Calculates total UMIs per cell and adds it to `adata.obs`.
            - Filters out cells with fewer than `min_genes` expressed genes.
            - Filters out genes expressed in fewer than `min_cells` cells.
            - Removes cells with total UMIs below a threshold `total_UMIs`.
            - Removes genes with zero expression across all remaining cells.

        Returns
        -------
        AnnData
            The filtered AnnData object.
        """
        total_UMI_per_cell = np.sum(self.adata.X, axis=1) 
        self.adata.obs['total_UMIs'] = total_UMI_per_cell
        sc.pp.filter_cells(self.adata,min_genes=self.min_genes)
        sc.pp.filter_genes(self.adata,min_cells=self.min_cells)
        self.adata = self.adata[self.adata.obs.total_UMIs>self.total_UMIs]
        self.adata = self.adata[:,self.adata.X.sum(axis=0) > 0]
        return self.adata
    
    def dataset_normalized(self):

        """
        Normalize the single-nucleus RNA-seq dataset.

        This method performs total-count normalization to scale counts per cell 
        to a common target sum, optionally excluding highly expressed genes, 
        followed by log-transformation with a specified logarithm base.

        Returns
        -------
        AnnData
            The normalized AnnData object.
        """
       
        logger.info("Starting normalization...")
         
        sc.pp.normalize_total(self.adata, target_sum=self.target_sum,exclude_highly_expressed=self.exclude_highly_expressed,max_fraction=self.max_fraction)
        sc.pp.log1p(self.adata, base=self.log_base)

        logger.info("Normalization completed successfully")

        return self.adata
    
    def dataset_common_genes(self,gene_list: list = None):

        """
        Filter the dataset to keep only genes common to the provided gene list.

        Parameters
        ----------
        gene_list : list, optional
            List of gene names to intersect with the dataset's genes. If None,
            no filtering is applied (default is None).

        Returns
        -------
        AnnData
            The filtered AnnData object containing only the common genes.
        """

        if gene_list is None:

            logger.warning("No gene_list provided, skipping gene filtering")

            return self.adata
        
        initial_genes = self.adata.n_vars
     
        logger.info("Starting gene filtering...")

        if self.adata.var_names.is_unique == False:
            self.adata = self.adata[:, ~self.adata.var_names.duplicated()]

        if gene_list:
            sc_gene = self.adata.var.Gene.values.tolist()
            common_gene = [col for col in sc_gene if col in gene_list]
            self.adata = self.adata[:,common_gene]

        logger.info(f"Gene filtering completed. {len(common_gene)} genes retained "
                       f"({len(common_gene)/initial_genes:.1%} of original)")

        return self.adata
    
    def dataset_batch_correct(self):
        """
        Perform batch effect correction.
        This includes optional regression of total UMIs and batch correction using Combat.

        Returns
        -------
        AnnData
            The AnnData object after batch correction.
        """
        logger.info("Starting batch effect correction...")

        if self.regress_out:

            logger.info("Regressing out total UMIs...")
            sc.pp.regress_out(self.adata, ['total_UMIs'],n_jobs=self.n_jobs)
            logger.info("Total UMIs regressed out")
        
        if self.combat:

            if 'donor_id' not in self.adata.obs:
                    
                    logger.warning("'donor_id' not found in obs, skipping Combat")
            else:

                logger.info(f"Running Combat correction...")
                sc.pp.combat(self.adata, key='donor_id', covariates=None, inplace=True)
                logger.info("Combat correction completed")

        return self.adata
    
    def dataset_smooth(self):

        """
        Perform intra-regional gene expression smoothing.

        For each region, perform smoothing by averaging gene expression over random samples of cells within that region.

        Returns
        -------
        AnnData
            The smoothed AnnData object.
        """
    
        initial_shape = self.adata.shape
        n_regions = len(self.adata.obs['dissection'].unique())
    
        logger.info(f"Beginning smoothing for {n_regions} regions "
                    f"with {self.num_samples} samples")
    
        smooth_dict = {}
    
        dissection_groups = self.adata.obs.groupby('dissection').groups
    
        for region, indices in tqdm(dissection_groups.items(), 
                               desc="Smoothing regions"):
            region_data = self.adata[indices]
            n_cells = len(indices)
        
            if self.num_samples > 1:

                rng = np.random.RandomState(abs(hash(region)) % (2**32))
                samples = rng.randint(0, n_cells, size=(n_cells, self.num_samples))

                if issparse(region_data.X):
                    smoothed = np.array([
                        region_data.X[samples[i]].mean(axis=0).A1 
                        for i in range(n_cells)])
                else:
                    smoothed = np.mean(region_data.X[samples], axis=1)
            else:
                smoothed = region_data.X.toarray() if issparse(region_data.X) else region_data.X.copy()
        
            smooth_dict[region] = smoothed

        self.adata = anndata.AnnData(
            X=np.vstack(list(smooth_dict.values())),
            var=self.adata.var.copy(),
            obs=pd.DataFrame({
                'Tissue': np.repeat(list(smooth_dict.keys()), 
                                [len(v) for v in smooth_dict.values()])
            })
        )
    
        logger.info(f"Smoothing completed. Shape changed {initial_shape} -> {self.adata.shape}")

        return self.adata

    def dataset_scaled(self,cortical_regions: list = None,subcortical_regions: list = None,scale_type='split'):
        """
        Scale gene expression data, optionally splitting into cortical and subcortical regions.

        Parameters
        ----------
        cortical_regions : list, optional
            List of cortical region names to subset and scale separately.
        subcortical_regions : list, optional
            List of subcortical region names to subset and scale separately.
        scale_type : {'split', 'global'}, optional
            Scaling method to apply. 'split' scales cortical and subcortical regions separately,
            'global' scales the whole dataset at once. Default is 'split'.

        Returns
        -------
        AnnData
            The scaled AnnData object.
        """
        logger.info(f"Starting scaling (type: {scale_type})...")

        if scale_type == 'split':

            logger.info(f"Splitting into {len(cortical_regions)} cortical and "
                           f"{len(subcortical_regions)} subcortical regions")

            adata_cortical = self.adata[(self.adata.obs['Tissue'].isin(cortical_regions))]
            sc.pp.scale(adata_cortical,max_value=self.max_value)
            adata_subcortical = self.adata[(self.adata.obs['Tissue'].isin(subcortical_regions))]
            sc.pp.scale(adata_subcortical,max_value=self.max_value)
            scaled_adata = sc.concat([adata_cortical, adata_subcortical])

            self.adata =scaled_adata

        else:

            logger.info("Performing global scaling")
            sc.pp.scale(self.adata,max_value=self.max_value)

        return self.adata

    def dataset_save(self,save_path):
        """
        Save the current AnnData expression matrix to a Feather file.

        Parameters
        ----------
        save_path : str
            File path to save the Feather file.
        """
        logger.info(f"Preparing to save data to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        gene_info=self.adata.var_names
        H_region_list = self.adata.obs['Tissue'].tolist()

        sn_dataframe = pd.DataFrame(self.adata.X, columns=gene_info, dtype='float32')
        sn_dataframe['index']=H_region_list
        sn_dataframe=sn_dataframe.sample(frac=1)
        sn_dataframe.reset_index(drop=True,inplace=True)
        sn_dataframe.to_feather(save_path)
        logger.info(f"Successfully saved data to {save_path}\n"
                       f"Size: {os.path.getsize(save_path)/1024/1024:.2f} MB")
        
    def preprocess_pipeline(self, adata=None,min_genes = 200,min_cells = 3,total_UMIs = 800,log_base = 2,target_sum = 1e4,exclude_highly_expressed = True,max_fraction = 0.05,n_jobs = 60,regress_out = True,combat = False,
                            num_samples = 100,max_value = 10,dataset_path: str = None,gene_list: list = None, cortical_regions: list = None, subcortical_regions:list = None, scale_type: str='split', 
                            save_path: str = None,steps: Dict[str, bool] = None, **kwargs) -> None:

        """
        Preprocess single-nucleus RNA-seq data with flexible pipeline steps.

        Parameters
        ----------
        adata : AnnData, optional
            Input AnnData object (.h5ad) containing single-nucleus data.
            If None, data will be loaded from `dataset_path`.
        min_genes : int, default=200
            Minimum number of genes required per cell. Cells with fewer genes will be filtered out.
        min_cells : int, default=3
            Minimum number of cells in which a gene must be detected.
            Genes detected in fewer cells will be removed.
        total_UMIs : int, default=800
            Minimum total UMI counts per cell. Cells below this threshold will be filtered out.
        log_base : int or float, default=2
            Base of logarithm used for log transformation.
        target_sum : float, default=1e4
            Target sum for count normalization. After normalization, each cell's counts sum to this value.
        exclude_highly_expressed : bool, default=True
            Whether to exclude highly expressed genes during normalization to reduce technical artifacts.
        max_fraction : float, default=0.05
            Maximum fraction of counts that can come from a single gene to consider it for exclusion.
        n_jobs : int, default=60
            Number of parallel jobs for computation.
        regress_out : bool, default=True
            Whether to regress out technical covariates (e.g., total counts).
        combat : bool, default=False
            Whether to apply ComBat batch correction.
        num_samples : int, default=100
            Number of samples for downsampling/bootstrap during smoothing.
        max_value : float, default=10
            Maximum clip threshold for transformed values to avoid extreme outliers.
        dataset_path : str, optional
            Path to the single-nucleus dataset in .h5ad format, used if `adata` is None.
        gene_list : list of str, optional
            List of common genes to filter on.
        cortical_regions : list of str, optional
            List of cortical region names in the dataset.
        subcortical_regions : list of str, optional
            List of subcortical region names in the dataset.
        scale_type : {'split', 'all'}, default='split'
            Normalization method for scaling the dataset. 'split' scales regions separately,
            'all' scales all regions together.
        save_path : str, optional
            File path to save the processed data.
        steps : dict of str to bool, optional
            Dictionary controlling execution of processing steps. Keys and defaults:
                - 'qc' (bool): Quality control (default: True)
                - 'normalize' (bool): Data normalization (default: True)
                - 'filter_genes' (bool): Gene filtering (default: True)
                - 'batch_correct' (bool): Batch effect correction (default: True)
                - 'smooth' (bool): Data smoothing (default: True)
                - 'scale' (bool): Region-specific scaling (default: True)
                - 'save' (bool): Save processed data (default: True)
        **kwargs
            Additional optional parameters.

        Returns
        -------
        None

        """
        
        logger.info(f"Define preprocessing pipeline")

        try:

            default_steps = {'qc': True,'normalize': True,'filter_genes': True,'batch_correct': True,'smooth': True,'scale': True,'save': True}
            active_steps = {**default_steps, **(steps or {})}

            if adata:
                p = Preprocess_Sn(adata=adata,min_genes = min_genes,min_cells = min_cells,total_UMIs = total_UMIs, log_base = log_base,target_sum = target_sum,exclude_highly_expressed = exclude_highly_expressed,
                                  max_fraction = max_fraction,n_jobs = n_jobs,regress_out = regress_out,combat = combat,num_samples = num_samples,max_value = max_value)
            else:
                p = Preprocess_Sn(min_genes = min_genes,min_cells = min_cells,total_UMIs = total_UMIs, log_base = log_base,target_sum = target_sum,exclude_highly_expressed = exclude_highly_expressed,
                                  max_fraction = max_fraction,n_jobs = n_jobs,regress_out = regress_out,combat = combat,num_samples = num_samples,max_value = max_value).load_data(dataset_path)
                
            active_steps['qc'] and p.dataset_qc()
            active_steps['normalize'] and p.dataset_normalized()
            active_steps['filter_genes'] and gene_list and p.dataset_common_genes(gene_list)
            active_steps['batch_correct'] and p.dataset_batch_correct()
            active_steps['smooth'] and p.dataset_smooth()
            active_steps['scale'] and p.dataset_scaled(cortical_regions = cortical_regions,subcortical_regions = subcortical_regions,scale_type=scale_type)
            active_steps['save'] and p.dataset_save(save_path)

            logger.info("Preprocessing completed")
            
        except Exception as e:

            logger.error(f"Error in steps: {[k for k,v in active_steps.items() if v]}\n{e}")
            raise
        

if __name__ == '__main__':

    None
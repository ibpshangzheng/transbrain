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
        

        logger.info(f"Initialized Preprocess_Sn with params:\n"
                    f"min_genes={min_genes},min_cells={min_cells},total_UMIs={total_UMIs},log_base={log_base},target_sum={target_sum},exclude_highly_expressed={exclude_highly_expressed},max_fraction={max_fraction},n_jobs={n_jobs},regress_out={regress_out},combat={combat},num_samples={num_samples},max_value={max_value}.")

    def load_data(self,file_path):

        """
        Load single-nucleus data: h5ad format.
        """

        logger.info(f"Loading data from {file_path}")
        self.adata = anndata.read_h5ad(file_path)
        logger.info(f"Loaded data with shape: {self.adata.shape}")
        return self.adata

    def dataset_qc(self):
    
        total_UMI_per_cell = np.sum(self.adata.X, axis=1) 
        self.adata.obs['total_UMIs'] = total_UMI_per_cell
        sc.pp.filter_cells(self.adata,min_genes=self.min_genes)
        sc.pp.filter_genes(self.adata,min_cells=self.min_cells)
        self.adata = self.adata[self.adata.obs.total_UMIs>self.total_UMIs]
        self.adata = self.adata[:,self.adata.X.sum(axis=0) > 0]
        return self.adata
    
    def dataset_normalized(self):

        """
        Normalized single-nucleus dataset.
        """
        logger.info("Starting normalization...")
         
        sc.pp.normalize_total(self.adata, target_sum=self.target_sum,exclude_highly_expressed=self.exclude_highly_expressed,max_fraction=self.max_fraction)
        sc.pp.log1p(self.adata, base=self.log_base)

        logger.info("Normalization completed successfully")

        return self.adata
    
    def dataset_common_genes(self,gene_list: list = None):

        """
        gene_list: gene list for intersection.
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
        Intra-regional gene expression smoothing.
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
        Args:

             adata (AnnData, optional): Input AnnData(.h5ad) containing single-nucleus data. 
                If None, will be loaded from datasets_path.

            min_genes (int): Minimum number of genes required per cell (default: 200).
                Cells with fewer genes will be filtered out.

            min_cells (int): Minimum number of cells where a gene must be detected (default: 3). 
                Genes present in fewer cells will be removed.

            total_UMIs (int): Minimum total UMI counts per cell (default: 800).
                Cells with lower total counts will be filtered.

            log_base (int/float): Base for log transformation (default: 2).

            target_sum (float): Target sum for count normalization (default: 1e4).
                After normalization, counts per cell sum to this value.

            exclude_highly_expressed (bool): Whether to exclude highly expressed genes during 
                normalization (default: True). Helps mitigate technical artifacts.

            max_fraction (float): Maximum fraction of counts that can come from a single gene
                to be considered for exclusion (default: 0.05).

            n_jobs (int): Number of parallel jobs for computation (default: 60).

            regress_out (bool): Whether to regress out technical covariates (default: True).

            combat (bool): Whether to apply ComBat batch correction (default: False).

            num_samples (int): Number of samples for downsampling/bootstrap (default: 100).

            max_value (float): Clip threshold for transformed values (default: 10).
                Prevents extreme values after transformation.

            dataset_path (str): Path to single-nucleus dataset in h5ad format.

            gene_list (list): List of common genes for filtering.

            cortical_regions (list): List of cortical region names in the dataset.

            subcortical_regions (list): List of subcortical region names in the dataset.

            scale_type (str): Normalization method for the dataset: 'split' or 'all' (default: 'split').

            save_path (str): Output path for processed data storage.

            steps (Dict[str, bool]): Dictionary controlling execution of processing steps:
                - qc: Quality control (default: True)
                - normalize: Data normalization (default: True)
                - filter_genes: Gene filtering (default: True)
                - batch_correct: Batch effect correction (default: True)
                - smooth: Data smoothing (default: True)
                - scale: Region-specific scaling (default: True)
                - save: Save processed data (default: True)

            **kwargs: Additional optional parameters including.

        """
        
        logger.info(f"Define preprocessing pipeline")

        try:

            default_steps = {'qc': False,'normalize': False,'filter_genes': False,'batch_correct': False,'smooth': False,'scale': False,'save': False}
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
'''
@Time: 2024/02/11 09:11:30
@Author: Shangzhengii 
@Version: 1.0
@Contact: huangshangzheng@ibp.ac.cn
@Desc: Using to integrate Allen sample and single-nucleus dataset.
'''
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os

import anndata
import scanpy as sc
from tqdm import tqdm
from scipy import stats

from collections import Counter
from pipeline.code.dintegrate.data_loader import load_data
from multiprocessing import Pool


class DataIntegrator(object):

    """Single-nucleus dataset and AHBA dataset integration processor.
    
    For weighted integration of Allen Brain Atlas sample expression data
    with single-nucleus RNA-seq data to construct integrated expression matrices.
    Supports parallel processing and optimized downsampling.

    """

    def __init__(self, config: None) -> None:

        """
        Initialize the data integration processor.

        Parameters
        ----------
        config : object
            Configuration object containing parameters for integration.
        """

        self.cfg = config 

    def run(self)-> None:
        """
        Execute the integration pipeline.

        Steps:
            - Construct the cell pool if cfg.pool is True.
            - Clean the pool to remove repeatedly assigned cells.
            - Perform weighted integration over multiple iterations.
            - Generate final integrated anndata files.
        """
        if self.cfg.pool:
            args = [row for _, row in align_csv.iterrows()]
            with Pool(self.cfg.jobs) as p:
                list(tqdm(
                    p.imap(self.pool_construction, args),
                    total=len(align_csv),
                    desc='Constructing the pool......'))

        if self.cfg.integrate:
            self.clean_pool()
            regions = align_csv['brain_region'].tolist()
            for k in range(self.cfg.iterations):
                with Pool(self.cfg.jobs) as p:
                        list(tqdm(
                        p.imap(self.weighted_average, regions),
                        total=len(regions),
                        desc=f'Generating integrated dataset (round {k+1}/{self.cfg.iterations})'))

                self.generate_integrated_anndata(k_=k)
       
    def pool_construction(self,row) -> None:
        """
        Retain smooth cells with expression similarity in the top 10% of AHBA regional mean expression values
        for the sampling pool.

        Parameters
        ----------
        row : pandas.Series
            A row from the alignment CSV representing a brain region mapping.
        """
        pool_csv = pd.DataFrame()
        corr_list = []
        cell_id = []
        region_ = []

        allen_region = row['brain_region']
        sn_region = row['sn_region']
        
        Allen_ex_region = ahba_mean_ex.loc[allen_region].values
        h19_data_region = h19_data[h19_data.index.isin(sn_region.split('\\'))]
        
        for i,j in enumerate(h19_data_region.iloc[:,:-1].values):

            corr = stats.pearsonr(Allen_ex_region,j)[0]

            if corr >= thre_df.loc[allen_region,'correlation']:

                pool_csv = pool_csv.append(pd.Series(j),ignore_index=True)
                corr_list.append(corr)
                cell_id.append(h19_data_region.iloc[i,-1])
                region_.append((allen_region,sn_region))

        pool_csv['corr']=corr_list
        pool_csv['cell_id'] = cell_id
        pool_csv['sample_region'] = region_

        save_path = './{0}/{1}.csv'.format(self.cfg.data_files['pool_s_path'],allen_region)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        pool_csv.to_csv(save_path)

    def clean_pool(self) -> None:
        """
        Screening of repeatedly assigned cells.

        Cells that are assigned to more than a configured threshold number of regions
        are removed from the integration pool.
        """

        csv_paths = [
            os.path.join(self.cfg.data_files['pool_s_path'], f"{region}.csv") 
            for region in align_csv['brain_region']
        ]
        align_dataframe = pd.concat([pd.read_csv(p,index_col=0) for p in csv_paths])

        cell_counts = Counter(align_dataframe['cell_id'])
        over_assigned = [cell for cell, count in cell_counts.items() if count > self.cfg.screening_thre]
        filtered_df = align_dataframe[~align_dataframe['cell_id'].isin(over_assigned)]

        filtered_df.index = (
            filtered_df['sample_region']
            .str.split(',')
            .str[0]
            .str.replace(r"[(']", "", regex=True)
        )

        final_df = filtered_df.iloc[:, :-3]
        final_df.columns = gene_filter
        final_df.rename_axis('index', inplace=True)

        self.align_dataframe = final_df.copy()

    def weighted_average(self,brain_region):
        """
        Perform weighted integration of cell expressions from the pool with samples.

        Parameters
        ----------
        brain_region : str
            The brain region for which integration is performed.
        """

        region_sample = sample_ex.loc[[brain_region]]
        region_pool = self.align_dataframe.loc[[brain_region]]

        pool_data = self.downsample_data(region_pool) if self.cfg.downsample else region_pool
    
        results = []

        save_path = f"./{self.cfg.data_files['integrated_s_path']}/{brain_region}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for sample_idx, sample_exp in enumerate(region_sample.values):

            sample_norm = self._normalize(sample_exp)
        
            for cell_idx, cell_exp in enumerate(pool_data.values):

                cell_norm = self._normalize(cell_exp)
                integrated_name = f'{brain_region}_sample{sample_idx}_cell{cell_idx}'

                if stats.pearsonr(sample_norm, cell_norm)[0] > self.cfg.corr_threshold:
            
                    weighted_avg = self.cfg.ahba_weight * sample_norm + self.cfg.sn_weight * cell_norm
                else:
                    weighted_avg = sample_norm
                    
                results.append(pd.Series(weighted_avg, name=integrated_name))

        result_df = pd.concat(results, axis=1).T
        result_df.to_csv(save_path)

    def downsample_data(self,region_pool): 
        """
        Downsample cells from the constructed pool.

        Parameters
        ----------
        region_pool : pandas.DataFrame
            DataFrame of pooled cells for a brain region.

        Returns
        -------
        pandas.DataFrame
            Downsampled pool DataFrame.
        """

        if region_pool.values.shape[0] >= self.cfg.downsample_n:
            region_pool_sample = region_pool.sample(n=self.cfg.downsample_n)
        else:
            region_pool_sample = region_pool.copy()

        return region_pool_sample
    
    def _normalize(self, x):

        x_min, x_max = np.min(x), np.max(x)
        return 2 * (x - x_min) / (x_max - x_min) - 1
    
    def generate_integrated_anndata(self, k_=0):
        """
        Generate integrated AnnData object and save as h5ad file.

        Reads the integrated data of all brain regions, optionally performs z-score normalization,
        shuffles samples, and stores the result as an h5ad file.

        Parameters
        ----------
        k_ : int, optional
            Iteration index for output file naming (default is 0).
        """
        integrated_files = [
            f"./{self.cfg.data_files['integrated_s_path']}/{region}.csv"
            for region in align_csv['brain_region']
        ]
    
        integrated_data = pd.concat([pd.read_csv(f,index_col=0) 
                            for f in tqdm(integrated_files, desc='Loading region data')])

        if self.cfg.zscore:

            adata = anndata.AnnData(integrated_data.values)
            sc.pp.scale(adata, max_value=self.cfg.max_value)
            integrated_data = pd.DataFrame(
                adata.X,
                index=integrated_data.index,
                columns=integrated_data.columns
            )

        shuffled_data = integrated_data.sample(frac=1)
        shuffled_index = shuffled_data.index 

        integrated_adata = anndata.AnnData(
            X=shuffled_data.values, 
            obs=pd.DataFrame({'region_index': shuffled_index}), 
            var=pd.DataFrame({'Gene': gene_filter})
        )
    
        os.makedirs(self.cfg.data_files['integrated_s_path'], exist_ok=True)
        integrated_adata.write(os.path.join(self.cfg.data_files['integrated_s_path'], f'Integrated_dataset_{k_}.h5ad'))

if __name__=='__main__':

    global align_csv, ahba_mean_ex, sample_ex, h19_data, thre_df, gene_filter, config
    align_csv, ahba_mean_ex, sample_ex, h19_data, thre_df, gene_filter, config = load_data()

    # Initialize processor
    integrator = DataIntegrator(config=config)

    # Execute pipeline
    integrator.run()

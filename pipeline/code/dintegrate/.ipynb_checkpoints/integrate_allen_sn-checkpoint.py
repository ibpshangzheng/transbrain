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
import pickle
import os

import anndata
import argparse
import scanpy as sc
from tqdm import tqdm
from scipy import stats
from collections import Counter
from multiprocessing import Pool

# config

pool_type = 'cortex'
col_name = '127_region'
align_csv_path = './Files/sn_ahba_atlas_align.csv'
gene_filter_path = './Files/Common_stable_genes_{}.csv'.format(pool_type)
kernel_n=90
allen_w=0.7
sn_w=0.3
d_thre=10
s_thre=0.1


def get_args():
    
    parser=argparse.ArgumentParser(add_help=True,description='Run Aligment',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-mean_path')
    parser.add_argument('-expression_path')
    parser.add_argument('-H19_path')
    parser.add_argument('-align_s_path')
    parser.add_argument('-integrated_s_path')
    parser.add_argument('-thre_dict_path')

    args=parser.parse_args()
    args_dict = vars(args)
    args_tuple = tuple(args_dict.values()) 

    return args_tuple

Allen_expression_mean_path,sample_expression_path,H19_smooth_path,align_data_save_path,integrated_data_save_path,thre_dict_filter_gene_path = get_args()

pool_save_path = f'./Integrated_dataset/{independent_flag}/Allen{{}}_sn{{}}_align_{pool_type}'  
integrated_data_save_path = f'./Integrated_dataset/{independent_flag}/Allen{{}}_sn{{}}_integrated_{pool_type}'

data_files = {
    'align': 'sc_sample_atlas_align_{pool_type}.csv',
    'allen': 'Sample_expression_mean_9861_10021_12876_14380_15496_15697_{pool_type}.csv',
    'h19': [
        '/share/user_data/public/experiment/Human_SC_Transform/ProcessedData/Human_SC/H19.30.001_smooth100.feather',
        '/share/user_data/public/experiment/Human_SC_Transform/ProcessedData/Human_SC/H19.30.002_smooth100.feather'
    ],
    'genes': 'Allen_sample_common_sable_gene_mouse_filter_dataframe_{pool_type}.csv',
    'sample': './ex_d_sample_{pool_type}.csv',
    'thresholds': './thre_dict_filter_gene_{pool_type}.pkl'
}

def load_data(pool_type: str = 'cortex') -> tuple:
    """
    Args:
        pool_type: The cell pool type, with 'cortex' as the default.
    Returns:
        (align_csv, allen_ex, sample_ex, h19_data, thre_dict_filter, gene_filter)
    """
    align_csv = pd.read_csv(data_files['align'].format(pool_type=pool_type))
    allen_ex = pd.read_csv(data_files['allen'].format(pool_type=pool_type)).set_index('index')
    sample_ex = pd.read_csv(data_files['sample'].format(pool_type=pool_type)).set_index('index')

    h19_data = pd.concat([pd.read_feather(p).set_index('index') for p in data_files['h19']])
    
    gene_filter = pd.read_csv(data_files['genes'].format(pool_type=pool_type))['Gene'].tolist()

    h19_data = h19_data[gene_filter]
    allen_ex = allen_ex[gene_filter]
    
    if pool_type != 'all':
        is_cortex = h19_data.index.str.contains('Cerebral cortex')
        h19_data = h19_data[is_cortex] if pool_type == 'cortex' else h19_data[~is_cortex]
        h19_data['cell_id'] = range(len(h19_data))

    with open(data_files['thresholds'].format(pool_type=pool_type), 'rb') as f:
        thre_dict_filter = pickle.load(f)
    
    return align_csv, allen_ex, sample_ex, h19_data, thre_dict_filter, gene_filter


align_csv, allen_ex, sample_ex, h19_data, thre_dict_filter, gene_filter = load_data(pool_type=pool_type)

class IntegratedData(object):

    """Single-nucleus dataset and AHBA dataset integration processor.
    
    A class for weighted integration of Allen Brain Atlas sample expression data
    with single-nucleus RNA-seq data to construct integrated expression matrices.
    Supports parallel processing and optimized downsampling.

    Example:
        >>> processor = integrated(
                pool_type='cortex',
                col_name='105_region',
                pool_save='Allen{}_sn{}_align_cortex',
                integrated_save='Allen{}_sn{}_integrated_cortex',
                kernel_n=60,
                allen_w=0.7,
                sn_w=0.3
            )
        >>> processor.run(build_pool=True, integrated_data=True)
    """

    def __init__(self, pool_type: str, col_name: str, pool_save_path: str = None, integrated_data_save_path: str = None, kernel_n=90,allen_w=0.5,sn_w=0.5,d_thre=10,s_thre=0.1) -> None:

        """
         Initialize the data integration processor
        
        Args:
            pool_type: Type of cell pool ('cortex' or other subtypes)
            col_name: Column name containing region identifiers in align_csv
            pool_save_path: Format string for pool storage paths 
                          (e.g., 'Allen{}_sn{}_align_{}')
            integrated_data_save_path: Format string for integrated data storage
            kernel_n: Number of parallel workers
            allen_w: Allen data weight (0-1)
            sn_w: Single-nucleus data weight (0-1) 
            d_thre: Downsampling cell count threshold
            s_thre: Pearson correlation threshold for cell selection
        """

        self.pool_type = pool_type
        self.col_name = col_name
        self.pool_save = pool_save_path 
        self.integrated_save = integrated_data_save_path 
        self.kernel = kernel_n
        self.allen_w = allen_w
        self.sn_w = sn_w
        self.d_thre = d_thre
        self.s_thre = s_thre
       
    def pool_construction(self,arg) -> None:

        """
        Retain smooth cells with expression similarity in the top 10% of AHBA regional mean expression values for the sampling pool.

        """

        self.row=arg
        pool_csv = pd.DataFrame()
        corr_list = []
        cell_id = []
        region_ = []

        allen_region = self.row[self.col_name]

        sn_region = self.row['sn_atlas']
        h19_data_region = h19_data[h19_data.index.isin(sn_region.split('\\'))]
        Allen_ex_region = allen_ex.loc[allen_region].values
        
        for i,j in enumerate(h19_data_region.iloc[:,:-1].values):

            corr = stats.pearsonr(Allen_ex_region,j)[0]

            if corr >= thre_dict_filter[allen_region][0]:

                pool_csv = pool_csv.append(pd.Series(j),ignore_index=True)
                corr_list.append(corr)
                cell_id.append(h19_data_region.iloc[i,-1])
                region_.append((allen_region,sn_region))

        pool_csv['corr']=corr_list
        pool_csv['cell_id'] = cell_id
        pool_csv['sample_region'] = region_

        save_path = './{0}/{1}.csv'.format(self.pool_save.format(self.allen_w,self.sn_w,self.pool_type),allen_region)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        pool_csv.to_csv(save_path)


    def clean_pool(self,screening_thre=3) -> None:

        """
        Screening of repeatedly assigned cells.

        """
        csv_dir = self.pool_save.format(self.allen_w, self.sc_w, self.pool_type)
        csv_paths = [
            os.path.join(csv_dir, f"{region}.csv") 
            for region in align_csv[self.col_name]
        ]
        align_dataframe = pd.concat([pd.read_csv(p) for p in csv_paths])
        align_dataframe.drop(columns='Unnamed: 0', inplace=True)

        cell_counts = Counter(align_dataframe['cell_id'])
        over_assigned = [cell for cell, count in cell_counts.items() if count > screening_thre]
        filtered_df = align_dataframe[~align_dataframe['cell_id'].isin(over_assigned)]

        filtered_df.index = (
            filtered_df['sample_region']
            .str.split(',')
            .str[0]
            .str.replace(r"[(']", "", regex=True)
        )

        print(f"Allen align region number: {filtered_df.index.nunique()}")

        final_df = filtered_df.iloc[:, :-3]
        final_df.columns = gene_filter
        final_df.rename_axis('index', inplace=True)

        self.align_dataframe = final_df.copy()

    
    def downsample_data(self,region_pool): 

        """
        Downsampling from the constructed pool.

        """

        if region_pool.values.shape[0] >= self.d_thre:
            region_pool_sample = region_pool.sample(n=self.d_thre)
        else:
            region_pool_sample = region_pool.copy()

        return region_pool_sample
    

    def weighted_average(self,allen_region):

        """
        Weighted integration of cell expressions from the pool with samples at specified ratios.

        """

        region_sample = sample_ex.loc[[allen_region]]
        region_pool = self.align_dataframe.loc[[allen_region]]

        pool_data = self.downsample_data(region_pool) if self.downsample else region_pool
    
        results = []

        save_path = f'./{self.integrated_save.format(self.allen_w, self.sn_w, self.pool_type)}/{allen_region}.csv'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for sample_idx, sample_exp in enumerate(region_sample.values):

            sample_norm = self._normalize(sample_exp)
        
            for cell_idx, cell_exp in enumerate(pool_data.values):

                cell_norm = self._normalize(cell_exp)
                integrated_name = f'{allen_region}_sample{sample_idx}_cell{cell_idx}'

            if stats.pearsonr(sample_norm, cell_norm)[0] > self.s_thre:
        
                weighted_avg = self.allen_w * sample_norm + self.sn_w * cell_norm
            else:
                weighted_avg = sample_norm
                
            results.append(pd.Series(weighted_avg, name=integrated_name))

        result_df = pd.concat(results, axis=1).T
        result_df.to_csv(save_path)

    
    def _normalize(self, x):

        x_min, x_max = np.min(x), np.max(x)
        return 2 * (x - x_min) / (x_max - x_min) - 1
    
    def generate_integrated_anndata(self, z_score=True, k_=0):

        """
        Generate integrated anndata.

        """

        integrated_dir = self.integrated_save.format(self.allen_w, self.sn_w, self.pool_type)

        integrated_files = [
            f'./{integrated_dir}/{region}.csv' 
            for region in align_csv[self.col_name]
        ]
    
        integrated_data = pd.concat([pd.read_csv(f).set_index('Unnamed: 0') 
                            for f in tqdm(integrated_files, desc='Loading region data')])

        if z_score:
            adata = anndata.AnnData(integrated_data.values)
            sc.pp.scale(adata, max_value=10)
            integrated_data = pd.DataFrame(
                adata.X,
                index=integrated_data.index,
                columns=integrated_data.columns
            )

        integrated_adata = anndata.AnnData(
            X=integrated_data.sample(frac=1).values, 
            obs=pd.DataFrame({'region_index': integrated_data.index}),
            var=pd.DataFrame({'Gene': gene_filter})
        )
    
        os.makedirs(integrated_dir, exist_ok=True)
        integrated_adata.write(os.path.join(integrated_dir, f'{integrated_dir}_{k_}.h5ad'))

    def run(self,pool=False,integrated_data=True,downsample=True,repeat_n=1,screening_thre=3)-> None:
        
        """
        pool: Whether to construct the pool. In specific region, we retained the top 10% of smooth-cells with the highest correlation to AHBA mean expression, which were then added to the random sampling pool. 

        integrated_data: Whether to generate the integrated dataset.

        downsample: Whether to perform downsampling from the pool.

        repeat_n: Number of integrated dataset generations.

        """

        self.pool, self.integrated_data, self.downsample = pool, integrated_data, downsample

        if self.pool:

            with Pool(self.kernel) as p:
                list(tqdm(
                    p.imap(self.pool_construction, align_csv.itertuples(index=False)),
                    total=len(align_csv),
                    desc='Constructing the pool......'))

        if self.integrated_data:
            self.clean_pool(screening_thre=screening_thre)
        
            regions = align_csv[self.col_name].tolist()
            for k in range(repeat_n):
                with Pool(self.kernel) as p:
                        list(tqdm(
                        p.imap(self.weighted_average, regions),
                        total=len(regions),
                        desc=f'Generating integrated dataset (round {k+1}/{repeat_n})'))

            self.generate_integrated_anndata(k_=k)

if __name__=='__main__':

    data_align = integratedData(pool_type,'127_region',pool_save_path,integrated_data_save_path,sc_w=0.3,allen_w=0.7,d_thre=10,s_thre=0.1)
    data_align.run()
# run.py
import sys
sys.path.append('../code/dpreprocess')
import pandas as pd
from pathlib import Path
from sn_ import Preprocess_Sn

if __name__ == '__main__':
    
    try:
        
        gene_list = pd.read_csv('./files/ahba_genes.csv')['genes'].tolist()
        H19_common_regions = pd.read_csv('./files/h19_common_regions.csv')['sn_roi'].values
        cortical_regions = [i for i in H19_common_regions if 'Cerebral cortex (Cx)' in i]
        subcortical_regions = [i for i in H19_common_regions if 'Cerebral cortex (Cx)' not in i]
        datasets = [
            '../datasets/single_nucleus_dataset/Human_sn_data_H19001.h5ad',
            '../datasets/single_nucleus_dataset/Human_sn_data_H19002.h5ad'
        ]
        scale_types = ['split', 'all']
    
        for dataset_path in datasets:
            for scale_type in scale_types:
                
                base_filename = Path(dataset_path).stem
                save_path = Path(f'../datasets/processeddata/human_sn/{base_filename}_{scale_type}_scaled.feather')

                processing_steps = {
                    'qc': False, 
                    'normalize': True,
                    'filter_genes': True,
                    'batch_correct': True,
                    'smooth': True,
                    'scale': True,
                    'save': False
                }
                
                Preprocess_Sn().preprocess_pipeline(
                    dataset_path=dataset_path,
                    save_path=str(save_path),
                    gene_list=gene_list,
                    cortical_regions=cortical_regions,
                    subcortical_regions=subcortical_regions,
                    scale_type=scale_type,
                    steps=processing_steps,
                    regress_out=True,
                    combat=False,
                    min_genes=200,
                    min_cells=3,
                    total_UMIs=800,
                    log_base=2,
                    target_sum=1e4,
                    max_fraction=0.05,
                    max_value=10
                )
                
    except Exception as e:
        print(e)            
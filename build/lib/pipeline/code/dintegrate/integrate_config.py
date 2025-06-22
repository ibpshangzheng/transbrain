import argparse
from pathlib import Path

def str_to_bool(s):
    if s.lower() in ('true', 'yes', '1'):
        return True
    elif s.lower() in ('false', 'no', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: {}'.format(s))

def parse_arguments():
    
    """Defines all configurable pipeline parameters"""
    
    parser = argparse.ArgumentParser(description='AHBA and single-nucleus dataset integrated pipeline')

    # Pipeline control
    parser.add_argument('--pool', type=str_to_bool, default=False,
                        help='pool construction')
    parser.add_argument('--integrate', type=str_to_bool, default=True,
                        help='integrated')
    parser.add_argument('--downsample',type=str_to_bool,default=True,
                        help='downsample')
    parser.add_argument('--zscore', type=str_to_bool, default=True,
                        help='zscore')

    # Integrated type
    parser.add_argument('--pool_type', default='cortex', choices=['cortex', 'subcortex', 'all'],
                        help='Brain region type to process')
    
    # Weighting parameters
    parser.add_argument('--ahba_weight', type=float, default=0.7,
                        help='Weight for ahba data (0-1)')
    parser.add_argument('--sn_weight', type=float, default=0.3,
                        help='Weight for single-nucleus data (0-1)')
    
    # Processing parameters
    parser.add_argument('--jobs', type=int, default=60,
                        help='Number of parallel jobs')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of integrated iterations')
    parser.add_argument('--downsample_n', type=int, default=10,
                        help='The number of sampled smooth_cells')
    parser.add_argument('--corr_threshold', type=float, default=0.1,
                        help='Pearsonr correlation cutoff')
    parser.add_argument('--screening_thre', type=int, default=3,
                        help='Remove smooth_cells with redundant assignments')
    parser.add_argument('--max_value', type=int, default=10,
                        help='Limit the maximum value of the data after scaling')
    
    # Path parameters
    parser.add_argument('--mean_path', type=str,
                        help='.csv file containing AHBA regional mean expression data. Required format: | Index: rois | Columns: genes |') 
    parser.add_argument('--sample_path',type=str,
                        help='.csv file containing AHBA sample expression data. Required format: | Index: rois of sample | Columns: genes |') 
    parser.add_argument('--h19_path',type=str,
                        help='.feather file containing single-nucleus expression data. Required format: | Index: rois of smooth_cells | Columns: genes |')
    parser.add_argument('--alignment_file', type=str,
                        help='.csv file containing regional alignment. Columns: | brain_region | sn_region |')
    parser.add_argument('--gene_file',type=str,
                        help='.csv file containing stabel genes corresponding to pool_type. Columns: | genes |')
    parser.add_argument('--thre_file',type=str,
                        help='.csv file containing the similarity threshold for constructing the pool. | Index: rois | Columns: | correlation |')
    parser.add_argument('--pool_s_path', type=str,
                        help='output director of pool')
    parser.add_argument('--integrated_s_path', type=str,
                        help='output director of integrated datasets')
    
    return parser.parse_args()

class Config:
    
    """Centralized configuration management"""
    
    def __init__(self):


        self.pool = False
        self.integrate = True
        self.downsample = True
        self.zscore = True
        self.pool_type = 'cortex'
        self.ahba_weight = 0.7
        self.sn_weight = 0.3
        self.jobs = 60
        self.iterations = 100
        self.downsample_n = 10
        self.corr_threshold = 0.1
        self.screening_thre = 3
        self.max_value = 10
    
        # path parameters

        self.data_files = {
            'mean_path': None,
            'sample_path': None,
            'h19_path': None,
            'alignment_file': None,
            'gene_file': None,
            'thre_file': None,
            'pool_s_path': None,
            'integrated_s_path': None
        }

    def update_parameters(self, **kwargs):
        """Update processing parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                
    def update_paths(self, **path_updates):
        """Update file system paths"""
        for path_type, new_path in path_updates.items():
                if path_type in self.data_files.keys():
                    self.data_files[path_type] = Path(new_path)
    
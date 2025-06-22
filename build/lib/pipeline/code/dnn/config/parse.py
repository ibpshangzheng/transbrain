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

    parser=argparse.ArgumentParser(add_help=True,description='Dnn parameters',formatter_class=argparse.RawDescriptionHelpFormatter)

    # model parameter
    parser.add_argument('-train', '--train',type=str_to_bool,default=True,help='Enable model training pipeline (True/False). Default: True')
    parser.add_argument('-trans', '--trans',type=str_to_bool,default=True,help='Activate model transformation pipeline (True/False). Default: True')
    parser.add_argument('-itest', '--independent_test',type=str_to_bool,default=False,help='Enable independent test. Default: False')
    parser.add_argument('-d','--device',type=str,default='cuda:0',help='Processing unit selection (cpu/cuda:0/cuda:1...). Default: cuda:0')
    
    # network parameter
    parser.add_argument('-n','--repeat_n',type=int,default=10,help='Number of training repetitions in one dataset. Default: 10')
    parser.add_argument('-iu', '--input_units',type=int,default=5000,help='Input layer dimension (gene feature count). Default: 5000')
    parser.add_argument('-hu1', '--hidden_units1',type=int,default=500,help='Number of first hidden layer. Default: 500')
    parser.add_argument('-hu2', '--hidden_units2',type=int,default=500,help='Number of second hidden layer. Default:500')
    parser.add_argument('-hu3', '--hidden_units3',type=int,default=500,help='Number of third hidden layer. Default:500')
    parser.add_argument('-ou', '--output_units',type=int,default=127,help='Output layer dimension (class labels). Default: 127')
    parser.add_argument('-wd','--weight_decay',type=float,default=1e-6,help='Regularization coefficient. Default: 1e-6')
    parser.add_argument('-me','--max_epochs',type=int,default=50,help='Maximum training iterations. Default: 50')
    parser.add_argument('-lr', '--learning_rate',type=float,default=6e-5,help='Initial learning rate. Default: 1e-5')
    parser.add_argument('-bs','--batch_size',type=int,default=256,help='Samples per gradient update. Default: 256')
    parser.add_argument('-vr','--valid_ratio',type=float,default=0.1,help='Validation set proportion. Default: 0.1')
    parser.add_argument('-br','--use_bestrank',type=bool,default=False,help='Select model by performance of cross-species rank.')
    parser.add_argument('-bv','--use_bestvalid',type=bool,default=True,help='Select model by performance of validation.')
    parser.add_argument('-shuffle','--shuffle',type=bool,default=False,help = "Shuffle the training label to test a random classifier")
    
    # cross species path
    parser.add_argument('-g_path','--gene_path',type=str,default=None,help='.csv path to common gene list file. Columns: | genes |')
    parser.add_argument('-d_path','--data_path',type=str, default=None,help='.h5ad training dataset path. Required fields:|X (expression matrix), obs (region_index: {region id}_{sample id}_{cell id}), var (Gene)|')
    parser.add_argument('-s_path','--save_path',type=str,default=None,help='Output path for model.')
    parser.add_argument('-m_trans_path','--mouse_trans_path',type=str,default=None,help='.h5ad path of mouse transform. Required fields:|X (expression matrix), obs (region_index), var (Gene)|')
    parser.add_argument('-h_trans_path','--human_trans_path',type=str,default=None,help='.h5ad path of human transform. Required fields:|X (expression matrix), obs (region_index), var (Gene)|')

    # independent test path
    parser.add_argument('-id_path','--independent_data_path',type=str,default=None,help='.h5ad independent training dataset path. Required fields:|X (expression matrix), obs (region_index: {region id}_{sample id}_{cell id}), var (Gene)|')
    parser.add_argument('-is_path','--independent_s_path',type=str,default=None,help='Output path for independent results.')
    parser.add_argument('-itest_path','--independent_test_path',type=str,default=None,help='.h5ad independent testing dataset path. Required fields:|X (expression matrix), obs (region_index: {region id}_{sample id}_{cell id}), var (Gene)|')
    
    # homologous brain region path
    parser.add_argument('-hm_path','--human_mouse_path',type=str,default=None,help='.csv path to homologous regions file. Columns: |human_region|mouse_region|')
    
    return parser.parse_args()

class Config:
    
    def __init__(self):

        self.train = True
        self.trans = True
        self.independent_test = False
        self.repeat_n = 10
        self.input_units = 5000
        self.hidden_units1 = 500
        self.hidden_units2 = 500
        self.hidden_units3 = 500
        self.output_units = 127
        self.weight_decay = 1e-6
        self.max_epochs = 50
        self.learning_rate = 6e-5
        self.batch_size = 256
        self.valid_ratio = 0.1
        self.use_bestrank = True
        self.use_bestvalid = False
        self.shuffle = False

        # path parameters

        self.data_files = {
            'gene_path': None,
            'data_path': None,
            'save_path': None,
            'mouse_trans_path': None,
            'human_trans_path': None,
            'independent_data_path': None,
            'independent_s_path': None,
            'independent_test_path': None,
            'human_mouse_path': None
        }

    def update_parameters(self, **kwargs):
        """Update processing parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                
    def update_paths(self, **path_updates):
        """Update file system paths"""
        for path_type, new_path in path_updates.items():
                if path_type in self.data_files.keys() and new_path is not None:
                    self.data_files[path_type] = Path(new_path)
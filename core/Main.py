# Main script to run experiments
import yaml
import logging
import argparse
import os       
from datetime import datetime
import sys
sys.path.insert(0, './')
from dl_utils.config_utils import *
import warnings


class Main(object):
    #Constructor
    def __init__(self, config_file):
        self.config_file = config_file
        super(Main, self).__init__()

    #Setup experiment
    def setup_experiment(self):
        warnings.filterwarnings(action='ignore') #Keep terminal clean from warnings
        logging.info("[Main::setup_experiment]: ################ Starting setup ################") #See the Progress

        #Create unique checkpoint directory for each run
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        
        #Update checkpoint paths in config file (avoid overriting previous runs)
        base_path = self.config_file['trainer']['params']['checkpoint_path'] # './weights/thor/idrid/' (original)
        save_path = os.path.join(base_path, date_time) # './weights/thor/idrid/' + date_time (later)
        self.config_file['trainer']['params']['checkpoint_path'] = save_path #update config file, so it will save run always here
    
        #Create checkpoint directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logging.info(f"[Main]: Created checkpoint directory: {save_path}")

        #Force to save downstream tasks checkpoints in the same folder
        for idx, dst_name in enumerate(self.config_file['downstream_tasks']): #Downstream tasks = segmentation, classification ... all in config file
            self.config_file['downstream_tasks'][dst_name]['checkpoint_path'] = save_path #Downstream task saved in same folder

        # Check if configurator is defined/exist in general, if not terminates
        if self.config_file['configurator'] is None:
            logging.info("[Main::setup_experiment::ERROR]: Configurator module not found, exiting...")
            exit()
        
        # Import Configurator class dynamically
        configurator_class = import_module(self.config_file['configurator']['module_name'], self.config_file['configurator']['class_name']) 
        configurator = configurator_class(config_file=self.config_file, log_wandb=False) #Creating configurator object

        #Read experiment and method names from config file
        exp_name = configurator.dl_config['experiment']['name'] # THOR_IDRiD
        method_name = configurator.dl_config['name'] # THOR_IDRiD_Experiment
        logging.info("[Main::setup_experiment]: ################ Starting experiment * {} * using method * {} * ################".format(exp_name, method_name))

        #Use GPU or CPU, load checkpoint if...
        device = 'cuda' if self.config_file['device'] == 'gpu' else 'cpu'
        checkpoint = dict()

        #Start from scratch or from checkpoint
        #Here start from scratch
        if configurator.dl_config['experiment']['weights'] is not None:
            checkpoint = torch.load(configurator.dl_config['experiment']['weights'], map_location=torch.device(device))

        #Start from checkpoint
        if configurator.dl_config['experiment']['task'] == 'train':
            configurator.start_training(checkpoint)
        else:
            configurator.start_evaluations(checkpoint['model_weights'])  #start evaluation only with model weights


#Collect command line arguments Main.py ----
def add_args(parser):
    parser.add_argument('--log_level', type=str, default='INFO', metavar='L',
                        help='log level from : ["INFO", "DEBUG", "WARNING", "ERROR"]')
    parser.add_argument('--config_path', type=str, default='projects/dummy_project/config/mnist.yaml', metavar='C',
                        help='path to configuration yaml file')
    parser.add_argument('--visualize_samples', action='store_true', help='Visualize evaluation samples')

    return parser


#Main function
if __name__ == "__main__":
    #Create argument parser and read terminal
    arg_parser = add_args(argparse.ArgumentParser(description='IML-DL'))
    args = arg_parser.parse_args()
    #Allowed ones
    if args.log_level == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.log_level == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    elif args.log_level == 'WARNING':
        logging.basicConfig(level=logging.WARNING)
    elif args.log_level == 'ERROR':
        logging.basicConfig(level=logging.ERROR)

    config_file = None
    logging.info
    (

        '------------------------------- DEEP LEARNING FRAMEWORK *IML-COMPAI-DL*  -------------------------------')
    #Load configuration file into Python
    try:
        stream_file = open(args.config_path, 'r') #open
        config_file = yaml.load(stream_file, Loader=yaml.FullLoader) #load
        logging.info('[IML-COMPAI-DL::main] Success: Loaded configuration file at: {}'.format(args.config_path))
    except:
        logging.error('[IML-COMPAI-DL::main] ERROR: Invalid configuration file at: {}, exiting...'.format(args.config_path))
        exit()

    #Now setup and run experiment (Creates main object and calls setup experiment)
    Main(config_file).setup_experiment()

    logging.info('-------------------------------  END OF EXPERIMENT  -------------------------------'
    )

"""
Configurator.py

Default class for configuring DL experiments
"""
import copy
import logging
import inspect
import torch
from dl_utils.config_utils import check_config_file, import_module, set_seed
#check_config_file: verify the config file structure that is passed to the configurator

class DLConfigurator(object):
    """
    DL Configurator and what it does:
        - parametrization of deep learning via 'config.yaml'
        - initializes Trainer:
            routine for training neural networks
        - initializes FedDownstreamTask:
            routing for downstream evaluations, e.g., classification, anomaly detection
        - starts the experiments
    """

    def __init__(self, config_file, log_wandb=False):
        """
        :param config_file:
            config.yaml file that contains the DL configuration
        """
        #Validation of config file
        self.dl_config = check_config_file(config_file) 
        if self.dl_config is None:
            print('[Configurator::init] ERROR: Invalid configuration file. Configurator will exit...')
            return

        #Set seeds
        set_seed(2109)

        #Init model and device (use GPU 0solely)
        dev = self.dl_config['device']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if dev == 'gpu' else torch.device('cpu')
        self.log_wandb = log_wandb
        model_class = import_module(self.dl_config['model']['module_name'], self.dl_config['model']['class_name'])
        self.model = model_class(**(self.dl_config['model']['params']))


    #This starts training (TRAINER)
    def start_training(self, global_model: dict = None):
        # Set the training data loader and trainer 
        data = self.load_data(self.dl_config['trainer']['data_loader'], train=True)
        trainer_class = import_module(self.dl_config['trainer']['module_name'], self.dl_config['trainer']['class_name'])
        self.trainer = trainer_class(training_params=self.dl_config['trainer']['params'], model=copy.deepcopy(self.model),
                                     data=data, device=self.device, log_wandb=self.log_wandb)

        # Train the model (if any checkpoint is provided, load it)
        model_state, opt_state, epoch = None, None, 0
        if global_model is None:
            global_model = {}
        if self.trainer is None:
            logging.error("[Configurator::train::ERROR]: Trainer not defined! Shutting down...")
            exit()
        if 'model_weights' in global_model.keys():
            model_state = global_model['model_weights']
            logging.info("[Configurator::train::INFO]: Model weights loaded!")
        if 'optimizer_weights' in global_model.keys():
            opt_state = global_model['optimizer_weights']
        if 'epoch' in global_model.keys():
            epoch = global_model['epoch']

        #Train itself
        logging.info("[Configurator::train]: ################ Starting training ################")
        trained_model_state, trained_opt_state = self.trainer.train(model_state, opt_state, epoch)
        logging.info("[Configurator::train]: ################ Finished training ################")

        logging.info("[Configurator::train]: ################ Starting testing ################")
        #Testing (MAIN TRAINING PIPELINE's TESTING)
        self.trainer.test(trained_model_state, data.test_dataloader(), task='Test')
        logging.info("[Configurator::train]: ################ Finished testing ################")

        #Extra jobs DOWNSTREAM TASKS (more complex evaluation)!
        self.start_evaluations(trained_model_state)



    def start_evaluations(self, global_model): #global model is result of training
        # Downstream Tasks
        #1. Prepare downstream tasks
        self.downstream_tasks = [] # list of downstream tasks
        nr_tasks = len(self.dl_config['downstream_tasks']) #how many downstream tasks
        for idx, dst_name in enumerate(self.dl_config['downstream_tasks']): # dst_name : retinopathy evaluation
            logging.info("[Configurator::eval]: ################ Starting downstream task nr. {}/{} -- {}-- ################".format(idx+1, nr_tasks, dst_name))
            
            #this config container contains which evaluator to use, which data loader to use, checkpoint path, and params
            dst_config = self.dl_config['downstream_tasks'][dst_name]
            downstream_class = import_module(dst_config['module_name'], dst_config['class_name']) # projects.thor.DownstreamEvaluatorIDRiD, PDownstreamEvaluator
            data = self.load_data(dst_config['data_loader'], train=False) #train false, not train then
            val_data = None
            if 'val_data_loader' in dst_config.keys():
                val_data = self.load_data(dst_config['val_data_loader'], train=False)

            accepts_val_data = False
            if val_data is not None:
                try:
                    sig = inspect.signature(downstream_class.__init__)
                    if 'val_data_dict' in sig.parameters:
                        accepts_val_data = True
                    else:
                        for param in sig.parameters.values():
                            if param.kind == inspect.Parameter.VAR_KEYWORD:
                                accepts_val_data = True
                                break
                except (TypeError, ValueError):
                    accepts_val_data = False
            if 'params' in dst_config.keys(): 
                params = self._filter_downstream_params(downstream_class, dst_config['params'], dst_name=dst_name)
                if val_data is not None and accepts_val_data:
                    dst = downstream_class(dst_name, self.model, self.device, data, dst_config['checkpoint_path'], val_data_dict=val_data, **params)
                else:
                    dst = downstream_class(dst_name, self.model, self.device, data, dst_config['checkpoint_path'], **params)
            else:
                if val_data is not None and accepts_val_data:
                    dst = downstream_class(dst_name, self.model, self.device, data, dst_config['checkpoint_path'], val_data_dict=val_data)
                else:
                    dst = downstream_class(dst_name, self.model, self.device, data, dst_config['checkpoint_path'])
            self.downstream_tasks.append(dst)
            dst.start_task(global_model=global_model)
            logging.info("[Configurator::eval]: ################ Finished downstream task nr. {}/{} ################" .format(idx+1, nr_tasks))



    @staticmethod
    def load_data(data_loader_config, train=True):
        """
        :param data_loader_config: dict
            parameters for data loaders -  must include module/class name and params
        :param train: bool
            True if the datasets are used for training, False otherwise
        :return: list of
           train, val, test datasets if train is True, dict with downstream datasets otherwise
        """
        data_loader_module = import_module(data_loader_config['module_name'], data_loader_config['class_name'])
        
        if train:
            return data_loader_module(**(data_loader_config['params']))

        downstream_datasets = dict()
        #multiple datasets for downstream tasks if it is for downstream for each
        for dataset_name in data_loader_config['datasets']:
            data = data_loader_module({**(data_loader_config['params']['args']), **(data_loader_config['datasets'][dataset_name])})
            downstream_datasets[dataset_name] = data.test_dataloader()
        return downstream_datasets

    @staticmethod
    def _filter_downstream_params(downstream_class, params, dst_name=None):
        if not params:
            return {}
        try:
            sig = inspect.signature(downstream_class.__init__)
        except (TypeError, ValueError):
            return params

        # Pass through if the class accepts **kwargs.
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return params

        allowed = {p.name for p in sig.parameters.values() if p.name != 'self'}
        filtered = {k: v for k, v in params.items() if k in allowed}
        dropped = sorted(set(params.keys()) - allowed)
        if dropped:
            name = downstream_class.__name__
            task = f" ({dst_name})" if dst_name else ""
            logging.warning(f"[Configurator::eval]: Ignoring unsupported params for {name}{task}: {dropped}")
        return filtered

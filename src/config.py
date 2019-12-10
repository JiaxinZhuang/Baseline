"""Config.

Jiaxin Zhuang, lincolnz9511@mail.com
"""
# -*- coding: utf-8 -*-

import sys
import argparse


class Config:
    """Config.
    """
    def __init__(self):
        """Load common and customized settings
        """
        super(Config, self).__init__()
        self.parser = argparse.ArgumentParser(description='Baseline')
        self.config = dict()
        # add setting via parser
        self._add_common_setting()
        self._add_customized_setting()
        # get argument parser
        self.args = self.parser.parse_args()
        # load them into config
        self._load_common_setting()
        self._load_customized_setting()
        # load config for specific server
        self._path_suitable_for_server()

    def _add_common_setting(self):
        self.parser.add_argument('--experiment_index', default="None",
                                 type=str, help="001, 002, ...")
        self.parser.add_argument('--cuda', default='0',
                                 help="cuda visible device")
        self.parser.add_argument("--num_workers", default=6, type=int,
                                 help="num_workers of dataloader")
        self.parser.add_argument('--dataset', default="CUB", type=str,
                                 choices=["CUB"],
                                 help="dataset name")
        self.parser.add_argument('--learning_rate', default=0.01, type=float,
                                 help="lr")
        self.parser.add_argument("--batch_size", default=64, type=int,
                                 help="batch size of each epoch, \
                                 for test only")
        self.parser.add_argument('--resume', default=0, type=int,
                                 help="0 means no resume from saved model")
        self.parser.add_argument("--n_epochs", default=100, type=int,
                                 help="n epochs to train")
        self.parser.add_argument("--eval_frequency", default=1, type=int,
                                 help="Eval train and test frequency")
        self.parser.add_argument('--seed', default=47, type=int,
                                 help="Random seed for pytorch and Numpy ")
        self.parser.add_argument("--optimizer", default="Adam", type=str,
                                 choices=["Adam", "SGD"],
                                 help="SGD or Adam.")
        self.parser.add_argument("--backbone", default="ResNet50", type=str,
                                 choices=["ResNet50"],
                                 help="backbone for model")
        self.parser.add_argument("--input_size", default=224, type=int,
                                 help="image input size for model")
        self.parser.add_argument("--re_size", default=256, type=int,
                                 help="resize to the size")
        # log related
        self.parser.add_argument('--model_dir', default="./saved/models/",
                                 type=str,
                                 help='store models, ../saved/models')
        self.parser.add_argument('--log_dir', default="./saved/logdirs/",
                                 type=str, help='store tensorboard files, \
                                 None means not to store')

    def _load_common_setting(self):
        """Load default setting from Parser
        """
        self.config['experiment_index'] = self.args.experiment_index
        self.config['cuda'] = self.args.cuda
        self.config["num_workers"] = self.args.num_workers
        self.config['dataset'] = self.args.dataset
        self.config["resume"] = self.args.resume
        self.config['n_epochs'] = self.args.n_epochs
        self.config['learning_rate'] = self.args.learning_rate
        self.config['batch_size'] = self.args.batch_size
        self.config['seed'] = self.args.seed
        self.config["eval_frequency"] = self.args.eval_frequency
        self.config["optimizer"] = self.args.optimizer
        self.config["backbone"] = self.args.backbone
        self.config["input_size"] = self.args.input_size
        self.config["re_size"] = self.args.re_size

        self.config['log_dir'] = self.args.log_dir
        self.config['model_dir'] = self.args.model_dir

    def _add_customized_setting(self):
        """Add customized setting
        """
        self.parser.add_argument("--server", default="ls15", type=str,
                                 choices=["ls15", "ls16", "local",
                                          "lab_server"])
        self.parser.add_argument("--test_input_size", default=224, type=int,
                                 help="re_size for test")

    def _load_customized_setting(self):
        """Load sepcial setting
        """
        self.config["server"] = self.args.server
        self.config["test_input_size"] = self.args.test_input_size

    def _path_suitable_for_server(self):
        """Path suitable for server
        """
        if self.config["server"] == "local":
            self.config["log_dir"] = "/media/lincolnzjx/Disk2/saved/logdirs"
            self.config["model_dir"] = "/media/lincolnzjx/Disk2/saved/models"
        elif self.config["server"] in ["ls15", "ls16", "lab_center"]:
            self.config["log_dir"] = "./saved/logdirs"
            self.config["model_dir"] = "./saved/models"
        else:
            print("Valid path for server need!")
            sys.exit(-1)

    def print_config(self, _print=None):
        """print config
        """
        _print("==================== basic setting start ====================")
        for arg in self.config:
            _print('{:20}: {}'.format(arg, self.config[arg]))
        _print("==================== basic setting end ====================")

    def get_config(self):
        """return config
        """
        return self.config

"""This file contains the configurations for the project.
"""
from src.config.util.base_config import _Arg, _BaseConfig
from src.config.logging_config import LoggingConfig
import src.config.models as models_config
import src.config.trainers as trainers_config
import src.config.trainer_stats as trainer_stats_config

class CodeCarbonConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(type=int, help="The run number used for codecarbon file tracking.", default=0)
        self._arg_project_name = _Arg(type=str, help="The name of the project used for codecarbon file tracking.", default="energy-efficiency")

class TrainerStatsConfigs(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self.codecarbon = CodeCarbonConfig()

class Config(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_model = _Arg(type=str, help="Which model to train.")
        self._arg_trainer = _Arg(type=str, help="How to train the model", default="simple", choices=["simple"])
        self._arg_dataset = _Arg(type=str, help="Which dataset to use.", default="allenai/c4",  choices=["allenai/c4"])
        self._arg_dataset_train_files = _Arg(type=str, help="Which files to use for the dataset.", default="/mnt/nobackup/omicha1/c4/downloaded/multilingual/c4-en.tfrecord-00000-of-*.json.gz")
        self._arg_dataset_split = _Arg(type=str, help="How to split the dataset (ex: train[:100])", default="train[:100]")
        self._arg_dataset_load_num_proc = _Arg(type=int, help="Number of threads used to load the dataset.", default=20)
        self._arg_tokenize_num_process = _Arg(type=int, help="Number of threads used to tokenize the dataset.", default=20)
        self._arg_batch_size = _Arg(type=int, help="Size of batches", default=4)
        self._arg_train_stats = _Arg(type=str, help="Type of statistics to gather. By default it is set to no-op, which ignores everything.", default="no-op", choices=["no-op", "no-op-sync", "simple", "torch-profiler", "codecarbon", "averaged-energy"])
        self._arg_run_num = _Arg(type=int, help="The run number used for codecarbon file tracking.", default=0)
        self._arg_project_name = _Arg(type=str, help="The name of the project used for codecarbon file tracking.", default="energy-efficiency")
        self._arg_learning_rate = _Arg(type=float, help="The learning rate for training. It is used by the optimizer for all models.", default=1e-6)
        self.trainer_stats = TrainerStatsConfigs()
        self.logging = LoggingConfig()
        self.models = models_config.ModelsConfig()
        self.trainers = trainers_config.TrainersConfig()
        self.trainer_stats = trainer_stats_config.TrainerStatsConfig()

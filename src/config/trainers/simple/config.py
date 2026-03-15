from src.config.util.base_config import _Arg, _BaseConfig

config_name = "simple"


class TrainerConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_max_duration_sec = _Arg(
            type=float,
            default=0.0,
            help="Maximum measurement duration in seconds. Set <=0 to disable time-based stop.",
        )
        self._arg_warmup_steps = _Arg(
            type=int,
            default=0,
            help="Number of warmup steps to run before starting measured execution.",
        )
        self._arg_warmup_sec = _Arg(
            type=float,
            default=0.0,
            help="Warmup time in seconds before measured execution starts.",
        )

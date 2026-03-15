from src.config.util.base_config import _Arg, _BaseConfig

config_name = "composite"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_components = _Arg(
            type=str,
            default="simple,resource",
            help="Comma-separated trainer stats names to combine (e.g., simple,resource).",
        )

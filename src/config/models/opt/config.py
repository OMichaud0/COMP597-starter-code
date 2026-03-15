from src.config.util.base_config import _Arg, _BaseConfig

config_name = "opt"


class ModelConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_hf_name = _Arg(type=str, help="HuggingFace model id for OPT.", default="facebook/opt-350m")
        self._arg_dtype = _Arg(type=str, help="Model load dtype: fp16 or fp32.", default="fp32")

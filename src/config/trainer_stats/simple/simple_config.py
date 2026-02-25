from src.config.util.base_config import _Arg, _BaseConfig

config_name = "simple"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(type=str, default=".", help="Directory where simple trainer stats files are written.")
        self._arg_output_file_prefix = _Arg(type=str, default="simple_stats", help="Filename prefix for simple trainer stats files.")
        self._arg_plot_metrics = _Arg(type=int, default=1, help="Generate metric-over-time PNG plots at the end of training (1=yes, 0=no).")
        self._arg_plot_x_axis = _Arg(type=str, default="elapsed_sec", help="Column to use on x-axis for generated plots (e.g. elapsed_sec or step).")
        self._arg_plot_max_metrics = _Arg(type=int, default=0, help="Maximum number of metrics to plot (<=0 means plot all).")

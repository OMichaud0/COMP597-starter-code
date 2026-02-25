import logging
import csv
import os
import pynvml
import time
from typing import Dict, Iterable, List, Optional
import torch

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - environment dependent
    Image = None
    ImageDraw = None

logger = logging.getLogger(__name__)

_TRAINER_STATS_AUTO_DISCOVERY_IGNORE=True

class RunningAverage:
    """Implements a running average.
    
    Attributes
    ----------
    average : float
        Current average.
    n : int
        Number of samples.
    """

    def __init__(self) -> None:
        self.average = 0
        self.n = 0

    def update(self, value : int) -> None:
        """Update the average with a new value.
        
        This will perform an inplace update on the average.

        Parameters
        ----------
        value
            The value to include in the running average.

        """
        self.average = (self.average * self.n + value) / (self.n + 1)
        self.n += 1

    def get(self) -> float:
        """The current average.
        """
        return self.average

class RunningStat:
    """An updatable statistics.

    This is used to accumulate an integer statistics and provide a breakdown on 
    the data.

    Attributes
    ----------
    average : RunningAverage
        The running average of the statistics.
    history : List[Any]
        All the values accumulated.

    """

    # TODO Add a unit transformation parameter. Log analysis should not by default divide values by 1e6. It should be configurable.
    def __init__(self) -> None:
       self.average = RunningAverage()
       self.history = []

    def update(self, value : int) -> None:
        """Include a new value.

        This includes a new value in the history and the running average.

        Parameters
        ----------
        value
            The value to add to the statistic.

        """
        self.history.append(value)
        self.average.update(value)

    def get_average(self) -> float:
        """The current average of the statistic.
        """
        return self.average.get()

    def get_last(self) -> int:
        """The last value added to the statistic.
        """
        if len(self.history) == 0:
            return -1
        return self.history[-1]

    def log_analysis(self) -> None:
        """Logs the mean and key quantiles of accumulated data.
        """
        data = torch.tensor(self.history)
        data = data.to(torch.float)
        print(f"mean   : {data.mean() / 1000000 : .4f}")
        print(f"q0.001 : {data.quantile(q=torch.tensor(0.001), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.01  : {data.quantile(q=torch.tensor(0.010), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.1   : {data.quantile(q=torch.tensor(0.100), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.25  : {data.quantile(q=torch.tensor(0.250), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.5   : {data.quantile(q=torch.tensor(0.500), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.75  : {data.quantile(q=torch.tensor(0.750), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.9   : {data.quantile(q=torch.tensor(0.900), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.99  : {data.quantile(q=torch.tensor(0.990), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.999 : {data.quantile(q=torch.tensor(0.999), interpolation='nearest') / 1000000 : .4f}")

class RunningTimer:
    """A reusable timer.

    Provides a timer that can be reused. It keeps track of all the time 
    measurements performed. Measurements are in nanoseconds as provided by 
    Python's native `time.perf_counter_ns()`.

    Attributes
    ----------
    stat : RunningStat
        The statistic object containing all the measurements done with this timer.
    stats_ts : int
        The timestamp measured by `start`.

    Notes
    -----
        This timer does not support recursive starts. It should be stopped 
        before being started again. There is not verification of proper usage, 
        it is the programmer's task to ensure proper usage.

    """

    def __init__(self) -> None:
       self.stat = RunningStat()
       self.start_ts = 0

    def start(self) -> None:
        """Start the timer.
        
        Starts the timer. The timestamp is stored in the attribute `start_ts`.

        """
        self.start_ts = time.perf_counter_ns()

    def stop(self) -> None:
        """Stop the timer.

        Stops the timer. It records the time different between the timestamp of 
        when the timer was started and the current timestamp. The measurement 
        is added to the statistics.
        
        """
        self.stat.update(time.perf_counter_ns() - self.start_ts)

    def get_last(self) -> int:
        """The last time measurement.
        """
        return self.stat.get_last()

    def get_average(self) -> float:
        """The average time measurement.
        """
        return self.stat.get_average()

    def log_analysis(self) -> None:
        """Analysis of the time measurements.

        Refer to the `log_analysis` implementation of `RunningStat` for more 
        details on the output of this method.

        """
        self.stat.log_analysis()

class RunningEnergy:
    """A reusable GPU energy counter.

    Provides an energy counter for GPUs that can be reused. It keeps track of 
    all the energy consumption measurements performed. Measurements are in 
    millijoules.

    Parameters
    ----------
    gpu_index
        The index of the GPU for which to measure the energy consumption.

    Attributes
    ----------
    stat : RunningStat
        The statistic object containing all the measurements done with this 
        counter.
    start_energy : int
        The energy measured when the counter was started.
    gpu_index : int
        The GPU index as provided to the constructor.
    handle : pynvml._Pointer[struct_c_nvmlDevice_t]
        The NVML handle of the GPU tracked by this counter..

    Notes
    -----
        This counter does not support recursive starts. It should be stopped 
        before being started again. There is not verification of proper usage, 
        it is the programmer's task to ensure proper usage.

    """

    def __init__(self, gpu_index : int) -> None:
        self.stat = RunningStat()
        self.start_energy = 0
        if gpu_index is None:
            logger.warning("GPU index not provided, defaulting to 0.")
            gpu_index = 0
        self.gpu_index = gpu_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def _get_energy(self) -> int:
        """The current Energy consumption of the GPU.

        This method returns the total energy consumption in millijoules that 
        the GPU has consumed since the last the driver was reloaded as 
        specified by the NVML library.
        
        """
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def start(self) -> None:
        """Start the counter.

        It records the energy consumption *timestamp* of the GPU.

        """
        self.start_energy = self._get_energy()

    def stop(self) -> None:
        """Stops the counter.

        The recorded energy measurement is the different between the current 
        GPU energy consumption and the energy consumption when the counter was 
        started. This yields how much energy was consumed between the calls to 
        `start` and `stop`.

        """
        self.stat.update(self._get_energy() - self.start_energy)

    def get_last(self) -> int: 
        """The last energy measurement.
        """
        return self.stat.get_last()

    def get_average(self) -> float:
        """The average energy consumptions so far.
        """
        return self.stat.get_average()

    def log_analysis(self) -> None:
        """Analysis of the energy measurements.

        Refer to the `log_analysis` implementation of `RunningStat` for more 
        details on the output of this method.

        """
        self.stat.log_analysis()


def _to_float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_numeric_csv_rows(path: str) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k: _to_float_or_none(v) for k, v in row.items()})
    return rows


def _draw_simple_line_plot(
    x_values: Iterable[float],
    y_values: Iterable[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: str,
) -> bool:
    if Image is None or ImageDraw is None:
        logger.warning("Pillow is not available; cannot render metric plots.")
        return False

    xs = list(x_values)
    ys = list(y_values)
    if len(xs) < 2 or len(ys) < 2:
        return False

    width = 1200
    height = 700
    left = 90
    right = 30
    top = 55
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(xv: float) -> int:
        return int(left + ((xv - x_min) / (x_max - x_min)) * plot_w)

    def sy(yv: float) -> int:
        return int(top + (1.0 - ((yv - y_min) / (y_max - y_min))) * plot_h)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Axes
    draw.line((left, top, left, top + plot_h), fill=(0, 0, 0), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(0, 0, 0), width=2)

    # Line plot
    points = [(sx(xv), sy(yv)) for xv, yv in zip(xs, ys)]
    draw.line(points, fill=(35, 99, 255), width=3)

    # Labels and range annotations
    draw.text((left, 10), title, fill=(0, 0, 0))
    draw.text((width // 2 - 80, height - 28), x_label, fill=(0, 0, 0))
    draw.text((6, top + plot_h // 2), y_label, fill=(0, 0, 0))
    draw.text((left, height - 50), f"min={x_min:.3f}, max={x_max:.3f}", fill=(0, 0, 0))
    draw.text((left + 250, height - 50), f"y-min={y_min:.3f}, y-max={y_max:.3f}", fill=(0, 0, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return True


def generate_metric_plots_from_csv(
    csv_path: str,
    output_dir: str,
    output_file_prefix: str,
    x_axis: str = "elapsed_sec",
    exclude_metrics: Optional[Iterable[str]] = None,
    max_metrics: int = 0,
) -> List[str]:
    rows = read_numeric_csv_rows(csv_path)
    if not rows:
        return []

    exclude = set(exclude_metrics or [])
    exclude.update({x_axis})

    series: Dict[str, List[float]] = {}
    series_x: Dict[str, List[float]] = {}
    for idx, row in enumerate(rows):
        x = row.get(x_axis, None)
        if x is None:
            x = float(idx + 1)
        x = float(x)

        for key, value in row.items():
            if key in exclude or value is None:
                continue
            series.setdefault(key, []).append(float(value))
            series_x.setdefault(key, []).append(x)

    if not series:
        return []

    plot_dir = os.path.join(output_dir, f"{output_file_prefix}_plots")
    os.makedirs(plot_dir, exist_ok=True)

    metrics = sorted(series.keys())
    if max_metrics is not None and max_metrics > 0:
        metrics = metrics[:max_metrics]

    written: List[str] = []
    for metric in metrics:
        x_values = series_x[metric]
        y_values = series[metric]
        if len(y_values) < 2:
            continue
        output_path = os.path.join(plot_dir, f"{metric}_vs_{x_axis}.png")
        ok = _draw_simple_line_plot(
            x_values=x_values,
            y_values=y_values,
            title=f"{metric} vs {x_axis}",
            x_label=x_axis,
            y_label=metric,
            output_path=output_path,
        )
        if ok:
            written.append(output_path)
    return written


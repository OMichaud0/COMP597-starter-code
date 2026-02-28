# Run 10 Configuration

## Logging
| Parameter | Value |
|-----------|-------|
| logging.level | INFO |

## Model
| Parameter | Value |
|-----------|-------|
| model | opt |

## Data
| Parameter | Value |
|-----------|-------|
| data | opt |
| data_configs.opt.max_samples | 100000 |
| data_configs.opt.batch_size | 16 |
| data_configs.opt.num_workers | 0 |

## Trainer Stats
| Parameter | Value |
|-----------|-------|
| trainer_stats | simple |
| trainer_stats_configs.simple.output_dir | ./results/run-9 |
| trainer_stats_configs.simple.output_file_prefix | run_10 |
| trainer_stats_configs.simple.plot_metrics | 1 |
| trainer_stats_configs.simple.plot_x_axis | elapsed_sec |

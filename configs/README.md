# Configurations

Experiments are defined by a configuration file (JSON). All configuration files use the following attributes:

* `exp_name`
    * Experiment name. Dictates output directory folder.
* `agent`
    * Agent to run. Refer to [agents](/agents) for agent to use.
* `od_root` (optional)
    * Path to experiment output directory. Defaults to (`experiments/`)
* `tb_dir` (optional)
    * Path to experiment tensorboard log directory. Defaults to `{od_root}/{exp_name}/tb/`
* `chkpt_dir` (optional)
    * Path to experiment model checkpoints. Defaults to `{od_root}/{exp_name}/checkpoints/`
* `out_dir` (optional)
    * Path to experiment output. Defaults to `{od_root}/{exp_name}/out/`
* `log_dir` (optional)
    * Path to experiment log files. Defaults to `{od_root}/{exp_name}/logs/`

Other configuration defined keys are agent specific.

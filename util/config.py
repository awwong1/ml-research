import logging
from logging.handlers import RotatingFileHandler
import json
from pprint import pprint, pformat
from os import path, makedirs


def configure_logging(log_dir):
    log_file_format = (
        "[%(levelname)s] %(asctime)s %(name)s: %(message)s @ %(pathname)s:%(lineno)d"
    )
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)
    # Console handles INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_console_format))
    # File handles DEBUG
    exp_debug_file_handler = RotatingFileHandler(
        "{}debug.log".format(log_dir), maxBytes=10 ** 6, backupCount=5
    )
    exp_debug_file_handler.setLevel(logging.DEBUG)
    exp_debug_file_handler.setFormatter(logging.Formatter(log_file_format))
    # Warning file handles WARN
    exp_warn_file_handler = RotatingFileHandler(
        "{}warn.log".format(log_dir), maxBytes=10 ** 6, backupCount=5
    )
    exp_warn_file_handler.setLevel(logging.WARNING)
    exp_warn_file_handler.setFormatter(logging.Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_debug_file_handler)
    main_logger.addHandler(exp_warn_file_handler)


def create_dirs(dirs):
    for _dir in dirs:
        makedirs(_dir, exist_ok=True)


def deep_merge_dict(a, b, path=[], overwrite=True):
    """Deeply merges b into a"""
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deep_merge_dict(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            elif overwrite:
                a[key] = b[key]
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def parse_config(config_fp, raw_config_override=None):
    with open(config_fp, "r") as config_f:
        try:
            config = json.load(config_f)
        except Exception as err:
            print("ERROR: Failed to parse configuration: {}".format(err))
            exit(-1)

    if raw_config_override:
        override = json.loads(raw_config_override)
        print(
            "WARN: Overriding configuration file. Deep merging in values: \n{}".format(
                pformat(override)
            )
        )
        config = deep_merge_dict(config, override)

    print("Configuration:")
    pprint(config)

    exp_name = config.get("exp_name", None)
    if exp_name is None:
        print("ERROR: Configuration exp_name not set.")
        exit(-1)
    else:
        print("=" * 60 + "\nExperiment: {}\n".format(exp_name) + "=" * 60)

    # setup the experiment required directories
    od_root = config.get("od_root", "experiments/")
    od_summary = config.get("tb_dir", path.join(od_root, exp_name, "tb/"))
    od_chkpnt = config.get("chkpt_dir", path.join(od_root, exp_name, "checkpoints/"))
    od_out = config.get("out_dir", path.join(od_root, exp_name, "out/"))
    od_logs = config.get("log_dir", path.join(od_root, exp_name, "logs/"))
    create_dirs([od_summary, od_chkpnt, od_chkpnt, od_out, od_logs])
    config["tb_dir"] = od_summary
    config["chkpt_dir"] = od_chkpnt
    config["out_dir"] = od_out
    config["log_dir"] = od_logs

    # setup project wide logging
    configure_logging(config["log_dir"])
    logger = logging.getLogger()

    logger.debug(pformat(config))
    logger.info("tensorboard output: {}".format(config["tb_dir"]))
    logger.info("checkpoint output: {}".format(config["chkpt_dir"]))
    logger.info("output directory: {}".format(config["out_dir"]))
    logger.info("log output: {}".format(config["log_dir"]))

    return config

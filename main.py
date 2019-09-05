#!/usr/bin/env python3
from argparse import ArgumentParser
from util.reflect import fetch_class
from util.config import parse_config


def main():
    parser = ArgumentParser("Research Experiment Runner")
    parser.add_argument(
        "config", metavar="config_json", help="Experiment configuration JSON file"
    )
    parser.add_argument(
        "--override",
        metavar="override_json",
        default=None,
        type=str,
        help="Serialized JSON object to merge into configuration (overrides config)",
    )
    args = parser.parse_args()

    config = parse_config(args.config, args.override)
    agent_query = config.get("agent", None)
    agent_class = fetch_class(agent_query)
    agent_instance = agent_class(config)
    try:
        agent_instance.run()
    finally:
        agent_instance.finalize()


if __name__ == "__main__":
    main()

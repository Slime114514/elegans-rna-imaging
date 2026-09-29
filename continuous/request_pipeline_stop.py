# -*- coding: utf-8 -*-
"""Create a stop-request file that long pipeline modules check between chunks."""
import importlib.util
from pathlib import Path


def _load_config():
    module_path = Path(__file__).resolve().with_name("00_config_utils.py")
    spec = importlib.util.spec_from_file_location("pipeline_config_utils", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()


def main():
    path = cfg.write_stop_request()
    cfg.log(f"Stop requested. Pipeline modules will stop at the next checkpoint: {path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Clear the stop-request file so the pipeline can resume."""
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
    removed = cfg.clear_stop_request()
    if removed:
        cfg.log(f"Cleared stop request: {cfg.STOP_REQUEST_FILE}")
    else:
        cfg.log(f"No stop request was present: {cfg.STOP_REQUEST_FILE}")


if __name__ == "__main__":
    main()

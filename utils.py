import yaml
from pathlib import Path
import pdb

def ReadConfig(config_path:str="model.config"):
    cwd=Path.cwd()
    cfg_path=cwd/config_path
    if cfg_path.exists():
        with open(config_path,'r') as cfg:
            cfg=yaml.safe_load(cfg)
        return cfg
    else:
        print(f"Path to config file does not exist: {cfg_path}")


ReadConfig()

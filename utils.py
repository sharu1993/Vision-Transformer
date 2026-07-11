import yaml
from pathlib import Path
import pdb

def read_config(config_path:str="model.config"):
    cwd=Path.cwd()
    cfg_path=cwd/config_path
    if cfg_path.exists():
        with open(config_path,'r') as cfg:
            cfg=yaml.safe_load(cfg)
        return cfg['model'],cfg['training'],cfg['data']
    else:
        print(f"Path to config file does not exist: {cfg_path}")

def flatten_dict(d:dict, parent='',sep='_'):
    items=[]
    for key, value in d.items():
        new_key=f"{parent}{sep}{key}" if parent else key
        if isinstance(value,dict):
            items.extend(flatten_dict(value,new_key,sep=sep).items())
        else:
            items.append((new_key,value))
    return dict(items)
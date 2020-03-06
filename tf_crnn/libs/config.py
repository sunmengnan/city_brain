import yaml
import os
#from pathlib import Path
from easydict import EasyDict

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def load_config(conf_name):
    """
    :param conf_name: 配置文件的名称，不需要带 yml 后缀
    :return: 
    """
    #conf_path = CURRENT_DIR.parent / 'data' / 'cfgs' / ('%s.yml' % conf_name)
    conf_path = os.path.join(CURRENT_DIR,'../data','cfgs',conf_name +'.yml')
    with open(conf_path, mode='r') as f:
        cfg = yaml.load(f.read())
        cfg = EasyDict(cfg)

    for k, v in cfg.items():
        print('%s: %s' % (k, v))
    return cfg

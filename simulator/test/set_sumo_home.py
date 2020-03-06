import sys
import os
from sys import platform


def set_sumo_home():
    if platform == "linux" or platform == "linux2":
        os.environ['SUMO_HOME'] = '/usr/share/sumo'
    elif platform == "darwin":
        os.environ['SUMO_HOME'] = "/usr/local/opt/sumo/share/sumo"
    else:
        os.environ['SUMO_HOME'] = 'C:\Program Files (x86)\Eclipse\Sumo'
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)
from continuous_cartpole import CartPoleEnv
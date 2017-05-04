"""Safely set numexpr.set_num_threads(1) before attempting multiprocessing"""

import numexpr
numexpr.set_num_threads(1)

from multiprocessing import *
import multiprocessing.sharedctypes
sharedctypes = multiprocessing.sharedctypes


""" yluo - 09/25/2014 creation
common data reading, writing, and transforming utilities. 
"""
__author__= """Yuan Luo (yuanluo@mit.edu)"""
__revision__="0.5"

import numpy as np
from operator import itemgetter
from datetime import datetime
import os
import re
import csv
import copy
import sys


def add_2l_hash(h, k1, k2, v):
    if not h.has_key(k1):
        h[k1] = {}
    if not h[k1].has_key(k2):
        h[k1][k2] = 0
    h[k1][k2] += v
    return;
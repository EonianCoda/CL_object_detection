from main import get_parser, Params
import torch
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from collections import defaultdict
from utils import Experimental_tool

#TODO 尚未完成

def cal_norm(x):
    return float(torch.norm(x))


def get_classifier_norm(params, state:int, epoch:int):
    model = params.get_model(state, epoch)
    tool = Experimental_tool(model, params)
    classifier = tool.get_classed_classifier()
    
    weight_norms = []
    bias_norms = []
    for p in classifier:
        weight_norms.append(cal_norm(p['weight']))
        bias_norms.append(cal_norm(p['bias']))
    return weight_norms, bias_norms



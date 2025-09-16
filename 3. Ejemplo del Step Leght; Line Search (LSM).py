# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:54:18 2023

@author: 000095840
"""


# First let us import all the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as au
from autograd import grad, jacobian
import scipy

def himm(x): # Objective function
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

grad_himm = grad(himm) # Gradient of the objective function
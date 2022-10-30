#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:05:48 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab

def plotData(x, y):
  length = len(y)

  pylab.figure(1)

  pylab.plot(x, y, 'rx')
  pylab.xlabel('x')
  pylab.ylabel('y')

  pylab.show()#让绘制的图像在屏幕上显示出来

x = []
y = []


x = [float(l.split()[3]) for l in open("rec.txt")]
y = [float(l.split()[11]) for l in open("prec.txt")]


plotData(x,y)
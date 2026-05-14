# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:39:40 2023

@author: yez4
"""


def preset_mpl():
    import matplotlib as mpl
    # mpl.use('QtAgg')

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 8 #small
    mpl.rcParams['legend.markerscale'] = 0.5      # Smaller legend markers(1)
    mpl.rcParams['legend.handlelength'] = 0.5     # Shorter legend lines (2)
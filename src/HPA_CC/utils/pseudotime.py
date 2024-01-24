# -*- coding: utf-8 -*-
"""
Utilities for calculating pseudotime from FUCCI cell images.

Adapted from:
    Title: FucciCellCycle.py, FucciPseudotime.py, StretchTime.py
    Author: Anthony J. Cesnik, devinsullivan
    Date: 01/23/24
    Code version: Commit 1a235bb
    Availability: https://github.com/CellProfiling/SingleCellProteogenomics/
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from copy import deepcopy

class FucciCellCycle:
    def __init__(self):
        # Length of the cell cycle observed for the FUCCI cell line through live cell imaging exp
        self.G1_LEN = 10.833  # hours (plus 10.833, so 13.458hrs for the S/G2 cutoff)
        self.G1_S_TRANS = (
            2.625  # hours (plus 10.833 and 2.625 so 25.433 hrs for the G2/M cutoff)
        )
        self.S_G2_LEN = (
            11.975  # hours (this should be from the G2/M cutoff above to the end)
        )
        self.M_LEN = 0.5  # We excluded M-phase from this analysis
        self.TOT_LEN = self.G1_LEN + self.G1_S_TRANS + self.S_G2_LEN
        self.G1_PROP = self.G1_LEN / self.TOT_LEN
        self.G1_S_PROP = self.G1_S_TRANS / self.TOT_LEN + self.G1_PROP
        self.S_G2_PROP = self.S_G2_LEN / self.TOT_LEN + self.G1_S_PROP

def intensities_to_polar_pseudotime(log_intensities, center=None):
    if center is None:
        center_estimate = np.mean(log_intensities, axis=0)
        center_est2 = least_squares(f_2, center_estimate, args=(log_intensities[:, 0], log_intensities[:, 1]))
        center = center_est2.x
    centered_intensities = log_intensities - center
    r = np.sqrt(np.sum(centered_intensities ** 2, axis=1))
    theta = np.arctan2(centered_intensities[:, 1], centered_intensities[:, 0])
    polar = np.stack([r, theta], axis=-1)
    fucci_time = calculate_pseudotime(polar.T, centered_intensities)
    return fucci_time

def calculate_pseudotime(pol_data, centered_data, save_dir=""):
    pol_sort_inds = np.argsort(pol_data[1])
    pol_sort_rho = pol_data[0][pol_sort_inds]
    pol_sort_phi = pol_data[1][pol_sort_inds]
    centered_data_sort0 = centered_data[pol_sort_inds, 0]
    centered_data_sort1 = centered_data[pol_sort_inds, 1]

    # Rezero to minimum --resoning, cells disappear during mitosis, so we should have the fewest detected cells there
    bins = plt.hist(pol_sort_phi, 1000)
    plt.close()
    start_phi = bins[1][np.argmin(bins[0])]

    # Move those points to the other side
    more_than_start = np.greater(pol_sort_phi, start_phi)
    less_than_start = np.less_equal(pol_sort_phi, start_phi)
    pol_sort_rho_reorder = np.concatenate(
        (pol_sort_rho[more_than_start], pol_sort_rho[less_than_start])
    )
    pol_sort_inds_reorder = np.concatenate(
        (pol_sort_inds[more_than_start], pol_sort_inds[less_than_start])
    )
    pol_sort_phi_reorder = np.concatenate(
        (pol_sort_phi[more_than_start], pol_sort_phi[less_than_start] + np.pi * 2)
    )
    pol_sort_centered_data0 = np.concatenate(
        (centered_data_sort0[more_than_start], centered_data_sort0[less_than_start])
    )
    pol_sort_centered_data1 = np.concatenate(
        (centered_data_sort1[more_than_start], centered_data_sort1[less_than_start])
    )
    pol_sort_shift = pol_sort_phi_reorder + np.abs(np.min(pol_sort_phi_reorder))

    # Shift and re-scale "time"
    # reverse "time" since the cycle goes counter-clockwise wrt the fucci plot
    pol_sort_norm = pol_sort_shift / np.max(pol_sort_shift)
    pol_sort_norm_rev = 1 - pol_sort_norm
    pol_sort_norm_rev = stretch_time(pol_sort_norm_rev)
    pol_unsort = np.argsort(pol_sort_inds_reorder)
    fucci_time = pol_sort_norm_rev[pol_unsort]
    return fucci_time

def stretch_time(time_data,nbins=1000):
    '''This function is supposed to create uniform density space'''
    n, bins, patches = plt.hist(time_data, histedges_equalN(time_data, nbins), density=True)
    tmp_time_data = deepcopy(time_data)
    trans_time = np.zeros([len(time_data)])
    
    # Get bin indexes
    for i,c_bin in enumerate(bins[1:]):
        c_inds = np.argwhere(tmp_time_data<c_bin)
        trans_time[c_inds] = i/nbins
        tmp_time_data[c_inds] = np.inf
    return trans_time

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i = round(i + step, 14)

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))

def histedges_equalA(x, nbin):
    pow = 0.5
    dx = np.diff(np.sort(x))
    tmp = np.cumsum(dx ** pow)
    tmp = np.pad(tmp, (1, 0), 'constant')
    return np.interp(np.linspace(0, tmp.max(), nbin + 1), tmp, np.sort(x))
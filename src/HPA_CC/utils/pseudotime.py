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

def intensities_to_pseudotime(log_intensities, center=None):
    # converts FUCCI GMNN and CDT1 intensities from cartesian (x, y) to polar (r, theta) coordinates
    if center is None:
        center_estimate = np.mean(log_intensities, axis=0)
        center_est2 = least_squares(f_2, center_estimate, args=(log_intensities[:, 0], log_intensities[:, 1]))
        center = center_est2.x
    centered_intensities = log_intensities - center
    centered_rescaled_intensities = centered_intensities / (center - np.array([0, 0]))
    r = np.sqrt(np.sum(centered_rescaled_intensities ** 2, axis=1))
    theta = np.arctan2(centered_rescaled_intensities[:, 1], centered_rescaled_intensities[:, 0])
    polar = np.stack([r, theta], axis=-1)
    fucci_time, raw_time = calculate_pseudotime(polar.T, centered_rescaled_intensities)
    return fucci_time, raw_time, centered_rescaled_intensities

def min_angle_diff(a, b):
    return min((a - b) % (2 * np.pi), (b - a) % (2 * np.pi))

def calculate_pseudotime(pol_data, centered_data, save_dir=""):
    r, theta = pol_data
    x, y = centered_data.T
    start_theta = np.arctan2(np.min(y), -1)

    # sort by theta
    pol_sort_inds = np.argsort(theta)
    pol_sort_r = pol_data[0][pol_sort_inds]
    pol_sort_theta = pol_data[1][pol_sort_inds]
    pol_sort_x = x[pol_sort_inds]
    pol_sort_y = y[pol_sort_inds]

    # Move those points to the other side
    more_than_start = np.greater(pol_sort_theta, start_theta)
    less_than_start = np.less_equal(pol_sort_theta, start_theta)
    pol_sort_rho_reorder = np.concatenate(
        (pol_sort_r[more_than_start], pol_sort_r[less_than_start])
    )
    pol_sort_inds_reorder = np.concatenate(
        (pol_sort_inds[more_than_start], pol_sort_inds[less_than_start])
    )
    pol_sort_theta_reorder = np.concatenate(
        (pol_sort_theta[more_than_start], pol_sort_theta[less_than_start] + np.pi * 2)
    )
    pol_sort_x_reorder = np.concatenate(
        (pol_sort_x[more_than_start], pol_sort_x[less_than_start])
    )
    pol_sort_y_reorder = np.concatenate(
        (pol_sort_y[more_than_start], pol_sort_y[less_than_start])
    )
    pol_sort_shift = pol_sort_theta_reorder + np.abs(np.min(pol_sort_theta_reorder))


    # Shift and re-scale "time"
    # reverse "time" since the cycle goes counter-clockwise wrt the fucci plot
    pol_sort_norm = pol_sort_shift / np.max(pol_sort_shift)
    pol_sort_norm_rev = 1 - pol_sort_norm
    raw_time = pol_sort_norm_rev.copy()
    pol_sort_norm_rev = stretch_time(pol_sort_norm_rev)
    pol_sort_norm_rev[0] = 1.0
    pol_unsort = np.argsort(pol_sort_inds_reorder)
    fucci_time = pol_sort_norm_rev[pol_unsort]
    raw_time = raw_time[pol_unsort]
    return fucci_time, raw_time

def stretch_time(time_data,nbins=1000):
    # This function creates a uniform across pseudotime
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

def f_2(c, x, y):
    """Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)"""
    # print(c)
    Ri = calc_R(c[0], c[1], x, y)
    return Ri - Ri.mean()

def calc_R(xc, yc, x, y):
    """Calculate the distance of each 2D points from the center (xc, yc)"""
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


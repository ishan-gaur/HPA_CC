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
    r = np.sqrt(np.sum(centered_intensities ** 2, axis=1))
    theta = np.arctan2(centered_intensities[:, 1], centered_intensities[:, 0])
    polar = np.stack([r, theta], axis=-1)
    fucci_time = calculate_pseudotime(polar.T, centered_intensities)
    return fucci_time, center

def min_angle_diff(a, b):
    return min((a - b) % (2 * np.pi), (b - a) % (2 * np.pi))

def calculate_pseudotime(pol_data, centered_data, save_dir=""):
    # note that pol_data is coord x cell, centered_data is cell x coord
    # find the index of the point that is closest 3/2 pi
    pol_data[1] = pol_data[1] - np.min(pol_data[1])
    lower_ind = np.argmin(np.array([min_angle_diff(x, 3 / 2 * np.pi) for x in pol_data[1]]))
    # print(pol_data.shape, centered_data.shape)
    # print(np.min(pol_data[1]), np.max(pol_data[1]))
    # print(np.max(centered_data[:, 1]), np.min(centered_data[:, 1]), centered_data[:, 1][lower_ind])

    # reindex phi so that the leftmost point is 0 (this is the most GMNN depleted and likely G1)
    leftmost_ind = np.argmin(centered_data[:, 0])
    # pol_data[0] = (pol_data[0] - pol_data[0][leftmost_ind]) % (2 * np.pi)
    # pol_data[0][pol_data[0] < 0] = 2 * np.pi + pol_data[0][pol_data[0] < 0]

    # plt.clf()
    # plt.scatter(centered_data[:, 0], centered_data[:, 1], c=pol_data[1] / (2 * np.pi), cmap="hsv")
    # plt.scatter(centered_data[lower_ind, 0], centered_data[lower_ind, 1], marker="*")
    # plt.scatter(centered_data[leftmost_ind, 0], centered_data[leftmost_ind, 1], marker="*")
    # plt.savefig("ping.png")

    pol_sort_inds = np.argsort(pol_data[1])
    leftmost_ind = np.where(pol_sort_inds == leftmost_ind)[0][0]
    lower_ind = np.where(pol_sort_inds == lower_ind)[0][0]
    pol_sort_rho = pol_data[0][pol_sort_inds]
    pol_sort_phi = pol_data[1][pol_sort_inds]
    centered_data_sort0 = centered_data[pol_sort_inds, 0]
    centered_data_sort1 = centered_data[pol_sort_inds, 1]

    # Rezero to minimum--reasoning, cells disappear during mitosis, so we should have the fewest detected cells there
    cc_props = FucciCellCycle()
    n_bins = int(cc_props.TOT_LEN / cc_props.M_LEN)
    n_bins_sector = int(n_bins / 8) # looking for something in the lower bottom left octant
    bins = plt.hist(pol_sort_phi[lower_ind:leftmost_ind], n_bins_sector)
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
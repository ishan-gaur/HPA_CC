"""
Takes as a parameter the name of the post-processed data version to use
Iterate well by well and get the total number of cells using the sc tensor first dimension
Need to load all the images in order of the index and apply the nuclear mask to get the total intensity
"""
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import math
import torch
import pickle as pkl
from tqdm import tqdm
from HPA_CC.data.dataset import DatasetFS
from HPA_CC.utils.pseudotime import intensities_to_pseudotime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import FUCCI_DS_PATH, FUCCI_NAME, OUTPUT_DIR
from multiprocessing import Pool
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()

fucci_ds = DatasetFS(FUCCI_DS_PATH)

images_per_well = [len(fucci_ds.images_in_well(well)) for well in fucci_ds.well_list]
plt.hist(images_per_well)
plt.savefig(OUTPUT_DIR / "images_per_well.png")
plt.xlabel("Number of images per well")
plt.ylabel("Number of wells")
plt.title(f"Distribution of Images per Well in the FUCCI dataset")
plt.close()

# plot a grid of the polar distrbution of each well
n_wells = len(fucci_ds.well_list)
if (OUTPUT_DIR / "well_intensity_cache.pkl").exists() and not args.rebuild:
    cells_per_well, well_intensities, well_pseudotimes, well_angles = pkl.load(open(OUTPUT_DIR / "well_intensity_cache.pkl", "rb"))
else:
    def process_well(well):
        sc_images = torch.load(well / f"images_{FUCCI_NAME}.pt") # Cells x Channels x H x W
        nuclei_masks = torch.load(well / f"nuclei_masks_{FUCCI_NAME}.pt") # Cells x H x W
        sc_nuclei = sc_images * nuclei_masks[:, None]
        mean_intensities = torch.sum(sc_nuclei[:, 2:], dim=(2, 3)) / torch.sum(nuclei_masks[:, None], dim=(2, 3)) # only calculating for GMNN and CDT1
        min_nonzero_GMNN = torch.min(mean_intensities[:, 0][mean_intensities[:, 0] > 0])
        min_nonzero_CDT1 = torch.min(mean_intensities[:, 1][mean_intensities[:, 1] > 0])
        log_mean_GMNN = torch.log(mean_intensities[:, 0] + min_nonzero_GMNN)
        log_mean_CDT1 = torch.log(mean_intensities[:, 1] + min_nonzero_CDT1)
        log_mean_fucci_intensities = torch.stack((log_mean_GMNN, log_mean_CDT1), dim=1)
        fucci_time, raw_time, center = intensities_to_pseudotime(log_mean_fucci_intensities.numpy())
        return len(sc_images), log_mean_fucci_intensities, fucci_time, raw_time

    with Pool(16) as pool:
        results = list(tqdm(pool.imap(process_well, fucci_ds.well_list), total=n_wells, desc="Plotting Pseudotime distributions"))

    cells_per_well, well_intensities, well_pseudotimes, well_angles = zip(*results)

    pkl.dump((cells_per_well, well_intensities, well_pseudotimes, well_angles), open(OUTPUT_DIR / "well_intensity_cache.pkl", "wb"))

# n_col = 12
# n_row = math.ceil(n_wells / n_col)
# plt.clf()
# fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4), sharex=True, sharey=True)
# for i, well, cells, intensities, pseudotime in tqdm(zip(range(n_wells), fucci_ds.well_list, cells_per_well, well_intensities, well_pseudotimes), total=n_wells, desc="Plotting Pseudotime Colors"):
#     ax = axes[i // n_col, i % n_col]
#     ax.set_title(f"{well.name} n={cells}")
#     ax.scatter(intensities[:, 0], intensities[:, 1], c=pseudotime, cmap="RdYlGn")
# plt.savefig(OUTPUT_DIR / "fucci_plot_pseudotimes.png")
# plt.close()


# bin_res = 0.05
# pseudotime_bins = np.arange(0, 1 + bin_res, bin_res)
# n_col = 12
# n_row = math.ceil(n_wells / n_col)
# plt.clf()
# fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4), sharex=True, sharey=True)
# for i, well, cells, intensities, pseudotime in zip(range(n_wells), fucci_ds.well_list, cells_per_well, well_intensities, well_pseudotimes):
#     binned_pseudotime = (np.digitize(pseudotime, pseudotime_bins) - 0.5) * bin_res
#     well_df = pd.DataFrame({"GMNN": intensities[:, 0], "CDT1": intensities[:, 1], "pseudotime": binned_pseudotime})
#     ax = axes[i // n_col, i % n_col]
#     ax.set_title(f"{well.name} n={cells}")
#     sns.lineplot(x="pseudotime", y="GMNN", data=well_df, ax=ax, color="green")
#     sns.lineplot(x="pseudotime", y="CDT1", data=well_df, ax=ax, color="red")
# plt.savefig(OUTPUT_DIR / "average_intensity_pseudotime.png")
# plt.close()

n_col = 12
n_row = math.ceil(n_wells / n_col)
plt.clf()
fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4), sharex=True, sharey=False)
gmm_well_params = []
sample_classes = []
for i, well, cells, intensities, angles in tqdm(zip(range(n_wells), fucci_ds.well_list, cells_per_well, well_intensities, well_angles), total=n_wells, desc="Plotting Angular Distributions"):
    ax = axes[i // n_col, i % n_col]
    ax.set_title(f"{well.name} n={cells}")
    sns.histplot(angles, ax=ax, kde=False, stat="density", bins=50)

    gmm = GaussianMixture(n_components=3, covariance_type='spherical', n_init=10, init_params='k-means++')
    gmm.fit(angles.reshape(-1, 1))

    x = np.linspace(0, 1, 1000)
    y = np.exp(gmm._estimate_weighted_log_prob(x.reshape(-1, 1)))

    ax.plot(x, y[:, 0].flatten(), color='blue', label='Mode 1')
    ax.plot(x, y[:, 1].flatten(), color='orange', label='Mode 2')
    ax.plot(x, y[:, 2].flatten(), color='green', label='Mode 3')

    likelihood = np.mean(gmm.score_samples(x.reshape(-1, 1)))
    ax.set_title(f"{well.name} n={cells} l={likelihood:.2f}")
    ax.legend()

    weights, means, vars = gmm.weights_.copy(), gmm.means_.flatten().copy(), gmm.covariances_.flatten().copy()
    # Sort the Gaussian components by means
    sorted_indices = np.argsort(means)
    weights = weights[sorted_indices]
    means = means[sorted_indices]
    vars = vars[sorted_indices]

    gmm_well_params.append(np.concatenate([weights, means, vars]))
    sample_classes.append(np.argmax(gmm._estimate_weighted_log_prob(angles.reshape(-1, 1))[:, sorted_indices], axis=1))
plt.savefig(OUTPUT_DIR / "angular_distribution.png")
plt.close()

n_col = 12
n_row = math.ceil(n_wells / n_col)
plt.clf()
fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4), sharex=True, sharey=False)
for i, well, cells, intensities, angles, sample_class in tqdm(zip(range(n_wells), fucci_ds.well_list, cells_per_well, well_intensities, well_angles, sample_classes), total=n_wells, desc="Plotting Sample Classes"):
    ax = axes[i // n_col, i % n_col]
    ax.set_title(f"{well.name} n={cells}")
    sns.scatterplot(x=intensities[:, 0], y=intensities[:, 1], hue=sample_class, ax=ax)
plt.savefig(OUTPUT_DIR / "sample_classes.png")
plt.close()


gmm_well_params = np.array(gmm_well_params)
print("Cluster Mean and Medians")
print("Mean:\t", np.mean(gmm_well_params, axis=0))
print("StdDev:\t", np.std(gmm_well_params, axis=0))
print("Median:\t", np.median(gmm_well_params, axis=0))
pca = PCA(n_components=2)
pca_result = pca.fit_transform(gmm_well_params)
print(pca.explained_variance_ratio_)

plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of GMM Well Parameters")
plt.savefig(OUTPUT_DIR / "pca_gmm_well_params.png")
plt.close()

total_cells = np.sum(cells_per_well)
plt.hist(cells_per_well)
plt.xlabel("Number of cells per well")
plt.ylabel("Number of wells")
plt.title(f"Distribution of Cells per Well in the FUCCI dataset (Total Cells: {total_cells})")
plt.savefig(OUTPUT_DIR / "cells_per_well.png")
plt.close()

scopes = set([well.name.split("--")[0] for well in fucci_ds.well_list])
scope_gmms = {s: GaussianMixture(n_components=3, covariance_type='spherical', n_init=10, init_params='k-means++') for s in scopes}
scope_angles = {s: [] for s in scopes}
scope_mode_order = {}
for i, well, angles in zip(range(n_wells), fucci_ds.well_list, well_angles):
    scope = well.name.split("--")[0]
    scope_angles[scope].append(angles)
for scope in scopes:
    scope_angles[scope] = np.concatenate(scope_angles[scope])
    scope_gmms[scope].fit(scope_angles[scope].reshape(-1, 1))
    scope_mode_order[scope] = np.argsort(scope_gmms[scope].means_.flatten())
scope_sample_classes = []
for i, well, angles in tqdm(zip(range(n_wells), fucci_ds.well_list, well_angles), total=n_wells, desc="Predicting Scope-level Sample Classes"):
    scope = well.name.split("--")[0]
    sample_class = np.argmax(scope_gmms[scope]._estimate_weighted_log_prob(angles.reshape(-1, 1))[:, scope_mode_order[scope]], axis=1)
    scope_sample_classes.append(sample_class)

n_col = 12
n_row = math.ceil(n_wells / n_col)
plt.clf()
fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4), sharex=True, sharey=False)
for i, well, cells, intensities, angles, scope_class, sample_class in tqdm(zip(range(n_wells), fucci_ds.well_list, cells_per_well, well_intensities, well_angles, scope_sample_classes, sample_classes), total=n_wells, desc="Plotting Scope-level Sample Classes"):
    ax = axes[i // n_col, i % n_col]
    sns.scatterplot(x=intensities[:, 0], y=intensities[:, 1], hue=scope_class, ax=ax)
    diff_sample_idxs = (sample_class != scope_class)
    ax.scatter(intensities[diff_sample_idxs, 0], intensities[diff_sample_idxs, 1], marker='x', color='red', alpha=0.2)
    ax.set_title(f"{well.name} n={cells}, error={np.sum(diff_sample_idxs)}")
plt.savefig(OUTPUT_DIR / "scope_sample_classes.png")
plt.close()
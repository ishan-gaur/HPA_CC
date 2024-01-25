"""
Takes as a parameter the name of the post-processed data version to use
Iterate well by well and get the total number of cells using the sc tensor first dimension
Need to load all the images in order of the index and apply the nuclear mask to get the total intensity
"""

import math
import torch
import pickle as pkl
from tqdm import tqdm
from HPA_CC.data.dataset import DatasetFS
from HPA_CC.utils.pseudotime import intensities_to_pseudotime
import matplotlib.pyplot as plt
from config import FUCCI_DS_PATH, FUCCI_NAME, OUTPUT_DIR

fucci_ds = DatasetFS(FUCCI_DS_PATH)

images_per_well = [len(fucci_ds.images_in_well(well)) for well in fucci_ds.well_list]
plt.hist(images_per_well)
plt.savefig(OUTPUT_DIR / "images_per_well.png")
plt.xlabel("Number of images per well")
plt.ylabel("Number of wells")
plt.title("Distribution of Images per Well in the FUCCI dataset")
plt.close()

# plot a grid of the polar distrbution of each well
n_wells = len(fucci_ds.well_list)
if (OUTPUT_DIR / "well_intensity_cache.pkl").exists() and False:
    cells_per_well, well_intensities, well_pseudotimes = pkl.load(open(OUTPUT_DIR / "well_intensity_cache.pkl", "rb"))
else:
    cells_per_well = []
    well_intensities = []
    well_pseudotimes = []
    for well in tqdm(fucci_ds.well_list, total=n_wells, desc="Plotting Pseudotime distributions"):
        sc_images = torch.load(well / f"images_{FUCCI_NAME}.pt") # Cells x Channels x H x W
        nuclei_masks = torch.load(well / f"nuclei_masks_{FUCCI_NAME}.pt") # Cells x H x W
        sc_nuclei = sc_images * nuclei_masks[:, None]
        mean_intensities = torch.sum(sc_nuclei[:, 2:], dim=(2, 3)) / torch.sum(nuclei_masks[:, None], dim=(2, 3)) # only calculating for GMNN and CDT1
        min_nonzero_GMNN = torch.min(mean_intensities[:, 0][mean_intensities[:, 0] > 0])
        min_nonzero_CDT1 = torch.min(mean_intensities[:, 1][mean_intensities[:, 1] > 0])
        log_mean_GMNN = torch.log(mean_intensities[:, 0] + min_nonzero_GMNN)
        log_mean_CDT1 = torch.log(mean_intensities[:, 1] + min_nonzero_CDT1)
        log_mean_fucci_intensities = torch.stack((log_mean_GMNN, log_mean_CDT1), dim=1)
        pseudotime, center = intensities_to_pseudotime(log_mean_fucci_intensities.numpy())
        cells_per_well.append(len(sc_images))
        well_intensities.append(log_mean_fucci_intensities)
        # well_intensities.append(log_mean_fucci_intensities - center)
        well_pseudotimes.append(pseudotime)
    pkl.dump((cells_per_well, well_intensities, well_pseudotimes), open(OUTPUT_DIR / "well_intensity_cache.pkl", "wb"))

print(n_wells, "wells")
n_col = 12
n_row = math.ceil(n_wells / n_col)
plt.clf()
fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4), sharex=True, sharey=True)
for i, well, cells, intensities, pseudotime in zip(range(n_wells), fucci_ds.well_list, cells_per_well, well_intensities, well_pseudotimes):
    ax = axes[i // n_col, i % n_col]
    ax.set_title(f"{well.name} n={cells}")
    ax.scatter(intensities[:, 0], intensities[:, 1], c=pseudotime, cmap="RdYlGn")
plt.savefig(OUTPUT_DIR / "pseudotime_distributions.png")
plt.close()

plt.hist(cells_per_well)
plt.xlabel("Number of cells per well")
plt.ylabel("Number of wells")
plt.title("Distribution of Cells per Well in the FUCCI dataset")
plt.savefig(OUTPUT_DIR / "cells_per_well.png")
plt.close()
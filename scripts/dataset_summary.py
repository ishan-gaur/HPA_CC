# -*- coding: utf-8 -*-
"""Produces a summary of the intensity statistics for the FUCCI dataset as specified in the config file.
Plots PCAs for the images, colored by each well and scope. Also plots PCAs for the average intensity of each well.
Finally, it writes a sorted list of the image directories to a pickle file, to be used by other scripts as well.
"""
from pathlib import Path
import pickle as pkl
from tqdm import tqdm
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from config import FUCCI_DS_PATH
from HPA_CC.utils.dataset import Dataset

fucci_ds = Dataset(FUCCI_DS_PATH)

OUTPUT_DIR = Path.cwd() / "scripts" / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

well_type_dict = {}

for f in fucci_ds.well_list:
    well_type = f.name.split('--')[0]
    if not well_type in well_type_dict:
        well_type_dict[well_type] = []
    well_type_dict[well_type].append(len(list(filter(lambda d: d.is_dir(), f.iterdir()))))

print("Number of images per well, grouped by microscope/run")
for scope, well_img_cts in well_type_dict.items():
    print(scope)
    print(f"\tNumber of wells: {len(well_img_cts)}")
    print(f"\tMean number of images per well: {int(np.mean(well_img_cts))}")
    print(f"\t{well_img_cts}")

bits = 8
percentiles = np.arange(0, 100, 100 / (2 ** bits))

intensity_files = ["nuc_percentiles.npy", "mt_percentiles.npy", "cdt1_percentiles.npy", "gmnn_percentiles.npy",
                   "microscopes_wells.pkl"]
cached_intensities_exist = all([Path.exists(OUTPUT_DIR / f) for f in intensity_files])
if not cached_intensities_exist:
    microscopes, wells = [], []
    nuc_percentiles, mt_percentiles, cdt1_percentiles, gmnn_percentiles = [[] for _ in range(4)]
    for img_dir in tqdm(fucci_ds.image_list, desc="Collecting image intensities"):
        well_dir = img_dir.parent
        microscopes.append(well_dir.name.split('--')[0])
        wells.append(well_dir.name)

        def get_percentiles_nonzero(img_path):
            img = np.array(Image.open(img_path))
            img = img.flatten()
            img = img[img > 0]
            return np.percentile(img, percentiles)

        nuc = get_percentiles_nonzero(img_dir / "nuclei.png")
        mt = get_percentiles_nonzero(img_dir / "microtubule.png")
        cdt1 = get_percentiles_nonzero(img_dir / "CDT1.png")
        gmnn = get_percentiles_nonzero(img_dir / "Geminin.png")

        nuc_percentiles.append(np.percentile(nuc, percentiles))
        mt_percentiles.append(np.percentile(mt, percentiles))
        cdt1_percentiles.append(np.percentile(cdt1, percentiles))
        gmnn_percentiles.append(np.percentile(gmnn, percentiles))

        if len(nuc_percentiles) > 10:
            break

    # sort all the lists by image directory path
    pkl.dump((microscopes, wells), open(OUTPUT_DIR / "microscopes_wells.pkl", "wb"))

    nuc_percentiles = np.array(nuc_percentiles)
    mt_percentiles = np.array(mt_percentiles)
    cdt1_percentiles = np.array(cdt1_percentiles)
    gmnn_percentiles = np.array(gmnn_percentiles)

    np.save(OUTPUT_DIR / "nuc_percentiles.npy", nuc_percentiles)
    np.save(OUTPUT_DIR / "mt_percentiles.npy", mt_percentiles)
    np.save(OUTPUT_DIR / "cdt1_percentiles.npy", cdt1_percentiles)
    np.save(OUTPUT_DIR / "gmnn_percentiles.npy", gmnn_percentiles)
else:
    microscopes, wells = pkl.load(open(OUTPUT_DIR / "microscopes_wells.pkl", "rb"))
    nuc_percentiles = np.load(OUTPUT_DIR / "nuc_percentiles.npy")
    mt_percentiles = np.load(OUTPUT_DIR / "mt_percentiles.npy")
    cdt1_percentiles = np.load(OUTPUT_DIR / "cdt1_percentiles.npy")
    gmnn_percentiles = np.load(OUTPUT_DIR / "gmnn_percentiles.npy")

pca = PCA(n_components=2)
scaler = StandardScaler()

ref_intensities = np.concatenate((nuc_percentiles, mt_percentiles), axis=1)
ref_int_pca = pca.fit_transform(scaler.fit_transform(ref_intensities))

pca_ref_int_df = pd.DataFrame({"PC1": ref_int_pca[:, 0], "PC2": ref_int_pca[:, 1], "microscope": microscopes})
sns.scatterplot(x="PC1", y="PC2", hue="microscope", data=pca_ref_int_df, alpha=0.5)
plt.title("PCA of Image Intensity Histograms")
plt.legend(title="Microscope")
plt.savefig(OUTPUT_DIR / "pca_scope.png")
plt.clf()

pca_ref_int_df = pd.DataFrame({"PC1": ref_int_pca[:, 0], "PC2": ref_int_pca[:, 1], "well_nums": wells})
sns.scatterplot(x="PC1", y="PC2", hue="well_nums", data=pca_ref_int_df, alpha=0.5, legend=False)
plt.title("PCA of Image Intensity Histograms")
plt.savefig(OUTPUT_DIR / "pca_well.png")
plt.clf()

well_averages = []
scope_idx = {m: i for i, m in enumerate(set(microscopes))}
well_idx = {w: i for i, w in enumerate(set(wells))}
well_nums = np.array([well_idx[w] for w in wells])
for w in set(wells):
    well_averages.append(np.mean(ref_intensities[well_nums == well_idx[w]], axis=0))
well_averages = pca.fit_transform(scaler.fit_transform(np.array(well_averages)))
wells_types = list(set(wells))
well_scopes = np.array([scope_idx[w.split('--')[0]] for w in wells_types])
well_scope_nums = np.array([scope_idx[w.split('--')[0]] for w in wells_types])
pca_well_avg_df = pd.DataFrame({"PC1": well_averages[:, 0], "PC2": well_averages[:, 1], "scope": [wells_types[i] for i in well_scope_nums]})
sns.scatterplot(x="PC1", y="PC2", hue="scope", data=pca_well_avg_df, alpha=0.5)
plt.title("Well-level Average of Non-zero Pixel Intensities")
plt.legend(title="Microscope")
plt.savefig(OUTPUT_DIR / "pca_well_averages.png")
plt.clf()
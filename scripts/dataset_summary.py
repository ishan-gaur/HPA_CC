# -*- coding: utf-8 -*-
"""Produces a summary of the intensity statistics for the FUCCI dataset as specified in the config file.
Plots PCAs for the images, colored by each well and scope. Also plots PCAs for the average intensity of each well.
Then, it writes a sorted list of the image directories to a pickle file, to be used by other scripts as well.

It also creates PCA plots of the well- and image-level intensity distributions for each microscope in the dataset.
This is primarily useful in designing useful training splits for the pseudotime models down the line.
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

from config import FUCCI_DS_PATH, HPA_DS_PATH, OUTPUT_DIR
from HPA_CC.data.dataset import DatasetFS
from multiprocessing import Pool

fucci_ds = DatasetFS(FUCCI_DS_PATH)
hpa_ds = DatasetFS(HPA_DS_PATH)

well_type_dict = {}

for f in fucci_ds.well_list:
    well_type = f.name.split('--')[0]
    if not well_type in well_type_dict:
        well_type_dict[well_type] = []
    well_type_dict[well_type].append(len(list(filter(lambda d: d.is_dir(), f.iterdir()))))

well_type_dict["hpa"] = []
for f in hpa_ds.well_list:
    well_type_dict["hpa"].append(len(list(filter(lambda d: d.is_dir(), f.iterdir()))))
    num_images = len(list(filter(lambda d: d.is_dir(), f.iterdir())))

print("Number of images per well, grouped by microscope/run")
for scope, well_img_cts in well_type_dict.items():
    print(scope)
    print(f"\tNumber of wells: {len(well_img_cts)}")
    print(f"\tMean number of images per well: {np.mean(well_img_cts):.1f}")
    print(f"\t{well_img_cts}")

bits = 8
percentiles = np.arange(0, 100, 100 / (2 ** bits))

intensity_files = ["nuc_percentiles.npy", "mt_percentiles.npy", "microscopes_wells.pkl"]
for dataset_name, dataset in zip(["fucci", "hpa"], [fucci_ds, hpa_ds]):
    cached_intensities_exist = all([Path.exists(OUTPUT_DIR / f"{dataset_name}_{f}") for f in intensity_files])
    if not cached_intensities_exist:
        microscopes, wells = [], []
        nuc_percentiles, mt_percentiles = [[] for _ in range(2)]
        if dataset_name == "fucci":
            cdt1_percentiles, gmnn_percentiles = [[] for _ in range(2)]
        if dataset_name == "hpa":
            er_percentiles = []

        def get_percentiles_nonzero(img_path):
            img = np.array(Image.open(img_path))
            img = img.flatten()
            img = img[img > 0]
            return np.percentile(img, percentiles)

        def process_image(img_dir):
            well_dir = img_dir.parent
            if dataset_name == "fucci":
                microscopes.append(well_dir.name.split('--')[0])
            else:
                microscopes.append("hpa")
            wells.append(well_dir.name)

            if dataset_name == "fucci":
                nuc = get_percentiles_nonzero(img_dir / "nuclei.png")
                mt = get_percentiles_nonzero(img_dir / "microtubule.png")
                cdt1 = get_percentiles_nonzero(img_dir / "CDT1.png")
                gmnn = get_percentiles_nonzero(img_dir / "Geminin.png")
            if dataset_name == "hpa":
                nuc = get_percentiles_nonzero(img_dir / f"{img_dir.name}_blue.png")
                mt = get_percentiles_nonzero(img_dir / f"{img_dir.name}_red.png")
                er = get_percentiles_nonzero(img_dir / f"{img_dir.name}_yellow.png")

            nuc_percentiles.append(np.percentile(nuc, percentiles))
            mt_percentiles.append(np.percentile(mt, percentiles))
            if dataset_name == "fucci":
                cdt1_percentiles.append(np.percentile(cdt1, percentiles))
                gmnn_percentiles.append(np.percentile(gmnn, percentiles))
            if dataset_name == "hpa":
                er_percentiles.append(np.percentile(er, percentiles))

        with Pool(16) as pool:
            pool.map(process_image, dataset.image_list)

        pkl.dump((microscopes, wells), open(OUTPUT_DIR / f"{dataset_name}_microscopes_wells.pkl", "wb"))

        nuc_percentiles = np.array(nuc_percentiles)
        mt_percentiles = np.array(mt_percentiles)
        np.save(OUTPUT_DIR / f"{dataset_name}_nuc_percentiles.npy", nuc_percentiles)
        np.save(OUTPUT_DIR / f"{dataset_name}_mt_percentiles.npy", mt_percentiles)

        if dataset_name == "fucci":
            cdt1_percentiles = np.array(cdt1_percentiles)
            gmnn_percentiles = np.array(gmnn_percentiles)
            np.save(OUTPUT_DIR / f"{dataset_name}_cdt1_percentiles.npy", cdt1_percentiles)
            np.save(OUTPUT_DIR / f"{dataset_name}_gmnn_percentiles.npy", gmnn_percentiles)
        if dataset_name == "hpa":
            er_percentiles = np.array(er_percentiles)
            np.save(OUTPUT_DIR / f"{dataset_name}_er_percentiles.npy", er_percentiles)

fucci_microscopes, fucci_wells = pkl.load(open(OUTPUT_DIR / f"fucci_microscopes_wells.pkl", "rb"))
fucci_nuc_percentiles = np.load(OUTPUT_DIR / f"fucci_nuc_percentiles.npy")
fucci_mt_percentiles = np.load(OUTPUT_DIR / f"fucci_mt_percentiles.npy")
fucci_cdt1_percentiles = np.load(OUTPUT_DIR / f"fucci_cdt1_percentiles.npy")
fucci_gmnn_percentiles = np.load(OUTPUT_DIR / f"fucci_gmnn_percentiles.npy")

hpa_microscopes, hpa_wells = pkl.load(open(OUTPUT_DIR / f"hpa_microscopes_wells.pkl", "rb"))
hpa_nuc_percentiles = np.load(OUTPUT_DIR / f"hpa_nuc_percentiles.npy")
hpa_mt_percentiles = np.load(OUTPUT_DIR / f"hpa_mt_percentiles.npy")
hpa_er_percentiles = np.load(OUTPUT_DIR / f"hpa_er_percentiles.npy")

microscopes = list(fucci_microscopes) + list(hpa_microscopes)
wells = list(fucci_wells) + list(hpa_wells)
nuc_percentiles = np.concatenate((fucci_nuc_percentiles, hpa_nuc_percentiles), axis=0)
mt_percentiles = np.concatenate((fucci_mt_percentiles, hpa_mt_percentiles), axis=0)


pca = PCA(n_components=2)
scaler = StandardScaler()

ref_intensities = np.concatenate((nuc_percentiles, mt_percentiles), axis=1)
ref_int_pca = pca.fit_transform(scaler.fit_transform(ref_intensities))

pca_ref_int_df = pd.DataFrame({"PC1": ref_int_pca[:, 0], "PC2": ref_int_pca[:, 1], "microscope": microscopes})
fig = plt.figure(figsize=(10, 10))
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
def well_to_scope(w):
    if not w.split('--')[0] in scope_idx:
        return "hpa"
    return w.split('--')[0]
scope_idx = {m: i for i, m in enumerate(set(microscopes))}
well_idx = {w: i for i, w in enumerate(set(wells))}
well_nums = np.array([well_idx[w] for w in wells])
for w in set(wells):
    well_averages.append(np.mean(ref_intensities[well_nums == well_idx[w]], axis=0))
well_averages = pca.fit_transform(scaler.fit_transform(np.array(well_averages)))
wells_types = list(set(wells))
well_scopes = []
well_scopes = np.array([well_to_scope(w) for w in wells_types])
well_scope_nums = np.array([scope_idx[well_to_scope(w)] for w in wells_types])
pca_well_avg_df = pd.DataFrame({"PC1": well_averages[:, 0], "PC2": well_averages[:, 1], "scope": well_scopes})
sns.scatterplot(x="PC1", y="PC2", hue="scope", data=pca_well_avg_df, alpha=0.5)
plt.title("Well-level Average of Non-zero Pixel Intensities")
plt.legend(title="Microscope")
plt.savefig(OUTPUT_DIR / "pca_well_averages.png")
plt.clf()
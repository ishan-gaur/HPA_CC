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

from config import OUTPUT_DIR, FUCCI_DS_PATH, HPA_DS_PATH
from HPA_CC.data.dataset import DatasetFS
from multiprocessing import Pool, Manager


def load_dataset_fs(dataset_name):
    if dataset_name == "fucci":
        return DatasetFS(FUCCI_DS_PATH)
    elif dataset_name == "hpa":
        return DatasetFS(HPA_DS_PATH)
    else:
        raise ValueError(f"Invalid dataset name {dataset_name}")

# goes through the dataset and computes the intensity percentiles for each image's channels
# these are then saved to a file for later use, 
def get_dataset(dataset_name):
    dataset = load_dataset_fs(dataset_name)
    intensity_files = ["nuc_percentiles.npy", "mt_percentiles.npy", "microscopes_wells.pkl"]
    cached_intensities_exist = all([Path.exists(OUTPUT_DIR / f"{dataset_name}_{f}") for f in intensity_files])

    if cached_intensities_exist:
        microscopes, wells = pkl.load(open(OUTPUT_DIR / f"{dataset_name}_microscopes_wells.pkl", "rb"))
        nuc_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_nuc_percentiles.npy")
        mt_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_mt_percentiles.npy")
        if dataset_name == "fucci":
            cdt1_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_cdt1_percentiles.npy")
            gmnn_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_gmnn_percentiles.npy")
            return microscopes, wells, nuc_percentiles, mt_percentiles, cdt1_percentiles, gmnn_percentiles
        elif dataset_name == "hpa":
            er_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_er_percentiles.npy")
            return microscopes, wells, nuc_percentiles, mt_percentiles, er_percentiles
        else:
            raise ValueError(f"Invalid dataset name {dataset_name}")
        
    manager = Manager()
    microscopes = manager.list()
    wells = manager.list()

    # define lists for channels needed per dataset
    nuc_percentiles = manager.list()
    mt_percentiles = manager.list()
    if dataset_name == "fucci":
        cdt1_percentiles = manager.list()
        gmnn_percentiles = manager.list()
        percentile_lists = [nuc_percentiles, mt_percentiles, cdt1_percentiles, gmnn_percentiles]
    if dataset_name == "hpa":
        er_percentiles = manager.list()
        percentile_lists = [nuc_percentiles, mt_percentiles, er_percentiles]

    # parallelize over images directly and add the percentiles to the lists above
    with Pool(16) as pool:
        tasks = []
        for img_dir in dataset.image_list:
            tasks.append(pool.apply_async(process_image, (img_dir, dataset_name, microscopes, wells, percentile_lists)))
        for task in tqdm(tasks, desc=f"Processing {dataset_name} images"):
            task.get()

    # save the outputs to files, different files depending on the dataset
    microscopes = list(microscopes)
    wells = list(wells)
    nuc_percentiles = list(nuc_percentiles)
    mt_percentiles = list(mt_percentiles)
    if dataset_name == "fucci":
        cdt1_percentiles = list(cdt1_percentiles)
        gmnn_percentiles = list(gmnn_percentiles)
    if dataset_name == "hpa":
        er_percentiles = list(er_percentiles)

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

    if dataset_name == "fucci":
        return microscopes, wells, nuc_percentiles, mt_percentiles, cdt1_percentiles, gmnn_percentiles
    elif dataset_name == "hpa":
        return microscopes, wells, nuc_percentiles, mt_percentiles, er_percentiles

# below are the helper functions that process each image and populate the lists above
# note that the nuclear and microtubule channels have difference extraction code due to 
# differences in the naming conventions between the two datasets

def process_image(img_dir, dataset_name, microscopes, wells, percentile_array_lists):
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
        image_percentiles = [nuc, mt, cdt1, gmnn]
    if dataset_name == "hpa":
        nuc = get_percentiles_nonzero(img_dir / f"{img_dir.name}_blue.png")
        mt = get_percentiles_nonzero(img_dir / f"{img_dir.name}_red.png")
        er = get_percentiles_nonzero(img_dir / f"{img_dir.name}_yellow.png")
        image_percentiles = [nuc, mt, er]

    for percentile_array, channel_percentiles in zip(percentile_array_lists, image_percentiles):
        percentile_array.append(channel_percentiles)

def get_percentiles_nonzero(img_path):
    bits = 8
    percentiles = np.arange(0, 100, 100 / (2 ** bits))
    img = np.array(Image.open(img_path))
    img = img.flatten()
    img = img[img > 0]
    return np.percentile(img, percentiles)

# below are the functions that load the data from the files created above

def load_microscopes_wells():
    fucci_m, fucci_w = load_fucci_microscopes_wells()
    hpa_m, hpa_w = load_hpa_microscopes_wells()
    return fucci_m + hpa_m, fucci_w + hpa_w

def load_fucci_microscopes_wells():
    return load_dataset_microscopes_wells("fucci")

def load_hpa_microscopes_wells():
    return load_dataset_microscopes_wells("hpa")

def load_dataset_microscopes_wells(dataset_name):
    if not Path.exists(OUTPUT_DIR / f"{dataset_name}_microscopes_wells.pkl"):
        dataset = load_dataset_fs(dataset_name)
        microscopes, wells = [], []
        for well_dir in dataset.well_list:
            if dataset_name == "fucci":
                microscopes.append(well_dir.name.split('--')[0])
            elif dataset_name == "hpa":
                microscopes.append("hpa")
            wells.append(well_dir.name)
        pkl.dump((microscopes, wells), open(OUTPUT_DIR / f"{dataset_name}_microscopes_wells.pkl", "wb"))
    microscopes, wells = pkl.load(open(OUTPUT_DIR / f"{dataset_name}_microscopes_wells.pkl", "rb"))
    return microscopes, wells

def load_fucci_percentiles():
    return load_dataset_percentiles("fucci")

def load_hpa_percentiles():
    return load_dataset_percentiles("hpa")

def load_dataset_percentiles(dataset_name):
    nuc_exists = Path.exists(OUTPUT_DIR / f"{dataset_name}_nuc_percentiles.npy")
    mt_exists = Path.exists(OUTPUT_DIR / f"{dataset_name}_mt_percentiles.npy")
    if dataset_name == "fucci":
        cdt1_exists = Path.exists(OUTPUT_DIR / f"{dataset_name}_cdt1_percentiles.npy")
        gmnn_exists = Path.exists(OUTPUT_DIR / f"{dataset_name}_gmnn_percentiles.npy")
        perc_files_exist = nuc_exists and mt_exists and cdt1_exists and gmnn_exists
    elif dataset_name == "hpa":
        er_exists = Path.exists(OUTPUT_DIR / f"{dataset_name}_er_percentiles.npy")
        perc_files_exist = nuc_exists and mt_exists and er_exists

    if not perc_files_exist:
        get_dataset(dataset_name)

    nuc_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_nuc_percentiles.npy")
    mt_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_mt_percentiles.npy")

    if dataset_name == "fucci":
        cdt1_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_cdt1_percentiles.npy")
        gmnn_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_gmnn_percentiles.npy")
        return nuc_percentiles, mt_percentiles, cdt1_percentiles, gmnn_percentiles
    elif dataset_name == "hpa":
        er_percentiles = np.load(OUTPUT_DIR / f"{dataset_name}_er_percentiles.npy")
        return nuc_percentiles, mt_percentiles, er_percentiles

def well_percentile_averages(wells, microscopes, ref_intensities):
    well_averages = []
    def well_to_scope(w):
        if not w.split('--')[0] in scope_idx:
            return "hpa"
        return w.split('--')[0]
    scope_idx = {m: i for i, m in enumerate(set(microscopes))}
    well_idx = {w: i for i, w in enumerate(set(wells))}
    well_nums = np.array([well_idx[w] for w in wells])
    for w in set(wells):
        well_intensities = ref_intensities[well_nums == well_idx[w]]
        well_mean = np.mean(well_intensities, axis=0)
        well_averages.append(well_mean)
    wells_types = list(set(wells))
    well_scopes = np.array([well_to_scope(w) for w in wells_types])
    well_averages = np.array(well_averages)
    return well_averages, well_scopes
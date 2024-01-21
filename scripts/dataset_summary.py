from pathlib import Path
import pickle as pkl
from tqdm import tqdm

import numpy as np
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from config import FUCCI_DS_PATH

OUTPUT_DIR = Path.cwd() / "scripts" / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

well_type_dict = {}

for f in FUCCI_DS_PATH.iterdir():
    if f.is_dir() and f.name != "__pycache__":
        well_type = f.name.split('--')[0]
        if not well_type in well_type_dict:
            well_type_dict[well_type] = []
        well_type_dict[well_type].append(len(list(filter(lambda d: d.is_dir(), f.iterdir()))))

bits = 8
percentiles = np.arange(0, 100, 100 / (2 ** bits))

intensity_files = ["nuc_percentiles.npy", "mt_percentiles.npy", "cdt1_percentiles.npy", "gmnn_percentiles.npy"]
cached_intensities_exist = all([Path.exists(OUTPUT_DIR / f) for f in intensity_files])
num_comp = 0
if not cached_intensities_exist:
    num_images = sum([sum(well_type_dict[k]) for k in well_type_dict])
    microscopes, wells = [], []
    nuc_percentiles, mt_percentiles, cdt1_percentiles, gmnn_percentiles = [[] for _ in range(4)]
    for well_dir in tqdm(FUCCI_DS_PATH.iterdir(), total=num_images):
        if not well_dir.is_dir() or well_dir.name == "__pycache__":
            continue
        for img_dir in well_dir.iterdir():
            if not img_dir.is_dir():
                continue
            nuc = Image.open(img_dir / "nuclei.png")
            mt = Image.open(img_dir / "microtubule.png")
            cdt1 = Image.open(img_dir / "CDT1.png")
            gmnn = Image.open(img_dir / "Geminin.png")
            nuc_percentiles.append(np.percentile(nuc, percentiles))
            mt_percentiles.append(np.percentile(mt, percentiles))
            cdt1_percentiles.append(np.percentile(cdt1, percentiles))
            gmnn_percentiles.append(np.percentile(gmnn, percentiles))
            microscopes.append(well_dir.name.split('--')[0])
            wells.append(well_dir.name)

    nuc_percentiles = np.array(nuc_percentiles)
    mt_percentiles = np.array(mt_percentiles)
    cdt1_percentiles = np.array(cdt1_percentiles)
    gmnn_percentiles = np.array(gmnn_percentiles)

    np.save(OUTPUT_DIR / "nuc_percentiles.npy", nuc_percentiles)
    np.save(OUTPUT_DIR / "mt_percentiles.npy", mt_percentiles)
    np.save(OUTPUT_DIR / "cdt1_percentiles.npy", cdt1_percentiles)
    np.save(OUTPUT_DIR / "gmnn_percentiles.npy", gmnn_percentiles)

    pkl.dump((microscopes, wells), open(OUTPUT_DIR / "microscopes_wells.pkl", "wb"))
    assert len(microscopes) == len(wells) and len(wells) == num_images, f"{len(microscopes)} {len(wells)} {num_images}"
else:
    nuc_percentiles = np.load(OUTPUT_DIR / "nuc_percentiles.npy")
    mt_percentiles = np.load(OUTPUT_DIR / "mt_percentiles.npy")
    cdt1_percentiles = np.load(OUTPUT_DIR / "cdt1_percentiles.npy")
    gmnn_percentiles = np.load(OUTPUT_DIR / "gmnn_percentiles.npy")
    microscopes, wells = pkl.load(open(OUTPUT_DIR / "microscopes_wells.pkl", "rb"))

pca = PCA(n_components=2)
scaler = StandardScaler()

ref_intensities = np.concatenate((nuc_percentiles, mt_percentiles), axis=1)
ref_int_pca = pca.fit_transform(scaler.fit_transform(ref_intensities))

scope_idx = {m: i for i, m in enumerate(set(microscopes))}
microscope_nums = np.array([scope_idx[m] for m in microscopes])
plt.scatter(ref_int_pca[:, 0], ref_int_pca[:, 1], c=microscope_nums, alpha=0.5)
plt.savefig(OUTPUT_DIR / "pca_scope.png")
plt.clf()

well_idx = {w: i for i, w in enumerate(set(wells))}
well_nums = np.array([well_idx[w] for w in wells])
plt.scatter(ref_int_pca[:, 0], ref_int_pca[:, 1], c=well_nums, alpha=0.5)
plt.savefig(OUTPUT_DIR / "pca_well.png")
plt.clf()

well_averages = []
for w in set(wells):
    well_averages.append(np.mean(ref_intensities[well_nums == well_idx[w]], axis=0))
well_averages = pca.fit_transform(scaler.fit_transform(np.array(well_averages)))
well_scopes = np.array([scope_idx[w.split('--')[0]] for w in set(wells)])
well_scope_nums = np.array([scope_idx[w.split('--')[0]] for w in set(wells)])
plt.scatter(well_averages[:, 0], well_averages[:, 1], c=well_scope_nums, alpha=0.5)
plt.savefig(OUTPUT_DIR / "pca_well_averages.png")
plt.clf()
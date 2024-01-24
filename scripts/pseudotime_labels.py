"""
Takes as a parameter the name of the post-processed data version to use
Iterate well by well and get the total number of cells using the sc tensor first dimension
Need to load all the images in order of the index and apply the nuclear mask to get the total intensity
"""

from HPA_CC.data.dataset import DatasetFS
import matplotlib.pyplot as plt
from config import FUCCI_DS_PATH, OUTPUT_DIR

fucci_ds = DatasetFS(FUCCI_DS_PATH)

images_per_well = [len(fucci_ds.images_in_well(well)) for well in fucci_ds.well_list]
plt.hist(images_per_well)
plt.savefig(OUTPUT_DIR / "images_per_well.png")
plt.close()

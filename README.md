# Cell-Cycle Prediction Model for the Human Protein Atlas
Uses https://github.com/broadinstitute/DINO4Cells_code as a backbone to get representations for a downstream classifier.
Includes DINOv2 and ResNet baselines as well.
Subset of HPA data covered by CC data can be found here: TODO
FUCCI U2OS data can be found here: TODO (should this be by well or the original data? will need scripts in this repo for the conversion if not)

# Setup
1. Download datasets from __TODO__, add the locations to scripts/config.py
2. Create the appropriate conda environment with ```conda env create -f environment.yml```
3. Install remaining dependencies with pip by running ```./install.sh```. Note that you may have to give the script execution privileges using ```chmod +x install.sh```

# FUCCI Dataset
IF images of U2-OS FUCCI cells, with four channels: DAPI, $\gamma$-tubulin, Geminin (GMNN), and CDT1. GMNN and CDT1 are cell cycle markers.
Datasets are organized into two levels: wells and images. The datasets (HPA dataset too) are preprocessed using the preprocessing.py script. This allows users to specify how to remove artifacts from the segmentations, how large to make the crops, what strategies to use for normalization and filtering out of focus samples, etc.

# Scripts
## dataset_summary.py
This just produces a summary of the numeber of images per microscope type in the FUCCI and HPA datasets. It also creates PCA plots of the well- and image-level intensity distributions for each microscope in the dataset. This is primarily useful in designing useful training splits for the pseudotime models down the line.

## pseudotime_labels.py

# DevLog
2/3/24: Looks like bulk is better than per-scope, esp for tilescan. This is probably due to clustering on the angles instead of the 2D distribution. Regardless, I can train a model to regress the angles and have functions for transforming all this later. Should make a class that handles the standardization of time and houses all the intensities and everything--will probably use the DatasetFS internally. It should also be able to take the intensity data from the Cell-Cycle paper and do OT between the two distributions and calculate the equivalent pseudotime.

2/2/24: Can get a matrix of KL-divergences between the wells' 2D intensity distribution GMM fit to find clusters that can be combined for learning a combined GMM--but maybe the barycenter thing scale so it doesn't matter anyways
Looks like standardized intensity plots GMM'ed in bulk is not good enough.

1/23/24: __TODO__ need to use the dataset class from the HPA-embedding repo and add the percentile stuff there directly so I don't have a custom implementation in the dataset_summary script. Need to run the dataset pipeline on the HPA dataset again. Also need to document its use.

1/21/24: Somehow the well-level differences are gone T_T; looks like it was because I forgot to exclude empty pixels

1/20/24: Below are the stats of well # of images
{'chamber': [10, 15, 14, 12, 15, 3, 13, 9, 6, 16, 9, 15, 7, 10, 15, 5, 15, 8, 16, 8, 6, 16, 12, 16, 14, 11, 10, 5, 5, 12, 10, 9, 5, 13, 13, 9, 16, 13, 16, 7, 10, 15, 2, 7, 14, 12, 14, 6, 6, 11, 16, 6, 7, 9, 6, 10, 15, 14, 16, 13, 15, 15, 12, 11, 16, 6, 12, 15, 9, 11, 6, 15, 11, 13, 6, 7, 3, 5, 9, 5, 8, 12, 15, 6, 13, 16, 14, 6, 9, 13, 8, 11, 15, 5], 'tilescan1': [29, 41, 16, 46, 37, 16, 24, 48, 28, 16, 43, 48, 35, 30, 27, 30, 48, 23, 44, 38, 24, 32, 7, 35, 29, 32, 32, 24, 47, 20, 48, 37, 30], 'tilescan2': [31, 3], 'overview2': [26, 4, 2, 40, 7, 43], 'overview1': [40, 3, 2, 4, 26, 43, 7]}
Still unclear to me if these are all the right wells, especially the tilescans
I should write some code to do a few things:
- find the intensity stats for each image
- group by microscope and well
- check if the intrawell/interwell distances within a microscope as smaller/larger
- and then check within microscope is smaller than across
- distance in PCA mb??
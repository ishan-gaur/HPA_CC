# Cell-Cycle Prediction Model for the Human Protein Atlas
Uses https://github.com/broadinstitute/DINO4Cells_code as a backbone to get representations for a downstream classifier.
Includes DINOv2 and ResNet baselines as well.
Subset of HPA data covered by CC data can be found here: TODO
FUCCI U2OS data can be found here: TODO (should this be by well or the original data? will need scripts in this repo for the conversion if not)

# FUCCI Dataset
IF images of U2-OS FUCCI cells, with four channels: DAPI, $\gamma$-tubulin, Geminin (GMNN), and CDT1. GMNN and CDT1 are cell cycle markers. The dataset starts with 

# DevLog
1/23/24: __TODO__ need to use the dataset class from the HPA-embedding repo and add the percentile stuff there directly so I don't have a custom implementation in the dataset_summary script.

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
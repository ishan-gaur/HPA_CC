from pathlib import Path
import torch    
from lightning import LightningDataModule
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from microfilm.microplot import microshow
from microfilm.colorify import multichannel_to_rgb
from HPA_CC.data.pipeline import load_channel_names, load_dir_images, load_index_paths
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from glob import glob

class DatasetFS:
    """Class to index wells and images in a dataset in a reproducible manner for inital poking around.
    Assumes that other than folders listed in folder_excl, all other folders in data_dir are wells,
    and all other folders in each well are images.
    """
    def __init__(self, data_dir, folder_excl=["__pycache__"]):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.folder_excl = folder_excl
        self.well_list = self.get_wells()
        self.image_list = self.get_images()
    
    def get_wells(self):
        return sorted([d for d in self.data_dir.iterdir() if d.is_dir() and not d.name in self.folder_excl])

    def images_in_well(self, well):
        return sorted([d for d in well.iterdir() if d.is_dir() and not d.name in self.folder_excl])
    
    def get_images(self):
        image_list = []
        for well in self.well_list:
            image_list += self.images_in_well(well)
        return image_list

class CellImageDataset(Dataset):
    # images are C x H x W
    def __init__(self, index_file, channel_colors=None, channels=None, batch_size=500):
        data_dir = Path(index_file).parent
        self.data_dir = data_dir
        self.channel_names = load_channel_names(data_dir)
        # self.images = torch.concat(load_dir_images(index_file))
        # image_tensors = load_dir_images(index_file)
        image_paths, _, _ = load_index_paths(index_file)
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Loading dataset images"):
            image_tensors = []
            for image_path in image_paths[i:i+batch_size]:
                image_tensors.append(torch.load(Path(image_path)))
            image_tensors = torch.concat(image_tensors)
            if i == 0:
                self.images = image_tensors
            else:
                self.images = torch.concat((self.images, image_tensors))

        print(f"Loaded {len(self.images)} images from {len(image_paths)} files.")
        
        self.channel_colors = channel_colors
        self.channels = channels if channels is not None else list(range(len(self.channel_names)))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.channels is None:
            return self.images[idx]
        else:
            return self.images[idx, self.channels]

    def get_channel_names(self):
        if self.channels is None:
            return self.channel_names
        else:
            return [self.channel_names[i] for i in self.channels]

    def set_channel_colors(self, channel_colors):
        self.channel_colors = channel_colors

    def get_channel_colors(self):
        return self.channel_colors

    def __set_default_channel_colors__(self):
        if len(self.channel_names) == 1:
            self.channel_colors = ["gray"]
        elif len(self.channel_names) == 2:
            self.channel_colors = ["blue", "red"]
        elif len(self.channel_names) == 3:
            self.channel_colors = ["blue", "green", "red"]
        else:
            raise ValueError(f"channel_colors not set and not suitable defaults for {len(self.channel_names)} channels.")

    def view(self, idx):
        image = self.__getitem__(idx).cpu().numpy()
        if self.channel_colors is None:
            self.__set_default_channel_colors__()
        microshow(image, cmaps=self.channel_colors)

    def convert_to_rgb(self, i):
        nans = torch.sum(torch.isnan(self.__getitem__(i)))
        if nans > 0:
            print(f"Warning: {nans} NaNs in image {i}")
        rgb_image, _, _, _ = multichannel_to_rgb(self.__getitem__(i).numpy(), cmaps=self.channel_colors,
                                                 limits=(np.nanmin(self.__getitem__(i)), np.nanmax(self.__getitem__(i))))
        return torch.Tensor(rgb_image)

    def as_rgb(self, channel_colors=None, num_workers=1):
        if self.channel_colors is None:
            if channel_colors is not None:
                self.channel_colors = channel_colors
            else:
                self.__set_default_channel_colors__()
        assert len(self.channels) == len(self.channel_colors), "Number of channel colors and channels must be equal"
        device = self.images.device
        self.images.cpu()
        rgb_images = []
        for i in tqdm(range(self.__len__()), total=self.__len__(), desc="Converting to RGB"):
            rgb_image, _, _, _ = multichannel_to_rgb(self.__getitem__(i).numpy(), cmaps=self.channel_colors)
            rgb_images.append(torch.Tensor(rgb_image))
        rgb_images = torch.stack(rgb_images)
        rgb_images.to(device)
        rgb_images = rgb_images[..., :-1].permute(0, 3, 1, 2)
        # rgb_images = rgb_images.permute(0, 3, 1, 2)
        rgb_dataset = SimpleDataset(rgb_images)
        self.images.to(device)
        return rgb_dataset

class SimpleDataset(Dataset):
    def __init__(self, tensor=None, path=None) -> None:
        if path is not None:
            cache_files = list(path.parent.glob(f"{path.stem}-*.pt"))
            # print(cache_files)
            tensors = []
            for cache_file in tqdm(cache_files, desc="Loading SimpleDataset"):
                tensors.append(torch.load(cache_file))
            self.tensor = torch.cat(tensors)
        elif tensor is None:
            raise ValueError("Must provide either tensor or path")
        else:
            self.tensor = tensor
    
    def __getitem__(self, idx):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.size(0)

    def save(self, path, batch_size=5000):
        for i in tqdm(range(0, self.tensor.size(0), batch_size), total=self.tensor.size(0) // batch_size, desc="Saving SimpleDataset"):
            torch.save(self.tensor[i:min(self.tensor.size(0), i+batch_size)].clone(), path.with_stem(f"{path.stem}-{i}"))
            
    def has_cache_files(path):
        cache_files = list(path.parent.glob(f"{path.stem}-*.pt"))
        return len(cache_files) > 0
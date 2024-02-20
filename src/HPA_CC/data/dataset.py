from pathlib import Path
import torch    
from torch.utils.data import Dataset, DataLoader, random_split
from microfilm.microplot import microshow
from microfilm.colorify import multichannel_to_rgb
from HPA_CC.data.pipeline import load_channel_names, load_index_paths, silent
from lightning import LightningDataModule
from tqdm import tqdm
import numpy as np

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
        self.data_dir = Path(index_file).parent
        self.dataset_fs = DatasetFS(self.data_dir)
        image_paths, _, _ = load_index_paths(index_file)
        self.n_cells = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Loading dataset images"):
            image_tensors = []
            for image_path in image_paths[i:i+batch_size]:
                image_tensors.append(torch.load(Path(image_path)))
                self.n_cells.append(image_tensors[-1].shape[0])
            image_tensors = torch.concat(image_tensors)
            if i == 0:
                self.images = image_tensors
            else:
                self.images = torch.concat((self.images, image_tensors))

        # self.images is now a tensor of shape (N, C, H, W)

        if not silent: print(f"Loaded {len(self.images)} images from {len(image_paths)} files.")
        
        self.channels = channels if channels is not None else list(range(len(channels)))
        self.channel_colors = channel_colors if channel_colors is not None else None
        self.channel_names = load_channel_names(self.data_dir)
        self.channel_names = [self.channel_names[c] if c is not None else "Padding" for c in self.channels]
        assert self.channel_colors is None or (len(self.channels) == len(self.channel_colors)), "Number of channel colors and channels must be equal"

        images_select_shape = list(self.images.shape)
        images_select_shape[1] = len(self.channels) # correct channel dim size
        self.images_select = torch.zeros(images_select_shape)
        for c in range(len(self.channels)):
            if self.channels[c] is None:
                self.images_select[:, c] = torch.zeros(self.images_select[:, c].shape)
            else:
                self.images_select[:, c] = self.images[:, self.channels[c]]
        self.images = self.images_select
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def __iter__(self):
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self.n_cells):
            raise StopIteration
        batch_size = self.n_cells[self.batch_index]
        start_index = sum(self.n_cells[:self.batch_index])
        end_index = start_index + batch_size
        batch = self.images[start_index:end_index]
        self.batch_index += 1
        return batch

    def iter_len(self):
        return len(self.n_cells)

    def get_channel_names(self):
        return self.channel_names

    def set_channel_colors(self, channel_colors):
        self.channel_colors = channel_colors

    def get_channel_colors(self):
        return self.channel_colors

    def __set_default_channel_colors(self):
        if len(self.channel_names) == 1:
            self.channel_colors = ["pure_gray"]
        elif len(self.channel_names) == 2:
            self.channel_colors = ["pure_blue", "pure_red"]
        elif len(self.channel_names) == 3:
            self.channel_colors = ["pure_blue", "pure_green", "pure_red"]
        else:
            raise ValueError(f"channel_colors not set and not suitable defaults for {len(self.channel_names)} channels.")

    def view(self, idx):
        image = self.__getitem__(idx).cpu().numpy()
        if self.channel_colors is None:
            self.__set_default_channel_colors()
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
                self.__set_default_channel_colors()
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


class RefCLSDM(LightningDataModule):
    """
    TODO: move the naming of the dataset types here and then read that from the data_pipeline so it's consistent if there 
    are changes to the data module/pipeline

    Data module for training a classifier on top of DINO embeddings of DAPI+TUBL reference channels
    Trying to match labels from a GMM or Ward cluster labeling algorithm of the FUCCI channel intensities
    """
    def __init__(self, data_dir, data_name, batch_size, num_workers, split, hpa, label, scope=None, concat_well_stats=False, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

        self.dataset = RefClsPseudo(self.data_dir, self.data_name, hpa, label, scope=scope, 
                                    concat_well_stats=concat_well_stats)
        generator = torch.Generator().manual_seed(seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.split, generator=generator)
        self.split_indices = {"train": self.train_dataset.indices, "val": self.val_dataset.indices, "test": self.test_dataset.indices}

    def shared_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)

    def train_dataloader(self):
        return self.shared_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self.shared_dataloader(self.val_dataset)
    
    def test_dataloader(self):
        return self.shared_dataloader(self.test_dataset)

class RefClsPseudo(Dataset):
    """
    Needs to handle whether or not to concatenate well intensity statistics onto the embeddings
    Needs to be able to read out each microscope in the data
    """
    def __init__(self, data_dir, data_name, hpa, label, scope=None, concat_well_stats=False):
        # TODO support for the dataset name
        cls_file = cls_embedding_name(data_dir, data_name, hpa=hpa)
        print(f"Loading {cls_file}")
        self.X = torch.load(cls_file)
        if concat_well_stats:
            intensity_file = intensity_data_name(data_dir, data_name)
            print(f"Loading {intensity_file}")
            self.well_stats = torch.load(intensity_file)
            print("X shape:", self.X.shape)
            self.X = torch.cat((self.X, self.well_stats), dim=1)

        self.Y = load_labels(label, data_dir, data_name, scope=scope)
        print(self.Y.min(), self.Y.max())

        self.X, self.Y = self.X.float(), self.Y.float()
        print("X shape:", self.X.shape)
        print("Y shape:", self.Y.shape)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def intensity_data_name(data_dir, data_name):
    return data_dir / f"intensity_distributions_{data_name}.pt"

def cls_embedding_name(data_dir, data_name, hpa=True):
    return data_dir / f"embeddings_{data_name}_{'dino_hpa' if hpa else 'dinov2'}.pt"

def angle_label_name(data_dir, data_name):
    return data_dir / f"{data_name}_sample_angles.pt"

def pseudotime_label_name(data_dir, data_name):
    return data_dir / f"{data_name}_sample_pseudotime.pt"

def phase_label_name(data_dir, data_name, scope):
    return data_dir / f"{data_name}_sample_phase{'_scope' if scope else ''}.pt"

def intensity_name(data_dir, data_name):
    return data_dir / f"{data_name}_intensity.npy"


def load_labels(label, data_dir, data_name, scope=None):
    label_types = ["angle", "pseudotime", "phase"]
    assert label in label_types, f"Invalid label type, must be in {label_types}"
    if label == "angle":
        label_file = angle_label_name(data_dir, data_name)
    elif label == "pseudotime":
        label_file = pseudotime_label_name(data_dir, data_name)
    elif label == "phase":
        if scope is None:
            raise ValueError("Must provide boolean scope flag for phase label")
        label_file = phase_label_name(data_dir, data_name, scope)
    print(f"Loading {label_file}")
    Y = torch.load(label_file)
    return Y

class RefImPseudo(Dataset):
    def __init__(self, data_dir, data_name, label, scope=None):
        self.X = CellImageDataset(data_dir / f"index_{data_name}.csv", channels=[0, 1])[:]
        self.Y = load_labels(label, data_dir, data_name, scope=scope)
        print("X shape:", self.X.shape)
        print("Y shape:", self.Y.shape)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

class RefImDM(LightningDataModule):
    def __init__(self, data_dir, data_name, batch_size, num_workers, split, label, scope=None, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

        self.dataset = RefImPseudo(self.data_dir, self.data_name, label, scope=scope)
        generator = torch.Generator().manual_seed(seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.split, generator=generator)
        self.split_indices = {"train": self.train_dataset.indices, "val": self.val_dataset.indices, "test": self.test_dataset.indices}

    def __shared_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)

    def train_dataloader(self):
        return self.__shared_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self.__shared_dataloader(self.val_dataset)
    
    def test_dataloader(self):
        return self.__shared_dataloader(self.test_dataset)
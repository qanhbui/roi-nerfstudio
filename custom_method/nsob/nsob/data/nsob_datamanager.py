"""
Nsob Datamanager.
"""
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from typing_extensions import TypeVar

from nerfstudio.cameras.cameras import CameraType

from nerfstudio.data.pixel_samplers import (
    EquirectangularPixelSampler,
    PatchPixelSampler,
    PixelSampler,
)

from nsob.data.nsob_pixel_samplers import NsobPixelSampler

from rich.progress import Console # information to print in console, color, bold, italic, size ...
CONSOLE = Console(width=120)

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from nerfstudio.data.datasets.base_dataset import InputDataset

@dataclass
class NsobDataManagerConfig(VanillaDataManagerConfig):
    """A Nsob data manager"""

    _target: Type = field(default_factory=lambda: NsobDataManager)
    """Target class to instantiate."""
    sample_only_in_mask: bool = True
    """Only samples inside the masks, if True. Else samples on over the whole image, including beyond the mask, even if there are masks in the json file."""
    

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

class NsobDataManager(VanillaDataManager):
    """Nsob stored data manager implementation.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: NsobDataManagerConfig
    all_dataset: TDataset

    def __init__(
        self,
        config: NsobDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.all_dataset = self.create_all_dataset() #

    def create_all_dataset(self) -> TDataset:
        """Sets up the data loaders for all dataset"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="all"),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, *args: Any, **kwargs: Any) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1:
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)

        # temporary code for sample pixels all over the image, including beyond the mask
        if not self.config.sample_only_in_mask:
            return NsobPixelSampler(*args, **kwargs)
        
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        
        return PixelSampler(*args, **kwargs)



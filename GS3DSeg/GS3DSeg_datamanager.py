"""
        Part of the code is taken from https://github.com/cvachha/instruct-gs2gs/blob/main/igs2gs/igs2gs_datamanager.py
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
from nerfstudio.cameras.cameras import Cameras
import torch
from copy import deepcopy
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
    FullImageDatamanager
)

@dataclass
class GS3DSegDataManagerConfig(FullImageDatamanagerConfig):

    _target: Type = field(default_factory=lambda: GS3DSegDataManager)


class GS3DSegDataManager(FullImageDatamanager):

    config: GS3DSegDataManagerConfig

    def __init__(
        self,
        config: GS3DSegDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

       
         # cache original training images for ip2p
        self.original_cached_train = deepcopy(self.cached_train)
        self.original_cached_eval = deepcopy(self.cached_eval)
        
        # Some logic to make sure we sample every camera in equal amounts
        self.editing_unseen_cameras = [i for i in range(len(self.train_dataset))]
        self.train_image_list = self.train_dataset.image_filenames
        self.eval_image_list  = self.eval_dataset.image_filenames

        
    def next_train_idx(self, idx: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        data = deepcopy(self.cached_train[idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[idx : idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = idx
        return camera, data
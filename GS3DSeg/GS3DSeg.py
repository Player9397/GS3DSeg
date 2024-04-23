
from dataclasses import dataclass, field
from typing import Type, Dict, List
import torch
from torch.nn import Parameter
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig 


@dataclass
class GS3DSegConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: GS3DSeg)


class GS3DSeg(SplatfactoModel):

    config: GS3DSegConfig

    def populate_modules(self):
        super().populate_modules()
        self.identity_vec = torch.nn.Parameter(torch.rand([self.num_points, 16]))
    
    def load_state_dict(self, dict, **kwargs):
        newp = dict["means"].shape[0]
        self.identity_vec = torch.nn.Parameter(torch.rand(newp, 16, device=self.device))
        super().load_state_dict(dict, **kwargs)
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "features_dc": [self.features_dc],
            "features_rest": [self.features_rest],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
            "identity_vec": [self.identity_vec]
        }


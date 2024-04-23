
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig 


@dataclass
class GS3DSegConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: GS3DSeg)


class GS3DSeg(SplatfactoModel):

    config: GS3DSegConfig


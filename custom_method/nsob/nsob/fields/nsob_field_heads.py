from nerfstudio.field_components.field_heads import FieldHead, FieldHeadNames
from typing import Callable, Optional, Union
from torch import Tensor, nn

class HitFieldHead(FieldHead):
    """Hit output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Sigmoid()) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, field_head_name=FieldHeadNames.DENSITY, activation=activation)
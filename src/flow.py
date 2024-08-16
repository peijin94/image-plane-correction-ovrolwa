from typing import Literal, Union

import jax.numpy as jnp
from interpax import interp2d
from jaxtyping import Array

from src.util import indices

Direction = Union[Literal["forwards"], Literal["backwards"]]


# inspired by https://github.com/CSRavasio/oflibnumpy
# For now, the flow direction is always "backwards" for performance reasons
# offsets should be of shape (x, y, 2)
class Flow:
    offsets: Array
    direction: Direction

    def __init__(self, offsets, direction):
        self.offsets = offsets
        self.direction = direction

    # warps an input single-channel image using bilinear interpolation
    def apply(self, image: Array):
        if self.direction == "backwards":
            H, W = image.shape

            # H*W query points, one for each pixel in warped image.
            # Offsets are given as as [dx, dy], while images are indexed [y, x], so we need to reverse the last axis of offsets.
            q = jnp.reshape(indices(H, W) + self.offsets[:, :, ::-1], shape=(H * W, 2))

            results = interp2d(
                q[:, 0],
                q[:, 1],
                jnp.arange(H),
                jnp.arange(W),
                image,
                method="linear",
                extrap=0,
            )

            return jnp.reshape(results, shape=(H, W))
        else:
            raise NotImplementedError()

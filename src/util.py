import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array


def indices(m, n):
    return jnp.indices((m, n)).transpose(1, 2, 0)


# https://github.com/google/jax/pull/15359
def _gaussian_kernel1d(sigma: float, order: int, radius: int) -> Array:
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = jnp.arange(order + 1)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = jnp.zeros(order + 1)
        q = q.at[0].set(1)
        D = jnp.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = jnp.diag(jnp.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def gaussian_filter1d(
    input: Array,
    sigma: float,
    axis=-1,
    order=0,
    truncate=4.0,
    *,
    radius: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    precision=None,
) -> Array:
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)

    if mode != "constant" or cval != 0.0:
        raise NotImplementedError(
            'Other modes than "constant" with 0. fill value are not' "supported yet."
        )

    if radius > 0.0:
        lw = radius
    if lw < 0:
        raise ValueError("Radius must be a nonnegative integer.")

    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

    # Be careful that modes in signal.convolve refer to the 'same' 'full' 'valid' modes
    # while in gaussian_filter1d refers to the way the padding is done 'constant' 'reflect' etc.
    # We should change the convolve backend for further features
    return jnp.apply_along_axis(
        jsp.signal.convolve,
        axis,
        input,
        weights,
        mode="same",
        method="auto",
        precision=precision,
    )


def gaussian_filter(
    input: Array,
    sigma: float,
    order: int = 0,
    truncate: float = 4.0,
    *,
    radius: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    precision=None,
) -> Array:
    input = jnp.asarray(input)

    for axis in range(input.ndim):
        input = gaussian_filter1d(
            input,
            sigma,
            axis=axis,
            order=order,
            truncate=truncate,
            radius=radius,
            mode=mode,
            precision=precision,
            cval=cval,
        )

    return input


def rescale_quantile(image, a, b):
    return jnp.clip(
        (image - jnp.quantile(image, a))
        / (jnp.quantile(image, b) - jnp.quantile(image, a)),
        0,
        1,
    )


def clip_quantile(image, a, b):
    return jnp.clip(image, jnp.quantile(image, a), jnp.quantile(image, b))


# https://stackoverflow.com/a/43346070
def gkern(l=5, sig=1.0):
    """\
    Creates gaussian kernel with side length `l` and a sigma of `sig`.
    Reaches a maximum of 1 at its center value
    """
    ax = jnp.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(sig))
    kernel = jnp.outer(gauss, gauss)
    return kernel / kernel.max()

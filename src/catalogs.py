import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

NVSS_CATALOG = "nvss_trim.dat"


# Returns the coordinates of sources in the reference catalog with at least the minimum flux (in mJy)
# The default value is 270 mJy since the NVSS catalog was observed at 1.4 GHz, the LWA testing images
# were taken at ~60 MHz with a lower-bound of ~2.7 Jy, and we assume a spectral index of -0.7.
def reference_sources_nvss(min_flux=270, with_flux=False):
    nvss = pd.read_csv(NVSS_CATALOG, sep="\s+")
    sorted_nvss = nvss.sort_values(by=["f"])

    # cut off refernce sources below a certain flux density
    sorted_nvss = sorted_nvss[sorted_nvss["f"] >= min_flux]

    # get coordinates of each reference source
    nvss_orig = sorted_nvss[["rah", "ram", "ras", "dd", "dm", "ds"]].iloc[:].to_numpy()

    # get flux of each reference source in Jy
    fluxes = sorted_nvss[["f"]].iloc[:].to_numpy().squeeze() / 1000

    # manually convert HMS:DMS into degrees
    nvss_ra = (
        15 * nvss_orig[:, 0]
        + (15 / 60) * nvss_orig[:, 1]
        + (15 / 3600) * nvss_orig[:, 2]
    )
    nvss_dec = (
        nvss_orig[:, 3] + (1 / 60) * nvss_orig[:, 4] + (1 / 3600) * nvss_orig[:, 5]
    )

    positions = np.stack((nvss_ra, nvss_dec), axis=-1)

    if with_flux:
        return SkyCoord(positions, unit=(u.degree, u.degree)), fluxes
    else:
        return SkyCoord(positions, unit=(u.degree, u.degree))

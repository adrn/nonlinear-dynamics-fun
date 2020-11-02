from math import gcd
from functools import reduce

import gala.coordinates as gc
import numpy as np
import superfreq as sf


def get_freqs(orbit):
    naff = sf.SuperFreq(orbit.t.value, keep_calm=True)

    if orbit.circulation().any():
        pp = gc.cartesian_to_poincare_polar(orbit.w().T)

        fs = [pp[:, 0] + 1j*pp[:, 3],
              pp[:, 1] + 1j*pp[:, 4],
              pp[:, 2] + 1j*pp[:, 5]]
    else:
        fs = [orbit.x.value + 1j*orbit.v_x.value,
              orbit.y.value + 1j*orbit.v_y.value,
              orbit.vz.value + 1j*orbit.v_z.value]

    res = naff.find_fundamental_frequencies(fs)

    return res.fund_freqs


def get_nvecs(max_int, ndim=3):
    # define meshgrid of integer vectors
    grids = [np.arange(-max_int+1, max_int+1)] * ndim
    nvecs = np.stack(list(map(np.ravel, np.meshgrid(*grids))),
                     axis=1)
    nvecs = np.delete(nvecs, np.where(np.all(nvecs == 0, axis=1))[0], axis=0)

    del_idx = []
    for i, nvec in enumerate(nvecs):
        if reduce(gcd, nvec) > 1:
            del_idx.append(i)
    nvecs = np.delete(nvecs, del_idx, axis=0)

    return nvecs


def closest_resonance(freqs, nvecs=None):
    # make sure the fundamental frequencies are a numpy array
    freqs = np.array(freqs)

    if isinstance(nvecs, int):
        nvecs = get_nvecs(nvecs)

    nvecs[:, 1] *= -1
    ndf = np.abs(nvecs.dot(freqs))  # / np.linalg.norm(freqs)
    min_ix = ndf.argmin()

    multi_check = np.isclose(ndf, ndf[min_ix])
    if multi_check.sum() > 1:
        min_ix2 = np.linalg.norm(nvecs[multi_check], axis=0).argmin()
        min_ix = np.where(multi_check)[0][min_ix2]

    return nvecs[min_ix], ndf[min_ix]

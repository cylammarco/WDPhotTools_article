import copy

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

teff_H_GF21 = copy.deepcopy(
    fits.open("gaiaedr3_wd_main.fits.gz")[1].data["teff_H"]
)

teff_h_convolved_total = np.load("compare_fusillo_edr3_teff_h_convolved.npy")
mbol_h_convolved_total = np.load("compare_fusillo_edr3_mbol_h_convolved.npy")
logg_h_convolved_total = np.load("compare_fusillo_edr3_logg_h_convolved.npy")
chi2_h_convolved_total = np.load("compare_fusillo_edr3_chi2_h_convolved.npy")
teff_h_interpolated_total = np.load(
    "compare_fusillo_edr3_teff_h_interpolated.npy"
)
mbol_h_interpolated_total = np.load(
    "compare_fusillo_edr3_mbol_h_interpolated.npy"
)
logg_h_interpolated_total = np.load(
    "compare_fusillo_edr3_logg_h_interpolated.npy"
)
chi2_h_interpolated_total = np.load(
    "compare_fusillo_edr3_chi2_h_interpolated.npy"
)

mask_convolved = (teff_H_GF21 > 3500) & (teff_h_convolved_total > 3500)
mask_interpolated = (teff_H_GF21 > 3500) & (teff_h_interpolated_total > 3500)

r_convolved = stats.spearmanr(
    teff_H_GF21[mask_convolved], teff_h_convolved_total[mask_convolved]
)
r_interpolated = stats.spearmanr(
    teff_H_GF21[mask_interpolated],
    teff_h_interpolated_total[mask_interpolated],
)


def density_estimation(m1, m2):
    X, Y = np.mgrid[1500:50000:500j, 1500:50000:500j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


X, Y, Z = density_estimation(
    teff_H_GF21[mask_convolved], teff_h_convolved_total[mask_convolved]
)


plt.figure(1, figsize=(8, 8))
plt.clf()
plt.scatter(
    teff_H_GF21[mask_interpolated],
    teff_h_interpolated_total[mask_interpolated],
    s=0.1,
    color="red",
    label="Interpolated",
)
plt.scatter(
    teff_H_GF21[mask_convolved],
    teff_h_convolved_total[mask_convolved],
    s=0.1,
    label="Convolved",
)
# Add contour lines
plt.contour(X, Y, Z, levels=20, lw=1)
plt.plot([0, 1e6], [0, 1e6], c="k", lw=1)
plt.xlim(3000, 100000)
plt.ylim(3000, 100000)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Temperature / K (GF21)")
plt.ylabel("Temperature / K (This work)")
plt.title("DA")
plt.grid(which="both")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig_03_compare_fusillo_edr3.png")
plt.savefig("fig_03_compare_fusillo_edr3.pdf")

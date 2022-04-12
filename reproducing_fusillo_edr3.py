from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools.fitter import WDfitter

# data = fits.open('GaiaEDR3_WD_SDSSspec.FITS')
data = fits.open("gaiaedr3_wd_main.fits.gz")

G3 = np.array(data[1].data["phot_g_mean_mag_corrected"])
G3_error = np.array(data[1].data["phot_g_mean_mag_error_corrected"])
G3_BP = np.array(data[1].data["phot_bp_mean_mag"])
G3_BP_error = np.array(data[1].data["phot_bp_mean_mag_error"])
G3_RP = np.array(data[1].data["phot_rp_mean_mag"])
G3_RP_error = np.array(data[1].data["phot_rp_mean_mag_error"])

Av = np.array(data[1].data["meanAV"])
AG = Av * 0.835
AGBP = Av * 1.3894
AGRP = Av * 0.6496

parallax = np.array(
    (data[1].data["parallax"] + data[1].data["ZP_correction"]) / 1000.0
)
parallax_error = np.array(data[1].data["parallax_error"]) / 1000.0

distance = 1.0 / parallax
distance_error = distance**2.0 * parallax_error

ftr = WDfitter()
output = []
length = len(G3)
length = 10000
teff_h = np.zeros(length)
logg_h = np.zeros(length)
teff_he = np.zeros(length)
logg_he = np.zeros(length)


for i in range(length):
    print("{} of {}".format(i + 1, length))
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP"],
        mags=[G3[i] - AG[i], G3_BP[i] - AGBP[i], G3_RP[i] - AGRP[i]],
        mag_errors=[G3_error[i], G3_BP_error[i], G3_RP_error[i]],
        independent=["Mbol", "logg"],
        method="minimize",
        distance=distance[i],
        distance_err=distance_error[i],
        initial_guess=[10.0, 8.0],
        reuse_interpolator=True,
    )
    output.append(ftr.results)
    teff_h[i] = ftr.best_fit_params["H"]["Teff"]
    logg_h[i] = ftr.best_fit_params["H"]["logg"]
    teff_he[i] = ftr.best_fit_params["He"]["Teff"]
    logg_he[i] = ftr.best_fit_params["He"]["logg"]

plt.figure(1, figsize=(8, 8))
plt.clf()
plt.scatter(data[1].data["teff_H"][:length], teff_h, s=5)
plt.xlabel("Temperature / K (Fusillo EDR3)")
plt.ylabel("Temperature / K (This work)")
plt.title("DA")
plt.xlim(4000, 50000)
plt.ylim(4000, 50000)
plt.grid()
plt.tight_layout()

plt.figure(2, figsize=(8, 8))
plt.clf()
plt.scatter(data[1].data["teff_He"][:length], teff_he, s=5)
plt.xlabel("Temperature / K (Fusillo EDR3)")
plt.ylabel("Temperature / K (This work)")
plt.title("DB")
plt.xlim(4000, 50000)
plt.ylim(4000, 50000)
plt.grid()
plt.tight_layout()


plt.figure(3, figsize=(8, 8))
plt.clf()
plt.scatter(data[1].data["logg_H"][:length], logg_h, s=5)
plt.xlabel("log(g) (Fusillo EDR3)")
plt.ylabel("log(g) (This work)")
plt.title("DA")
plt.xlim(7.0, 9.0)
plt.ylim(7.0, 9.0)
plt.grid()
plt.tight_layout()

plt.figure(4, figsize=(8, 8))
plt.clf()
plt.scatter(data[1].data["logg_He"][:length], logg_he, s=5)
plt.xlabel("log(g) (Fusillo EDR3)")
plt.ylabel("log(g) (This work)")
plt.title("DB")
plt.xlim(7.0, 9.0)
plt.ylim(7.0, 9.0)
plt.grid()
plt.tight_layout()


plt.show()

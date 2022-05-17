from astropy.io import fits
import copy
import gc
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import sys

from WDPhotTools.fitter import WDfitter
from WDPhotTools.reddening import reddening_vector_interpolated


n_parts = int(sys.argv[1])

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()

reddening = reddening_vector_interpolated(kind="cubic")
wave_GBR = np.array(
    (
        5822.39,
        5035.75,
        7619.96,
    )
)


if my_rank == 0:

    # data = fits.open('GaiaEDR3_WD_SDSSspec.FITS')
    data = fits.open("gaiaedr3_wd_main.fits.gz")

    G3 = copy.deepcopy(data[1].data["phot_g_mean_mag_corrected"])
    G3_error = copy.deepcopy(data[1].data["phot_g_mean_mag_error_corrected"])
    G3_BP = copy.deepcopy(data[1].data["phot_bp_mean_mag"])
    G3_BP_error = copy.deepcopy(data[1].data["phot_bp_mean_mag_error"])
    G3_RP = copy.deepcopy(data[1].data["phot_rp_mean_mag"])
    G3_RP_error = copy.deepcopy(data[1].data["phot_rp_mean_mag_error"])

    Av = copy.deepcopy(data[1].data["meanAV"])

    teff_H_GF21 = copy.deepcopy(data[1].data["teff_H"])

    distance = 1.0 / copy.deepcopy(
        (data[1].data["parallax"] + data[1].data["ZP_CORRECTION"]) / 1000.0
    )
    distance_error = distance**2.0 * copy.deepcopy(
        data[1].data["parallax_error"] / 1000.0
    )

    del data
    gc.collect()

else:

    G3 = None
    G3_error = None
    G3_BP = None
    G3_BP_error = None
    G3_RP = None
    G3_RP_error = None
    Av = None
    teff_H_GF21 = None
    distance = None
    distance_error = None

G3 = comm.bcast(G3, root=0)
G3_error = comm.bcast(G3_error, root=0)
G3_BP = comm.bcast(G3_BP, root=0)
G3_BP_error = comm.bcast(G3_BP_error, root=0)
G3_RP = comm.bcast(G3_RP, root=0)
G3_RP_error = comm.bcast(G3_RP_error, root=0)
Av = comm.bcast(Av, root=0)
teff_H_GF21 = comm.bcast(teff_H_GF21, root=0)
distance = comm.bcast(distance, root=0)
distance_error = comm.bcast(distance_error, root=0)

n_data = len(G3)

ith_by_rank = range(my_rank, n_data, n_parts)


ftr = WDfitter()
output = []

teff_h = np.zeros(n_data)
mbol_h = np.zeros(n_data)
logg_h = np.zeros(n_data)
chi2_h = np.zeros(n_data)
teff_he = np.zeros(n_data)
mbol_he = np.zeros(n_data)
logg_he = np.zeros(n_data)
chi2_he = np.zeros(n_data)

for i in ith_by_rank:

    sys.stdout.write("{} of {}{}".format(i + 1, n_data, os.linesep))
    ebv = Av[i] / reddening(wave_GBR, 3.1)
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP"],
        mags=[G3[i], G3_BP[i], G3_RP[i]],
        mag_errors=[G3_error[i], G3_BP_error[i], G3_RP_error[i]],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        reuse_interpolator=False,
        distance=distance[i],
        distance_err=distance_error[i],
        extinction_convolved=True,
        Rv=3.1,
        ebv=ebv,
        initial_guess=[10000.0, 8.0],
    )
    sys.stdout.write("GF21: {} K {}".format(teff_H_GF21[i], os.linesep))
    sys.stdout.write(
        "This work: {} K {}".format(
            ftr.best_fit_params["H"]["Teff"], os.linesep
        )
    )
    teff_h[i] = ftr.best_fit_params["H"]["Teff"]
    mbol_h[i] = ftr.best_fit_params["H"]["Mbol"]
    logg_h[i] = ftr.best_fit_params["H"]["logg"]
    chi2_h[i] = ftr.best_fit_params["H"]["chi2"]
    teff_he[i] = ftr.best_fit_params["He"]["Teff"]
    mbol_he[i] = ftr.best_fit_params["He"]["Mbol"]
    logg_he[i] = ftr.best_fit_params["He"]["logg"]
    chi2_he[i] = ftr.best_fit_params["He"]["chi2"]

if my_rank == 0:

    teff_h_total = np.zeros_like(teff_h)
    mbol_h_total = np.zeros_like(mbol_h)
    logg_h_total = np.zeros_like(logg_h)
    chi2_h_total = np.zeros_like(chi2_h)
    teff_he_total = np.zeros_like(teff_he)
    mbol_he_total = np.zeros_like(mbol_he)
    logg_he_total = np.zeros_like(logg_he)
    chi2_he_total = np.zeros_like(chi2_he)

else:

    teff_h_total = None
    mbol_h_total = None
    logg_h_total = None
    chi2_h_total = None
    teff_he_total = None
    mbol_he_total = None
    logg_he_total = None
    chi2_he_total = None

comm.Reduce(
    [teff_h, MPI.DOUBLE], [teff_h_total, MPI.DOUBLE], op=MPI.SUM, root=0
)
comm.Reduce(
    [mbol_h, MPI.DOUBLE], [mbol_h_total, MPI.DOUBLE], op=MPI.SUM, root=0
)
comm.Reduce(
    [logg_h, MPI.DOUBLE], [logg_h_total, MPI.DOUBLE], op=MPI.SUM, root=0
)
comm.Reduce(
    [chi2_h, MPI.DOUBLE], [chi2_h_total, MPI.DOUBLE], op=MPI.SUM, root=0
)

comm.Reduce(
    [teff_he, MPI.DOUBLE], [teff_he_total, MPI.DOUBLE], op=MPI.SUM, root=0
)
comm.Reduce(
    [mbol_he, MPI.DOUBLE], [mbol_he_total, MPI.DOUBLE], op=MPI.SUM, root=0
)
comm.Reduce(
    [logg_he, MPI.DOUBLE], [logg_he_total, MPI.DOUBLE], op=MPI.SUM, root=0
)
comm.Reduce(
    [chi2_he, MPI.DOUBLE], [chi2_he_total, MPI.DOUBLE], op=MPI.SUM, root=0
)


if my_rank == 0:

    np.save("compare_fusillo_edr3_teff_h_convolved", teff_h_total)
    np.save("compare_fusillo_edr3_mbol_h_convolved", mbol_h_total)
    np.save("compare_fusillo_edr3_logg_h_convolved", logg_h_total)
    np.save("compare_fusillo_edr3_chi2_h_convolved", chi2_h_total)
    np.save("compare_fusillo_edr3_teff_he_convolved", teff_he_total)
    np.save("compare_fusillo_edr3_mbol_he_convolved", mbol_he_total)
    np.save("compare_fusillo_edr3_logg_he_convolved", logg_he_total)
    np.save("compare_fusillo_edr3_chi2_he_convolved", chi2_he_total)

import copy
from matplotlib import pyplot as plt
import numpy as np
import os

from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0.0, 20.0, 0.1)


if os.path.exists("wdlf_5Gyr_compare_da_cooling_models.npy"):

    wdlf_5Gyr = np.load("wdlf_5Gyr_compare_da_cooling_models.npy")
    wdlf_7Gyr = np.load("wdlf_7Gyr_compare_da_cooling_models.npy")
    wdlf_9Gyr = np.load("wdlf_9Gyr_compare_da_cooling_models.npy")
    wdlf_11Gyr = np.load("wdlf_11Gyr_compare_da_cooling_models.npy")
    wdlf_13Gyr = np.load("wdlf_13Gyr_compare_da_cooling_models.npy")
    wdlf_15Gyr = np.load("wdlf_15Gyr_compare_da_cooling_models.npy")

else:

    # default montreal (A)
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_15Gyr = copy.deepcopy(wdlf.number_density)

    # lpcode_da_22, lpcode_da_22, lpcode_da_22 (B)
    wdlf.set_low_mass_cooling_model("lpcode_da_22")
    wdlf.set_intermediate_mass_cooling_model("lpcode_da_22")
    wdlf.set_high_mass_cooling_model("lpcode_da_22")
    wdlf.compute_cooling_age_interpolator()
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_lpcode_co_co_one_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_lpcode_co_co_one_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_lpcode_co_co_one_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_lpcode_co_co_one_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_lpcode_co_co_one_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_lpcode_co_co_one_15Gyr = copy.deepcopy(wdlf.number_density)

    # montreal_co_da_20, basti_co_da_10, basti_co_da_10 (C)
    wdlf.set_low_mass_cooling_model("montreal_co_da_20")
    wdlf.set_intermediate_mass_cooling_model("basti_co_da_10")
    wdlf.set_high_mass_cooling_model("basti_co_da_10")
    wdlf.compute_cooling_age_interpolator()
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_15Gyr = copy.deepcopy(wdlf.number_density)

    # montreal_co_da_20, basti_co_da_10_nps, basti_co_da_10_nps (D)
    wdlf.set_low_mass_cooling_model("montreal_co_da_20")
    wdlf.set_intermediate_mass_cooling_model("basti_co_da_10_nps")
    wdlf.set_high_mass_cooling_model("basti_co_da_10_nps")
    wdlf.compute_cooling_age_interpolator()
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_nps_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_nps_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_nps_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_nps_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_nps_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(
        Mag=Mag,
        n_points=500,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    wdlf_montreal_basti_nps_15Gyr = copy.deepcopy(wdlf.number_density)

    wdlf_5Gyr = np.vstack(
        (
            wdlf_montreal_5Gyr,
            wdlf_lpcode_co_co_one_5Gyr,
            wdlf_montreal_basti_5Gyr,
            wdlf_montreal_basti_nps_5Gyr,
        )
    )
    wdlf_7Gyr = np.vstack(
        (
            wdlf_montreal_7Gyr,
            wdlf_lpcode_co_co_one_7Gyr,
            wdlf_montreal_basti_7Gyr,
            wdlf_montreal_basti_nps_7Gyr,
        )
    )
    wdlf_9Gyr = np.vstack(
        (
            wdlf_montreal_9Gyr,
            wdlf_lpcode_co_co_one_9Gyr,
            wdlf_montreal_basti_9Gyr,
            wdlf_montreal_basti_nps_9Gyr,
        )
    )
    wdlf_11Gyr = np.vstack(
        (
            wdlf_montreal_11Gyr,
            wdlf_lpcode_co_co_one_11Gyr,
            wdlf_montreal_basti_11Gyr,
            wdlf_montreal_basti_nps_11Gyr,
        )
    )
    wdlf_13Gyr = np.vstack(
        (
            wdlf_montreal_13Gyr,
            wdlf_lpcode_co_co_one_13Gyr,
            wdlf_montreal_basti_13Gyr,
            wdlf_montreal_basti_nps_13Gyr,
        )
    )
    wdlf_15Gyr = np.vstack(
        (
            wdlf_montreal_15Gyr,
            wdlf_lpcode_co_co_one_15Gyr,
            wdlf_montreal_basti_15Gyr,
            wdlf_montreal_basti_nps_15Gyr,
        )
    )

    np.save("wdlf_5Gyr_compare_da_cooling_models", wdlf_5Gyr)
    np.save("wdlf_7Gyr_compare_da_cooling_models", wdlf_7Gyr)
    np.save("wdlf_9Gyr_compare_da_cooling_models", wdlf_9Gyr)
    np.save("wdlf_11Gyr_compare_da_cooling_models", wdlf_11Gyr)
    np.save("wdlf_13Gyr_compare_da_cooling_models", wdlf_13Gyr)
    np.save("wdlf_15Gyr_compare_da_cooling_models", wdlf_15Gyr)


# normalise the WDLFs relative to the density at 10 mag
wdlf_5Gyr = [
    wdlf_5Gyr[i] / wdlf_5Gyr[i][Mag == 10.0] for i in range(len(wdlf_5Gyr))
]
wdlf_7Gyr = [
    wdlf_7Gyr[i] / wdlf_7Gyr[i][Mag == 10.0] for i in range(len(wdlf_7Gyr))
]
wdlf_9Gyr = [
    wdlf_9Gyr[i] / wdlf_9Gyr[i][Mag == 10.0] for i in range(len(wdlf_9Gyr))
]
wdlf_11Gyr = [
    wdlf_11Gyr[i] / wdlf_11Gyr[i][Mag == 10.0] for i in range(len(wdlf_11Gyr))
]
wdlf_13Gyr = [
    wdlf_13Gyr[i] / wdlf_13Gyr[i][Mag == 10.0] for i in range(len(wdlf_13Gyr))
]
wdlf_15Gyr = [
    wdlf_15Gyr[i] / wdlf_15Gyr[i][Mag == 10.0] for i in range(len(wdlf_15Gyr))
]

ifmr_list = ["A", "B", "C", "D", "E", "F"]

fig1, (
    (ax1, ax9),
    (ax2, ax10),
    (ax3, ax11),
    (ax4, ax12),
    (ax5, ax13),
    (ax6, ax14),
    (ax7, ax15),
    (ax8, ax16),
) = plt.subplots(
    8,
    2,
    sharex=True,
    figsize=(12, 15),
    gridspec_kw={"height_ratios": [8, 4, 1, 8, 4, 1, 8, 4]},
)

# Burst SFR 5 Gyr
for w, l in zip(wdlf_5Gyr, ifmr_list):
    ax1.plot(Mag, np.log10(w), label=l)
    ax2.plot(Mag, np.log10(w / wdlf_5Gyr[0]))

# Burst SFR 7 Gyr
for w in wdlf_7Gyr:
    ax4.plot(Mag, np.log10(w))
    ax5.plot(Mag, np.log10(w / wdlf_7Gyr[0]))

# Burst SFR 9 Gyr
for w in wdlf_9Gyr:
    ax7.plot(Mag, np.log10(w))
    ax8.plot(Mag, np.log10(w / wdlf_9Gyr[0]))

# Burst SFR 11 Gyr
for w in wdlf_11Gyr:
    ax9.plot(Mag, np.log10(w))
    ax10.plot(Mag, np.log10(w / wdlf_11Gyr[0]))

# Burst SFR 13 Gyr
for w in wdlf_13Gyr:
    ax12.plot(Mag, np.log10(w))
    ax13.plot(Mag, np.log10(w / wdlf_13Gyr[0]))

# Burst SFR 13 Gyr
for w in wdlf_15Gyr:
    ax15.plot(Mag, np.log10(w))
    ax16.plot(Mag, np.log10(w / wdlf_15Gyr[0]))

ax1.set_xticks(range(2, 21), minor=True)
ax1.set_xticks(range(5, 21, 5), minor=False)

ax1.grid(which="minor", color="lightgrey", linestyle="--")
ax1.grid(which="major", color="grey", linestyle="-")
ax2.grid(which="minor", color="lightgrey", linestyle="--")
ax2.grid(which="major", color="grey", linestyle="-")

ax4.grid(which="minor", color="lightgrey", linestyle="--")
ax4.grid(which="major", color="grey", linestyle="-")
ax5.grid(which="minor", color="lightgrey", linestyle="--")
ax5.grid(which="major", color="grey", linestyle="-")

ax7.grid(which="minor", color="lightgrey", linestyle="--")
ax7.grid(which="major", color="grey", linestyle="-")
ax8.grid(which="minor", color="lightgrey", linestyle="--")
ax8.grid(which="major", color="grey", linestyle="-")

ax9.grid(which="minor", color="lightgrey", linestyle="--")
ax9.grid(which="major", color="grey", linestyle="-")
ax10.grid(which="minor", color="lightgrey", linestyle="--")
ax10.grid(which="major", color="grey", linestyle="-")

ax12.grid(which="minor", color="lightgrey", linestyle="--")
ax12.grid(which="major", color="grey", linestyle="-")
ax13.grid(which="minor", color="lightgrey", linestyle="--")
ax13.grid(which="major", color="grey", linestyle="-")

ax15.grid(which="minor", color="lightgrey", linestyle="--")
ax15.grid(which="major", color="grey", linestyle="-")
ax16.grid(which="minor", color="lightgrey", linestyle="--")
ax16.grid(which="major", color="grey", linestyle="-")

ax1.legend()

ax3.axis("off")
ax6.axis("off")
ax11.axis("off")
ax14.axis("off")

ax1.set_xlim(0, 20)

ax1.set_ylim(-3, 3.5)
ax4.set_ylim(-3, 3.5)
ax7.set_ylim(-3, 3.5)
ax9.set_ylim(-3, 3.5)
ax12.set_ylim(-3, 3.5)
ax15.set_ylim(-3, 3.5)

ax2.set_ylim(-1, 1)
ax5.set_ylim(-1, 1)
ax8.set_ylim(-1, 1)
ax10.set_ylim(-1, 1)
ax13.set_ylim(-1, 1)
ax16.set_ylim(-1, 1)

ax9.set_yticklabels([""])
ax10.set_yticklabels([""])
ax11.set_yticklabels([""])
ax12.set_yticklabels([""])
ax13.set_yticklabels([""])
ax14.set_yticklabels([""])
ax15.set_yticklabels([""])
ax16.set_yticklabels([""])

ax1.set_title("5 Gyr")
ax4.set_title("7 Gyr")
ax7.set_title("9 Gyr")
ax9.set_title("11 Gyr")
ax12.set_title("13 Gyr")
ax15.set_title("15 Gyr")

ax1.set_ylabel("log(arbitrary number density)")
ax2.set_ylabel(r"log(n/n$_{Combination A}$)")
ax4.set_ylabel("log(arbitrary number density)")
ax5.set_ylabel(r"log(n/n$_{Combination A}$)")
ax7.set_ylabel("log(arbitrary number density)")
ax8.set_ylabel(r"log(n/n$_{Combination A}$)")

fig1.supxlabel(r"M$_{\mathrm{bol}}$ / mag")

plt.subplots_adjust(
    left=0.1, right=0.98, top=0.96, bottom=0.075, hspace=0.00, wspace=0.01
)
plt.savefig("fig_11_wdlf_compare_da_cooling_models.png")
plt.savefig("fig_11_wdlf_compare_da_cooling_models.pdf")

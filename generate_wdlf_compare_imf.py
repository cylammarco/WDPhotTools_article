import copy
import os

from matplotlib import pyplot as plt
import numpy as np
from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0.0, 20.0, 0.1)


if os.path.exists('wdlf_5Gyr_compare_imf.npy'):

    wdlf_11Gyr = np.load('wdlf_11Gyr_compare_imf.npy')
    wdlf_13Gyr = np.load('wdlf_13Gyr_compare_imf.npy')
    wdlf_15Gyr = np.load('wdlf_15Gyr_compare_imf.npy')

else:

    # default
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C03_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C03_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C03_15Gyr = copy.deepcopy(wdlf.number_density)

    wdlf.set_imf_model("C03b")
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C03b_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C03b_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C03b_15Gyr = copy.deepcopy(wdlf.number_density)

    wdlf.set_imf_model("K01")
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K01_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K01_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K01_15Gyr = copy.deepcopy(wdlf.number_density)

    wdlf_11Gyr = np.vstack(
        (
            wdlf_C03_11Gyr,
            wdlf_C03b_11Gyr,
            wdlf_K01_11Gyr,
        )
    )
    wdlf_13Gyr = np.vstack(
        (
            wdlf_C03_13Gyr,
            wdlf_C03b_13Gyr,
            wdlf_K01_13Gyr,
        )
    )
    wdlf_15Gyr = np.vstack(
        (
            wdlf_C03_15Gyr,
            wdlf_C03b_15Gyr,
            wdlf_K01_15Gyr,
        )
    )

    np.save('wdlf_11Gyr_compare_imf', wdlf_11Gyr)
    np.save('wdlf_13Gyr_compare_imf', wdlf_13Gyr)
    np.save('wdlf_15Gyr_compare_imf', wdlf_15Gyr)


# normalise the WDLFs relative to the density at 10 mag
wdlf_11Gyr = [wdlf_11Gyr[i]/wdlf_11Gyr[i][Mag==10.0] for i in range(len(wdlf_11Gyr))]
wdlf_13Gyr = [wdlf_13Gyr[i]/wdlf_13Gyr[i][Mag==10.0] for i in range(len(wdlf_13Gyr))]
wdlf_15Gyr = [wdlf_15Gyr[i]/wdlf_15Gyr[i][Mag==10.0] for i in range(len(wdlf_15Gyr))]

fig1, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True, sharey=True, figsize=(10, 15)
)

# Burst SFR 11 Gyr
ax1.plot(
    Mag, np.log10(wdlf_11Gyr[0])-np.log10(wdlf_11Gyr[0]), label="C03"
)
ax1.plot(
    Mag, np.log10(wdlf_11Gyr[1])-np.log10(wdlf_11Gyr[0]), label="C03b"
)
ax1.plot(
    Mag, np.log10(wdlf_11Gyr[2])-np.log10(wdlf_11Gyr[0]), label="K01"
)

# Burst SFR 13 Gyr
ax2.plot(
    Mag, np.log10(wdlf_13Gyr[0])-np.log10(wdlf_13Gyr[0])
)
ax2.plot(
    Mag, np.log10(wdlf_13Gyr[1])-np.log10(wdlf_13Gyr[0])
)
ax2.plot(
    Mag, np.log10(wdlf_13Gyr[2])-np.log10(wdlf_13Gyr[0])
)

# Burst SFR 13 Gyr
ax3.plot(
    Mag, np.log10(wdlf_15Gyr[0])-np.log10(wdlf_15Gyr[0]))
ax3.plot(
    Mag, np.log10(wdlf_15Gyr[1])-np.log10(wdlf_15Gyr[0]))
ax3.plot(
    Mag, np.log10(wdlf_15Gyr[2])-np.log10(wdlf_15Gyr[0]))

ax1.legend()
ax1.grid()
ax1.set_xlim(0, 20)
ax1.set_ylim(-0.01, 0.01)
ax1.set_title("11 Gyr")

ax2.grid()
ax2.set_xlim(0, 20)
ax2.set_ylim(-0.01, 0.01)
ax2.set_ylabel(r"log(n / n$_{C03}$)")
ax2.set_title("13 Gyr")

ax3.grid()
ax3.set_xlim(0, 20)
ax3.set_ylim(-0.01, 0.01)
ax3.set_xlabel(r"M$_{\mathrm{bol}}$ / mag")
ax3.set_title("15 Gyr")

plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.075)
plt.savefig('wdlf_compare_imf.png')
plt.savefig('wdlf_compare_imf.pdf')

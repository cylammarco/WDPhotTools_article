from matplotlib import pyplot as plt
import numpy as np
import copy
from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(2.0, 20.0, 0.1)

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


fig1, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True, sharey=True, figsize=(10, 15)
)

# Burst SFR 11 Gyr
ax1.plot(
    Mag, np.log10(wdlf_C03_11Gyr)-np.log10(wdlf_C03_11Gyr), label="C03")
)
ax1.plot(
    Mag, np.log10(wdlf_C03b_11Gyr)-np.log10(wdlf_C03_11Gyr), label="C03b")
)
ax1.plot(
    Mag, np.log10(wdlf_K01_11Gyr)-np.log10(wdlf_C03_11Gyr), label="K01")
)

# Burst SFR 13 Gyr
ax2.plot(
    Mag, np.log10(wdlf_C03_13Gyr)-np.log10(wdlf_C03_13Gyr), label="C03")
)
ax2.plot(
    Mag, np.log10(wdlf_C03b_13Gyr)-np.log10(wdlf_C03_13Gyr), label="C03b")
)
ax2.plot(
    Mag, np.log10(wdlf_K01_13Gyr)-np.log10(wdlf_C03_13Gyr), label="K01")
)

# Burst SFR 13 Gyr
ax3.plot(
    Mag, np.log10(wdlf_C03_15Gyr)-np.log10(wdlf_C03_15Gyr))
ax3.plot(
    Mag, np.log10(wdlf_C03b_15Gyr)-np.log10(wdlf_C03_15Gyr))
ax3.plot(
    Mag, np.log10(wdlf_K01_15Gyr)-np.log10(wdlf_C03_15Gyr))

ax1.legend()
ax1.grid()
ax1.set_xlim(2, 20)
ax1.set_ylim(-0.01, 0.01)
ax1.set_title("11 Gyr")

ax2.grid()
ax2.set_xlim(2, 20)
ax2.set_ylim(-0.01, 0.01)
ax2.set_ylabel(r"log(n / n$_{C03}$)")
ax2.set_title("13 Gyr")

ax3.grid()
ax3.set_xlim(2, 20)
ax3.set_ylim(-0.01, 0.01)
ax3.set_xlabel(r"M$_{\mathrm{bol}}$ / mag")
ax3.set_title("15 Gyr")

plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.075)
plt.savefig('wdlf_compare_imf.png')
plt.savefig('wdlf_compare_imf.pdf')

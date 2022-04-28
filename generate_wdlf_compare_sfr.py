import os

from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools import theoretical_lf

try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except:
    HERE = os.path.dirname(os.path.realpath(__name__))

wdlf = theoretical_lf.WDLF()
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0, 20.0, 0.1)
age_list = 1e9 * np.arange(2, 15, 2)

constant_density = []
burst_density = []
decay_density = []

if os.path.exists('wdlf_compare_constant_density.npy'):

    constant_density = np.load('wdlf_compare_constant_density.npy')
    burst_density = np.load('wdlf_compare_burst_density.npy')
    decay_density = np.load('wdlf_compare_decay_density.npy')

else:

    for i, age in enumerate(age_list):

        # Constant SFR
        wdlf.set_sfr_model(mode="constant", age=age)
        constant_density.append(wdlf.compute_density(Mag=Mag)[1])

        # Burst SFR
        wdlf.set_sfr_model(mode="burst", age=age, duration=1e8)
        burst_density.append(wdlf.compute_density(Mag=Mag, passband="G3"))

        # Exponential decay SFR
        wdlf.set_sfr_model(mode="decay", age=age)
        decay_density.append(wdlf.compute_density(Mag=Mag, passband="G3"))


    np.save('wdlf_compare_constant_density', constant_density)
    np.save('wdlf_compare_burst_density', burst_density)
    np.save('wdlf_compare_decay_density', decay_density)


fig1, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True, sharey=True, figsize=(10, 15)
)

for i, age in enumerate(age_list):
    ax1.plot(
        Mag, np.log10(constant_density[i]), label="{0:.2f} Gyr".format(age / 1e9)
    )
    ax2.plot(
        Mag, np.log10(burst_density[i])
    )
    ax3.plot(
        Mag, np.log10(decay_density[i])
    )

ax1.legend()
ax1.grid()
ax1.set_xlim(7.5, 20)
ax1.set_ylim(-5, 0)
ax1.set_title("Constant")

ax2.grid()
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_title("100 Myr Burst")

ax3.grid()
ax3.set_xlabel(r"G$_{DR3}$ / mag")
ax3.set_title(r"Exponential Decay ($\tau=3$)")

plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.075)

plt.savefig(
    os.path.join(
        HERE,
        ".",
        "wdlf_compare_sfr.png",
    )
)

plt.savefig(
    os.path.join(
        HERE,
        ".",
        "wdlf_compare_sfr.pdf",
    )
)

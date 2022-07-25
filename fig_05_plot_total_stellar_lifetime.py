from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools import theoretical_lf


wdlf = theoretical_lf.WDLF()

ms_mass = np.arange(0.5, 10.0, 0.01)
ms_list = ["PARSECz00001",
           "PARSECz0001",
           "PARSECz001",
           "PARSECz0017",
           "PARSECz006",
           "GENEVAz014",
           "MISTFe000"]

plt.figure(1, figsize=(8, 6))
plt.clf()

for i, ms in enumerate(ms_list):
    wdlf.set_ms_model(ms)
    age = wdlf._ms_age(ms_mass)
    plt.plot(ms_mass, age, label=ms_list[i])

plt.legend()
plt.grid()

plt.xlim(min(ms_mass), max(ms_mass))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Initial (ZAMS Mass) / M$_\odot$")
plt.ylabel(r"Total Stellar Evolution Time / yr")

plt.tight_layout()
plt.savefig("fig_05_total_stellar_lifetime.png")
plt.savefig("fig_05_total_stellar_lifetime.pdf")

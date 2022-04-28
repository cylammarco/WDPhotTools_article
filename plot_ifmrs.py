from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools import theoretical_lf


wdlf = theoretical_lf.WDLF()

ms_mass = np.arange(0.5, 8.0, 0.1)
ifmr_list = ["C08", "C08b", "S09", "S09b", "W09", "K09", "K09b", "C18", "EB18"]
wd_mass = np.zeros((len(ifmr_list), len(ms_mass)))

for i, ifmr in enumerate(ifmr_list):
    wdlf.set_ifmr_model(ifmr)
    wd_mass[i] = wdlf._ifmr(M=ms_mass)


plt.figure(1, figsize=(8, 6))
plt.clf()
for i, ifmr in enumerate(ifmr_list):
    plt.plot(ms_mass, wd_mass[i], label=ifmr_list[i])

plt.legend()
plt.grid()

plt.xlim(0.5, 8)
plt.ylim(0.4, 1.4)

plt.xlabel(r"Initial (ZAMS Mass) / M$_\odot$")
plt.ylabel(r"Final (WD Mass) / M$_\odot$")

plt.tight_layout()
plt.savefig("ifmrs.png")
plt.savefig("ifmrs.pdf")

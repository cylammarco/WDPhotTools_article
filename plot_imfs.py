from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools import theoretical_lf


wdlf = theoretical_lf.WDLF()

ms_mass = np.arange(0.1, 10.0, 0.01)
imf_list = ["C03", "C03b", "K01"]
mf = np.zeros((len(imf_list), len(ms_mass)))

plt.figure(1, figsize=(8, 6))
plt.clf()

for i, imf in enumerate(imf_list):
    wdlf.set_imf_model(imf)
    mf[i] = wdlf._imf(M=ms_mass)
    plt.plot(ms_mass, mf[i], label=imf_list[i])

plt.legend()
plt.grid()

plt.xlim(0.1, 10)
plt.ylim(0.008, 20)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Initial (ZAMS Mass) / M$_\odot$")
plt.xlabel(r"Initial Mass Function")

plt.tight_layout()
plt.savefig("imfs.png")
plt.savefig("imfs.pdf")

import copy
from matplotlib import pyplot as plt
import numpy as np
import os

from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0.0, 20.0, 0.1)


if os.path.exists('wdlf_5Gyr_compare_ifmr.npy'):

    wdlf_5Gyr = np.load('wdlf_5Gyr_compare_ifmr.npy')
    wdlf_7Gyr = np.load('wdlf_7Gyr_compare_ifmr.npy')
    wdlf_9Gyr = np.load('wdlf_9Gyr_compare_ifmr.npy')
    wdlf_11Gyr = np.load('wdlf_11Gyr_compare_ifmr.npy')
    wdlf_13Gyr = np.load('wdlf_13Gyr_compare_ifmr.npy')
    wdlf_15Gyr = np.load('wdlf_15Gyr_compare_ifmr.npy')

else:

    # default C08
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08_15Gyr = copy.deepcopy(wdlf.number_density)

    # C08b
    wdlf.set_ifmr_model("C08b")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08b_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08b_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08b_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08b_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08b_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C08b_15Gyr = copy.deepcopy(wdlf.number_density)

    # S09
    wdlf.set_ifmr_model("S09")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09_15Gyr = copy.deepcopy(wdlf.number_density)

    # S09b
    wdlf.set_ifmr_model("S09b")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09b_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09b_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09b_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09b_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09b_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_S09b_15Gyr = copy.deepcopy(wdlf.number_density)

    # W09
    wdlf.set_ifmr_model("W09")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_W09_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_W09_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_W09_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_W09_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_W09_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_W09_15Gyr = copy.deepcopy(wdlf.number_density)

    # K09
    wdlf.set_ifmr_model("K09")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09_15Gyr = copy.deepcopy(wdlf.number_density)

    # K09b
    wdlf.set_ifmr_model("K09b")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09b_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09b_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09b_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09b_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09b_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_K09b_15Gyr = copy.deepcopy(wdlf.number_density)

    # C18
    wdlf.set_ifmr_model("C18")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C18_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C18_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C18_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C18_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C18_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_C18_15Gyr = copy.deepcopy(wdlf.number_density)

    # EB18
    wdlf.set_ifmr_model("EB18")
    wdlf.set_sfr_model(mode="burst", age=5.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_EB18_5Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=7.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_EB18_7Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=9.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_EB18_9Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=11.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_EB18_11Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=13.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_EB18_13Gyr = copy.deepcopy(wdlf.number_density)
    wdlf.set_sfr_model(mode="burst", age=15.0e9, duration=1e8)
    wdlf.compute_density(Mag=Mag)
    wdlf_EB18_15Gyr = copy.deepcopy(wdlf.number_density)

    wdlf_5Gyr = np.vstack(
        (
            wdlf_C08_5Gyr,
            wdlf_C08b_5Gyr,
            wdlf_S09_5Gyr,
            wdlf_S09b_5Gyr,
            wdlf_W09_5Gyr,
            wdlf_K09_5Gyr,
            wdlf_K09b_5Gyr,
            wdlf_C18_5Gyr,
            wdlf_EB18_5Gyr,
        )
    )
    wdlf_7Gyr = np.vstack(
        (
            wdlf_C08_7Gyr,
            wdlf_C08b_7Gyr,
            wdlf_S09_7Gyr,
            wdlf_S09b_7Gyr,
            wdlf_W09_7Gyr,
            wdlf_K09_7Gyr,
            wdlf_K09b_7Gyr,
            wdlf_C18_7Gyr,
            wdlf_EB18_7Gyr,
        )
    )
    wdlf_9Gyr = np.vstack(
        (
            wdlf_C08_9Gyr,
            wdlf_C08b_9Gyr,
            wdlf_S09_9Gyr,
            wdlf_S09b_9Gyr,
            wdlf_W09_9Gyr,
            wdlf_K09_9Gyr,
            wdlf_K09b_9Gyr,
            wdlf_C18_9Gyr,
            wdlf_EB18_9Gyr,
        )
    )
    wdlf_11Gyr = np.vstack(
        (
            wdlf_C08_11Gyr,
            wdlf_C08b_11Gyr,
            wdlf_S09_11Gyr,
            wdlf_S09b_11Gyr,
            wdlf_W09_11Gyr,
            wdlf_K09_11Gyr,
            wdlf_K09b_11Gyr,
            wdlf_C18_11Gyr,
            wdlf_EB18_11Gyr,
        )
    )
    wdlf_13Gyr = np.vstack(
        (
            wdlf_C08_13Gyr,
            wdlf_C08b_13Gyr,
            wdlf_S09_13Gyr,
            wdlf_S09b_13Gyr,
            wdlf_W09_13Gyr,
            wdlf_K09_13Gyr,
            wdlf_K09b_13Gyr,
            wdlf_C18_13Gyr,
            wdlf_EB18_13Gyr,
        )
    )
    wdlf_15Gyr = np.vstack(
        (
            wdlf_C08_15Gyr,
            wdlf_C08b_15Gyr,
            wdlf_S09_15Gyr,
            wdlf_S09b_15Gyr,
            wdlf_W09_15Gyr,
            wdlf_K09_15Gyr,
            wdlf_K09b_15Gyr,
            wdlf_C18_15Gyr,
            wdlf_EB18_15Gyr,
        )
    )

    np.save('wdlf_5Gyr_compare_ifmr', wdlf_5Gyr)
    np.save('wdlf_7Gyr_compare_ifmr', wdlf_7Gyr)
    np.save('wdlf_9Gyr_compare_ifmr', wdlf_9Gyr)
    np.save('wdlf_11Gyr_compare_ifmr', wdlf_11Gyr)
    np.save('wdlf_13Gyr_compare_ifmr', wdlf_13Gyr)
    np.save('wdlf_15Gyr_compare_ifmr', wdlf_15Gyr)


# normalise the WDLFs relative to the density at 10 mag
wdlf_5Gyr = [wdlf_5Gyr[i]/wdlf_5Gyr[i][Mag==10.0] for i in range(len(wdlf_5Gyr))]
wdlf_7Gyr = [wdlf_7Gyr[i]/wdlf_7Gyr[i][Mag==10.0] for i in range(len(wdlf_7Gyr))]
wdlf_9Gyr = [wdlf_9Gyr[i]/wdlf_9Gyr[i][Mag==10.0] for i in range(len(wdlf_9Gyr))]
wdlf_11Gyr = [wdlf_11Gyr[i]/wdlf_11Gyr[i][Mag==10.0] for i in range(len(wdlf_11Gyr))]
wdlf_13Gyr = [wdlf_13Gyr[i]/wdlf_13Gyr[i][Mag==10.0] for i in range(len(wdlf_13Gyr))]
wdlf_15Gyr = [wdlf_15Gyr[i]/wdlf_15Gyr[i][Mag==10.0] for i in range(len(wdlf_15Gyr))]


ifmr_list = ["C08", "C08b", "S09", "S09b", "W09", "K09", "K09b", "C18", "EB18"]

fig1, ((ax1, ax9), (ax2, ax10), (ax3, ax11), (ax4, ax12), (ax5, ax13), (ax6, ax14), (ax7, ax15), (ax8, ax16)) = plt.subplots(
    8, 2, sharex=True, figsize=(12, 15), gridspec_kw={'height_ratios': [8,4,1,8,4,1,8,4]}
)

# Burst SFR 5 Gyr
for w, l in zip(wdlf_5Gyr, ifmr_list):
    ax1.plot(Mag, np.log10(w), label=l)
    ax2.plot(Mag, np.log10(w/wdlf_5Gyr[0]))

# Burst SFR 7 Gyr
for w in wdlf_7Gyr:
    ax4.plot(Mag, np.log10(w))
    ax5.plot(Mag, np.log10(w/wdlf_7Gyr[0]))

# Burst SFR 9 Gyr
for w in wdlf_9Gyr:
    ax7.plot(Mag, np.log10(w))
    ax8.plot(Mag, np.log10(w/wdlf_9Gyr[0]))

# Burst SFR 11 Gyr
for w in wdlf_11Gyr:
    ax9.plot(Mag, np.log10(w))
    ax10.plot(Mag, np.log10(w/wdlf_11Gyr[0]))

# Burst SFR 13 Gyr
for w in wdlf_13Gyr:
    ax12.plot(Mag, np.log10(w))
    ax13.plot(Mag, np.log10(w/wdlf_13Gyr[0]))

# Burst SFR 13 Gyr
for w in wdlf_15Gyr:
    ax15.plot(Mag, np.log10(w))
    ax16.plot(Mag, np.log10(w/wdlf_15Gyr[0]))

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

ax3.axis('off')
ax6.axis('off')
ax11.axis('off')
ax14.axis('off')

ax1.set_xlim(2, 20)

ax1.set_ylim(-6.5, 0)
ax4.set_ylim(-6.5, 0)
ax7.set_ylim(-6.5, 0)
ax9.set_ylim(-6.5, 0)
ax12.set_ylim(-6.5, 0)
ax15.set_ylim(-6.5, 0)

ax1.set_ylim(-3, 3.5)
ax4.set_ylim(-3, 3.5)
ax7.set_ylim(-3, 3.5)
ax9.set_ylim(-3, 3.5)
ax12.set_ylim(-3, 3.5)
ax15.set_ylim(-3, 3.5)

ax9.set_yticklabels([""])
ax10.set_yticklabels([""])
ax11.set_yticklabels([""])
ax12.set_yticklabels([""])
ax13.set_yticklabels([""])
ax14.set_yticklabels([""])
ax15.set_yticklabels([""])
ax16.set_yticklabels([""])

ax1.set_title('5 Gyr')
ax4.set_title('7 Gyr')
ax7.set_title('9 Gyr')
ax9.set_title('11 Gyr')
ax12.set_title('13 Gyr')
ax15.set_title('15 Gyr')

ax1.set_ylabel("log(n)")
ax2.set_ylabel(r"log(n/n$_{C08}$)")
ax4.set_ylabel("log(n)")
ax5.set_ylabel(r"log(n/n$_{C08}$)")
ax7.set_ylabel("log(n)")
ax8.set_ylabel(r"log(n/n$_{C08}$)")

ax2.set_ylim(-1, 1)
ax5.set_ylim(-1, 1)
ax8.set_ylim(-1, 1)
ax10.set_ylim(-1, 1)
ax13.set_ylim(-1, 1)
ax16.set_ylim(-1, 1)


fig1.supxlabel(r"M$_{\mathrm{bol}}$ / mag")

plt.subplots_adjust(
    left=0.1, right=0.98, top=0.96, bottom=0.075, hspace=0.00, wspace=0.01
)
plt.savefig("wdlf_compare_ifmr.png")
plt.savefig("wdlf_compare_ifmr.pdf")

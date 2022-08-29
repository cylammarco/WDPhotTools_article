import glob
import os
import pkg_resources

from astropy.modeling import models
from astropy import units as u
import extinction
from matplotlib import pyplot as plt
import numpy as np
from spectres import spectres


def _fitzpatrick99(wavelength, Rv):
    """
    Parameters
    ----------
    wavelegnth: array
        Wavelength in Angstrom
    Rv: float
        The extinction in V per unit of A_V, i.e. A_V/E(B − V)

    Return
    ------
    The extinction at the given wavelength per unit of E(B − V),
    i.e. A/E(B - V)

    """

    if isinstance(wavelength, (float, int)):

        wavelength = np.array([wavelength])

    return extinction.fitzpatrick99(wavelength, 1.0, Rv) * Rv


atm_key = np.array(
    [
        "U",
        "B",
        "G3_BP",
        "V",
        "G3",
    ]
)
filter_key = np.array(
    [
        "Generic_Johnson.U",
        "Generic_Johnson.B",
        "GAIA_GAIA3.Gbp",
        "Generic_Johnson.V",
        "GAIA_GAIA3.G",
    ]
)
filter_colours = np.array(
    [
        "slateblue",
        "dodgerblue",
        "forestgreen",
        "orange",
        "red",
    ]
)

filter_order = {}
for i, k in enumerate(atm_key):
    filter_order[k] = i


filter_name_mapping = {}
for i, j in zip(filter_key, atm_key):
    filter_name_mapping[i] = j

model_filelist = glob.glob(
    os.path.join(
        pkg_resources.resource_filename("WDPhotTools", "koester_model"), "*"
    )
)
filter_filelist = glob.glob(
    os.path.join(
        pkg_resources.resource_filename("WDPhotTools", "filter_response"), "*"
    )
)


limit = 1e-3

# normalisation factor of the exponent
# 0.78 comes from Shlafly et al. 2010
# 1.32 also comes from them, it is the O'Donnell extinction at 1 micron
norm = 0.78 * 1.32 / 2.5 * limit

# Get the temperature and logg
teff = []
logg = []
filter = []
rv31 = {}
temperature_to_plot = np.array([5000.0, 10000.0, 30000.0])

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

counter = 0
for j, model in enumerate(model_filelist):
    temp = model.split("\\da")[-1].split("_")
    t = float(temp[0])
    g = float(temp[1].split(".")[0]) / 100.0
    if np.isclose(float(temp[0]), temperature_to_plot, atol=0.1).any() & (
        g == 8.0
    ):
        rv31[str(t)] = {}
        s_wave, s_flux = np.loadtxt(model).T
        bb_wave = np.arange(s_wave[-1], 300000)
        bb = models.BlackBody(temperature=t * u.K)
        bb_flux = bb(bb_wave * u.AA)
        bb_flux /= bb_flux[0]
        bb_flux *= s_flux[-1]
        total_wave = np.concatenate((s_wave, bb_wave[1:]))
        total_flux = np.concatenate((s_flux, bb_flux[1:]))
        wave_mask = (total_wave > 2000.0) & (total_wave < 12000.0)
        total_wave = total_wave[wave_mask]
        total_flux = total_flux[wave_mask]
        total_flux /= total_flux[np.argmin(abs(total_wave - 4200.0))] * 2
        ax1.plot(
            total_wave,
            total_flux,
            color=str(0.75 - counter * 0.25),
            label=str(t),
        )
        A_1um_31 = _fitzpatrick99(t, 3.1)
        counter += 1
        for i in filter_filelist:
            filter_i = i.split(os.sep)[-1].split(".dat")[0]
            if filter_i in filter_key:
                f_wave, f_response = np.loadtxt(i).T
                wave_bin = np.zeros_like(f_wave)
                wave_diff = np.diff(f_wave) / 2.0
                wave_bin[:-1] = wave_diff
                wave_bin[1:] += wave_diff
                wave_bin[0] += wave_diff[0]
                wave_bin[-1] += wave_diff[-1]
                current_filter = filter_name_mapping[i.split("\\")[-1][:-4]]
                filter.append(current_filter)
                teff.append(t)
                logg.append(g)
                # * 5.03411250E+07 / (3.08568e19)**2. for converting from flux to
                # photon is not needed because they cancel each other in the
                # normalisation. The s_wave is the only non-linear multiplier
                total_flux_resampled = spectres(
                    f_wave,
                    total_wave,
                    total_flux * total_wave,
                    fill=0.0,
                    verbose=False,
                )
                # source flux convolves with filter response
                SxW = total_flux_resampled * f_response * wave_bin
                rv31[str(t)][filter_i] = -2.5 * np.log10(
                    np.sum(
                        SxW
                        * 10.0
                        ** (-_fitzpatrick99(f_wave, 3.1) / A_1um_31 * norm)
                    )
                    / np.sum(SxW)
                )


for i in filter_filelist:
    if i.split(os.sep)[-1].split(".dat")[0] in filter_key:
        f_wave, f_response = np.loadtxt(i).T
        current_filter = filter_name_mapping[i.split("\\")[-1][:-4]]
        ax1.plot(
            f_wave,
            f_response / max(f_response),
            color=filter_colours[filter_order[current_filter]],
            label=current_filter,
        )


ax1.set_xlim(3000, 8000)
ax1.set_xtickslabels([])
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Arbitrary Flux and Filter Response")
ax1.legend(loc="upper right")


ax1b = ax1.twinx()
_ext31 = _fitzpatrick99(total_wave, Rv=3.1)
ax1b.plot(total_wave, _ext31, color="saddlebrown", ls=":")
ax1b.text(7000.0, 1.7, "Rv=3.1", color="saddlebrown")
ax1b.set_ylim(0, 6.0)
ax1b.set_ylabel("Extinction (mag / E(B - V))", color="saddlebrown")

ax1b.spines["right"].set_color("saddlebrown")
[t.set_color("saddlebrown") for t in ax1b.yaxis.get_ticklines()]
[t.set_color("saddlebrown") for t in ax1b.yaxis.get_ticklabels()]

# U, B, GBP, V, G
pivot_wavelength = np.array([3585.0, 4371.0, 5110.0, 5478.0, 6218.0])
ax2.scatter()
ax2.set_xlabel("Wavelength / A")

rv31_U = [
    rv31["5000.0"]["Generic_Johnson.U"],
    rv31["10000.0"]["Generic_Johnson.U"],
    rv31["30000.0"]["Generic_Johnson.U"],
]
rv31_B = [
    rv31["5000.0"]["Generic_Johnson.B"],
    rv31["10000.0"]["Generic_Johnson.B"],
    rv31["30000.0"]["Generic_Johnson.B"],
]
rv31_V = [
    rv31["5000.0"]["Generic_Johnson.V"],
    rv31["10000.0"]["Generic_Johnson.V"],
    rv31["30000.0"]["Generic_Johnson.V"],
]
rv31_G = [
    rv31["5000.0"]["GAIA_GAIA3.G"],
    rv31["10000.0"]["GAIA_GAIA3.G"],
    rv31["30000.0"]["GAIA_GAIA3.G"],
]
rv31_GBP = [
    rv31["5000.0"]["GAIA_GAIA3.Gbp"],
    rv31["10000.0"]["GAIA_GAIA3.Gbp"],
    rv31["30000.0"]["GAIA_GAIA3.Gbp"],
]

rv31_list = [rv31_U, rv31_B, rv31_GBP, rv31_V, rv31_G]

for i, w in enumerate(pivot_wavelength):
    ax2.scatter([w] * 3, rv31_list[i], color=filter_colours[i])


ax2.set_ylabel("A / E(B-V)$_{Rv=3.1}$")

plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
plt.savefig("fig_12_reddening_filter.png")
plt.savefig("fig_12_reddening_filter.pdf")

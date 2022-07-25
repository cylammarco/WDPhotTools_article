import numpy as np
from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0, 20.0, 2.5)
age = [3.0e9]
num = np.zeros((len(age), len(Mag)))

wdlf.set_sfr_model(mode="burst", age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
fig_input_models = wdlf.plot_input_models(
    cooling_model_use_mag=False,
    imf_log=True,
    display=True,
    folder=".",
    filename="fig_07_input_models",
    ext=["png", "pdf"],
    savefig=True,
)

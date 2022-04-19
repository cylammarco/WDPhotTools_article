from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools.fitter import WDfitter
from WDPhotTools.reddening import reddening_vector_interpolated


reddening = reddening_vector_interpolated(kind='cubic')
wave_GBRugriz = np.array((6218., 5110., 7769., 3557., 4702., 6175., 7491., 8946.))

#data = fits.open('GaiaEDR3_WD_SDSSspec.FITS')
data = fits.open('gaiaedr3_wd_main.fits.gz')

G3 = data[1].data['phot_g_mean_mag_corrected']
G3_error = data[1].data['phot_g_mean_mag_error_corrected']
G3_BP = data[1].data['phot_bp_mean_mag']
G3_BP_error = data[1].data['phot_bp_mean_mag_error']
G3_RP = data[1].data['phot_rp_mean_mag']
G3_RP_error = data[1].data['phot_rp_mean_mag_error']

u_sdss = data[1].data['umag']
u_sdss_error = data[1].data['e_umag']
g_sdss = data[1].data['gmag']
g_sdss_error = data[1].data['e_gmag']
r_sdss = data[1].data['rmag']
r_sdss_error = data[1].data['e_rmag']
i_sdss = data[1].data['imag']
i_sdss_error = data[1].data['e_imag']
z_sdss = data[1].data['zmag']
z_sdss_error = data[1].data['e_zmag']

Av = data[1].data['meanAV']

parallax = data[1].data['parallax'] / 1000
parallax_error = data[1].data['parallax_error']

distance = 1. / parallax
distance_error = distance**2. * parallax_error

ftr = WDfitter()
output = []
length = len(G3)
teff_h = np.zeros(length)
logg_h = np.zeros(length)
teff_he = np.zeros(length)
logg_he = np.zeros(length)
for i in range(length):
    print('{} of {}'.format(i + 1, length))
    ebv = Av[i] / reddening(wave_GBRugriz, 3.1)
    ftr.fit(filters=[
        'G3', 'G3_BP', 'G3_RP', 'u_sdss', 'g_sdss', 'r_sdss', 'i_sdss',
        'z_sdss'
    ],
            mags=[
                G3[i], G3_BP[i], G3_RP[i], u_sdss[i], g_sdss[i], r_sdss[i],
                i_sdss[i], z_sdss[i]
            ],
            mag_errors=[
                G3_error[i], G3_BP_error[i], G3_RP_error[i], u_sdss_error[i],
                g_sdss_error[i], r_sdss_error[i], i_sdss_error[i],
                z_sdss_error[i]
            ],
            independent=['Teff', 'logg'],
            atmosphere_interpolator='CT',
            allow_none=True,
            reuse_interpolator=False,
            distance=distance[i],
            distance_err=distance_error[i],
            Rv=3.1,
            ebv=ebv,
            initial_guess=[10000.0, 8.0])
    output.append(ftr.results)
    teff_h[i] = ftr.best_fit_params['H']['Teff']
    logg_h[i] = ftr.best_fit_params['H']['logg']
    teff_he[i] = ftr.best_fit_params['He']['Teff']
    logg_he[i] = ftr.best_fit_params['He']['logg']


np.save('compare_fusillo_edr3_output', output)
np.save('compare_fusillo_edr3_teff_h', teff_h)
np.save('compare_fusillo_edr3_logg_h', logg_h)
np.save('compare_fusillo_edr3_teff_he', teff_he)
np.save('compare_fusillo_edr3_logg_he', logg_he)


plt.figure(1, figsize=(8, 8))
plt.clf()
plt.scatter(data[1].data['teff_H'][:length], teff_h, s=5)
plt.xlabel('Temperature / K (Fusillo EDR3)')
plt.ylabel('Temperature / K (This work)')
plt.title('DA')
plt.grid()
plt.tight_layout()
plt.savefig('compare_fusillo_edr3.png')
plt.savefig('compare_fusillo_edr3.pdf')

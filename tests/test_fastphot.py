import numpy as npy
import numpy.testing as npt
from fastphot.fastphot import gaussian_PSF, build_catalog, model_MAP
from fastphot.fastphot import src_dtype, fastphot, save_pdf_MAP


def test_gaussian_PSF():
    """
    Testing the normalization of the gaussian PSF model
    For npix_x >> std
    sum(MAP_PSF) --> 1.0

    # """
    # Define PSF properties
    std = 2.0
    npix = int(100. * std)
    #
    # Create MAP_PSF
    PSF_MAP = gaussian_PSF(npix=npix, std=std)
    #
    # Test
    npt.assert_almost_equal(npy.sum(PSF_MAP), 1.e0, decimal=6)


def test_no_noise():
    # """
    # Testing the flux measurement on a small MAP npix_x, npix_y = 50, 50
    # with 11 sources without noise
    # source fluxes are homogeneously distributed between 0.5 and 10.0
    # """
    #
    # create the PSF_MAP
    PSF_MAP = gaussian_PSF(npix=31, std=2.0)
    #
    # Create the SC_MAP with some sources
    npix_x = 150
    npix_y = 150
    N_srcs = 110
    #
    SC_MAP = npy.zeros([npix_x, npix_y])
    MASK_MAP = (SC_MAP > 0.e0)
    #
    # Define flux domain
    minflux = 5.e-1
    maxflux = 1.e1
    #
    # Build the reference catalog
    xpos = npy.random.uniform(1, npix_x - 2, size=N_srcs)
    ypos = npy.random.uniform(1, npix_y - 2, size=N_srcs)
    flux = npy.random.uniform(minflux, maxflux, size=N_srcs)
    Catalog = build_catalog(xpos, ypos, flux)
    #
    SC_MAP = npy.ma.array(model_MAP(SC_MAP, PSF_MAP, Catalog), mask=MASK_MAP)
    #
    # Create the NOISE_MAP
    NOISE_MAP = npy.ma.array(npy.ones([npix_x, npix_y]), mask=MASK_MAP)
    #
    # Mask negative flux sources
    mask = (Catalog['flux'] < 0.)
    Masked_Catalog = npy.ma.array(Catalog, dtype=src_dtype(), mask=mask)
    # Save initial fluxes
    flux_in = npy.ma.compressed(Masked_Catalog['flux'])
    #
    Final_Catalog, bkg, RESIDUAL_MAP = fastphot(SC_MAP, PSF_MAP, NOISE_MAP,
                                                Masked_Catalog, nb_process=4)
    #
    # Test
    npt.assert_almost_equal(npy.ma.compressed(Final_Catalog['flux']), flux_in,
                            decimal=-6)
    #
    # Save SC_MAP
    save_pdf_MAP(SC_MAP, map_name='SC', src_cat=Final_Catalog)
    # Save RESIDUAL_MAP
    save_pdf_MAP(RESIDUAL_MAP, map_name='RESIDUAL', src_cat=Final_Catalog)

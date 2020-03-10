from scipy import signal                 # Convolution
from os.path import join                 # build path
#
import numpy as npy                      # array
import multiprocessing as mp             # distribution onto multi process
import math                              # for math operations
import matplotlib.pyplot as plt          # plot
import matplotlib.gridspec as gridspec   # for multiplot grid
import matplotlib.cm as cm               # colormap
import matplotlib.colors as colors       # colors
#
__all__ = ["src_dtype", "build_catalog", "gaussian_PSF", "fastphot",
           "model_MAP", "save_pdf_MAP"]
__test_path__ = 'tests'


def src_dtype():
    src_dtype = [('ID', int),
                 ('x_pos', float),
                 ('dx_pos', float),
                 ('y_pos', float),
                 ('dy_pos', float),
                 ('flux', float),
                 ('dflux', float)]
    return src_dtype


def build_catalog(xpos, ypos, flux):
    """
    Return a catalog of sources according to the source type src_type

    Parameters
    ----------
    xpos : npy array
        The "x" positions of the sources
    ypos : npy array
        The "y" positions of the sources
    flux : npy array
        The flux of the sources

    Returns
    -------
    Catalog : a npy array of sources
        The catalog of sources
    """

    N_srcs = len(xpos)
    if ((len(ypos) != N_srcs) or (len(flux) != N_srcs)):
        raise NameError('build_catalog: Unconsistent size xpos, ypos, flux')

    for i in range(len(xpos)):
        #
        s = (i, xpos[i], 0.e0, ypos[i], 0.e0, flux[i], 0.e0)
        #
        # Update source dict
        if (i == 0):
            # create the structured list
            Catalog = npy.array(s, dtype=src_dtype())
        else:
            Catalog = npy.append(Catalog, npy.array(s, dtype=src_dtype()))
    return Catalog


def gaussian_PSF(npix=31, std=3):
    """
    Return a gaussian PSF_MAP, we assume here PSF_MAP.shape = (npix_x, npix_x)
    By construction, we impose a flux of 1.0

    Parameters
    ----------
    npix : integer
        The size (fisrt and second dimension) of the Point Spread Function MAP
        by default = 31
    std : float
        The standard deviation

    Returns
    -------
    PSF_MAP : npy array
        The Point Spread Function MAP

    """
    #
    g1d = signal.gaussian(npix, std=std).reshape(npix, 1)
    g2d = npy.outer(g1d, g1d)
    g2d = g2d / npy.sum(g2d)
    return g2d


def fastphot(SC_MAP, PSF_MAP, NOISE_MAP, Catalog, nb_process=4):
    """
    Return flux of sources associated to given positions

    Parameters
    ----------
    SC_MAP : numpy masked array.
        The SCientific MAP.
    PSF_MAP : numpy array
        The Point Spread Function MAP.
    NOISE_MAP : numpy masked array
        The Signal/Noise MAP.
    Catalog : numpy scrutured and masked array
        The source catalog.
        It must contain at least in input the source positions
        The Phot function allows to complete it by saving source fluxes
    nb_process : integer
        number of independant cpu(s) used to build A matrix and B vector
        by default we assume nb_process = 4

    Returns
    -------
    RESIDUAL_MAP : numpy masked array
        The residual map (SC_MAP - MODEL_MAP)
    bkg : float
        The background level
    """
    #
    print('> PHOT')
    #
    # extract some information about maps and sources
    SC_MAP_npix_x, SC_MAP_npix_y = SC_MAP.shape
    PSF_MAP_npix_x, PSF_MAP_npix_y = PSF_MAP.shape
    #
    # Compress the input catalog to remove masked sources
    N_src = len(npy.ma.compressed(Catalog['ID']))
    #
    # SC_MAP and NOISE_MAP have to be imersed in a "full" MAP
    # taking into acount an half PSF-size on the edges
    edge_x = int(math.floor(PSF_MAP_npix_x / 2))
    edge_y = int(math.floor(PSF_MAP_npix_y / 2))
    x_i = edge_x
    x_f = x_i + SC_MAP_npix_x
    y_i = edge_y
    y_f = y_i + SC_MAP_npix_y
    # SC_MAP
    SC_full_MAP = npy.zeros([SC_MAP_npix_x + 2 * edge_x,
                             SC_MAP_npix_y + 2 * edge_y])  # create
    SC_full_MAP[x_i:x_f, y_i:y_f] = SC_MAP  # imerse
    # NOISE_MAP
    NOISE_full_MAP = npy.zeros([SC_MAP_npix_x + 2 * edge_x,
                                SC_MAP_npix_y + 2 * edge_y])  # create
    NOISE_full_MAP[x_i:x_f, y_i:y_f] = NOISE_MAP  # imerse
    #
    # Create the mask
    MASK = (NOISE_full_MAP <= 0.e0)
    #
    # Convert SC_MAP and NOISE_MAP in masked array
    SC_full_MAP = npy.ma.array(SC_full_MAP, mask=MASK)
    NOISE_full_MAP = npy.ma.array(NOISE_full_MAP, mask=MASK)
    #
    # Init B and F vectors and A matrix
    B = npy.zeros(N_src + 1)
    A = npy.zeros([N_src + 1, N_src + 1])
    F = npy.zeros(N_src + 1)
    #
    # Build Vectors and Matrix
    pool = mp.Pool(processes=nb_process)
    # (i, Bi, Ai_, A_)
    X_pos = npy.ma.compressed(Catalog['x_pos'])
    Y_pos = npy.ma.compressed(Catalog['y_pos'])
    R = [pool.apply_async(Coef_i, args=(SC_full_MAP, NOISE_full_MAP, PSF_MAP,
                                        X_pos, Y_pos, si))
         for si in range(N_src)]
    # Reformat result, build A and B
    for ri in R:
        r_i = ri.get()
        B[r_i[0]] = r_i[1]
        A[r_i[0], r_i[0]:N_src] = r_i[2]
        A[r_i[0]:N_src, r_i[0]] = r_i[2]
        A[r_i[0]][N_src] = r_i[3]
        A[N_src][r_i[0]] = r_i[3]
    # Complete
    B[N_src] = npy.nansum(SC_full_MAP / NOISE_full_MAP**2.)
    A[N_src][N_src] = npy.nansum(NOISE_full_MAP**(-2.))
    #
    # Solve system)
    F = npy.linalg.solve(A, B)
    dF = npy.diag(npy.linalg.inv(A[:N_src, :N_src]))
    #
    # Update FLux field in the catalog
    Catalog['flux'][~Catalog['ID'].mask] = F[:N_src] - npy.ones(len(F[:N_src])) * F[N_src]
    Catalog['dflux'][~Catalog['ID'].mask] = npy.sqrt(dF)
    #
    # Build residual MAP
    print(' > Build Residual Map')
    RESIDUAL_MAP = SC_MAP - model_MAP(SC_MAP, PSF_MAP, Catalog)
    #
    return Catalog, F[N_src], RESIDUAL_MAP
#
# -----------------------------------
# CREATE A SOURCE AT A GIVEN POSITION
# -----------------------------------
def create_source_at_pos(x_pos, y_pos, npix_x, npix_y, PSF_MAP, MASK):
    """
    Create a source according to the PSF_MAP at pos
    x_pos', 'y_pos', in a map of size, npix_x, npix_y
    As a source position is a float and therefore contain decimal,
    we distribute the position of the source
    on the four closer pixels (use barycentre position)

    Parameters
    ----------
    x_pos : float.
        The x (first dimension) position of the source
    y_pos : float
        The y (second dimension) position of the source
    npix_x : integer
        The x (fisrt dimension) size of the scientific map
    npix_y : integer
        The y (second dimension) size of the scientific map
    PSF_MAP : numpy array
        The Point Spread Function MAP.
    MASK : numpy boolean array
        This mask has to be setled to False for "bad" pixels
        It can set to True everywhere or build on a set of criterion
        (e.g. Signal/Noise > threshold)

    Returns
    -------
    SRC_MAP : numpy masked array
        The SouRCe MAP. A map of size (npix_x, npix_y) with a source
        created according to the PSF_MAP at the position (x_pos, y_pos)
    """
    #
    # Extract information about PSF_MAP
    PSF_MAP_npix_x, PSF_MAP_npix_y = PSF_MAP.shape
    # create the SRC_MAP
    SRC_MAP = npy.zeros([npix_x, npix_y])
    # Create the POS_MAP
    POS_MAP = npy.zeros(PSF_MAP.shape)
    # Compute coefficients
    rx_pos = math.floor(x_pos)
    ry_pos = math.floor(y_pos)
    Rx = 1.e0 - x_pos + rx_pos
    Ry = 1.e0 - y_pos + ry_pos
    F = npy.array([[Rx * Ry, Rx * (1.e0 - Ry)], [(1.e0 - Rx) * Ry, (1.e0 - Rx) * (1.e0 - Ry)]])
    # Add source position
    cpix_x = int(math.floor(PSF_MAP_npix_x / 2)) - 1 + int(round(Rx))
    cpix_y = int(math.floor(PSF_MAP_npix_y / 2)) - 1 + int(round(Ry))
    POS_MAP[cpix_x:cpix_x + 2, cpix_y:cpix_y + 2] = F
    #
    # convolve with PSF and inject in the SRC_MAP
    # Extract corner pixels of the sub-MAP around the position 'x_pos', 'y_pos'
    x_inf = int(rx_pos) - cpix_x
    x_sup = x_inf + PSF_MAP_npix_x
    y_inf = int(ry_pos) - cpix_y
    y_sup = y_inf + PSF_MAP_npix_y
    SRC_MAP[x_inf:x_sup, y_inf:y_sup] = signal.convolve2d(POS_MAP, PSF_MAP,
                                                          boundary='fill',
                                                          mode='same')
    # Apply mask
    SRC_MAP = npy.ma.array(SRC_MAP, mask=MASK)
    #
    return SRC_MAP


def extract_source_at_pos(MAP, x_pos, y_pos, PSF_npix_x, PSF_npix_y):
    """
    Extract a sub-MAP around the position (x_pos, y_pos)
    The shape of the sub-MAP is similar to the PSF_MAP

    Parameters
    ----------
    MAP : numpy masked array
        The original MAP in which the source as to be extracted
    x_pos : float.
        The x (first dimension) position of the source
    y_pos : float
        The y (second dimension) position of the source
    PSF_npix_x : integer
        The x (fisrt dimension) size of the Point Spread Map
    PSF_npix_y : integer
        The y (second dimension) size of the Point Spread Map

    Returns
    -------
    SCR_MAP : numpy masked array
        The SouRCe MAP. A map of size (npix_x, npix_y) with a source
        created according to the PSF_MAP at the position (x_pos, y_pos)
    """
    #
    cpix_x = int(math.floor(PSF_npix_x / 2))
    cpix_y = int(math.floor(PSF_npix_y / 2))
    # corner pixels
    x_inf = int(round(x_pos)) - cpix_x; x_sup = x_inf + PSF_npix_x
    y_inf = int(round(y_pos)) - cpix_y; y_sup = y_inf + PSF_npix_y
    # extract map
    SRC_MAP = MAP[x_inf:x_sup, y_inf:y_sup]
    #
    return SRC_MAP


def Coef_i(SC_MAP, NOISE_MAP, PSF_MAP, X_pos, Y_pos, i):
    """
    Build coefficients of the A Matrix and B vector

    B_i takes into account  : - the SC_MAP around position xi_pos, yi_pos
                              - a source model at position xi_pos, yi_pos
                              - the NOISE_MAP around position xi_pos, yi_pos

    A_ij takes into account : - a source model at position xi_pos, yi_pos
                              - a source model at position xj_pos, yj_pos
                                     EXTRACTED AT POSITION xi_pos, yi_pos
                              - the NOISE_MAP around position xi_pos, yi_pos
                              - the NOISE_MAP around position xi_pos, yi_pos
    Parameters
    ----------
    SC_MAP : numpy masked array.
        The SCientific MAP.
    NOISE_MAP : numpy masked array
        The Signal/Noise MAP.
    PSF_MAP : numpy array
        The Point Spread Function MAP.
    X_pos : numpy array
        list of x (first dimension) source positions.
    Y_pos : numpy array
        list of y (second dimension) source positions.
    i : integer
        index of the source: i in [0: len(X_pos)-1]

    Returns
    -------
    i : integer
        index of the source: i in [0: len(X_pos)-1]
        we return the index to sort correctly matrix elements
    Bi : float
        The ith element of the B vector
    Ai_ : numpy array of floats
        A matrix coefficent associated to the ith row; j in [i: Nsrc -1]
    A_ : float
        The A[Nsrc][Nsrc] matrix coefficiant
    """
    #
    # extract information about SC_MAP and PSF_MAP
    SC_MAP_npix_x, SC_MAP_npix_y = SC_MAP.shape
    PSF_MAP_npix_x, PSF_MAP_npix_y = PSF_MAP.shape
    #
    # Deduce number of sources
    Nsrcs = len(X_pos)
    #
    # To take into account new map size (with edges), source positions have to be shifted
    xi_pos = X_pos[i] + int(math.floor(PSF_MAP_npix_x / 2))
    yi_pos = Y_pos[i] + int(math.floor(PSF_MAP_npix_y / 2))
    #
    # 1-/ Extract a sub MAP in the SC_MAP around position xi_pos, yi_pos
    SUB_SC_MAP = extract_source_at_pos(SC_MAP, xi_pos, yi_pos, PSF_MAP_npix_x,
                                       PSF_MAP_npix_y)
    #
    # 2-/ Create a source model at position xi_pos, yi_pos and extract the PSF_i_MAP
    MODEL_MAP = create_source_at_pos(xi_pos, yi_pos, SC_MAP_npix_x,
                                     SC_MAP_npix_y, PSF_MAP, SC_MAP.mask)
    PSF_i_MAP = extract_source_at_pos(MODEL_MAP, xi_pos, yi_pos,
                                      PSF_MAP_npix_x, PSF_MAP_npix_y)
    #
    # 3-/ Extract a sub MAP in the NOISE_MAP around position xi_pos, yi_pos
    SUB_NOISE_MAP = extract_source_at_pos(NOISE_MAP, xi_pos, yi_pos,
                                          PSF_MAP_npix_x, PSF_MAP_npix_y)
    #
    # Compute B_i (i in [1; Nsrcs])
    Bi = npy.ma.sum(SUB_SC_MAP*PSF_i_MAP / SUB_NOISE_MAP**2.)
    #
    # Compute A_ (j = Nsrcs)
    A_ = npy.ma.sum(PSF_i_MAP / SUB_NOISE_MAP**2.)
    #
    # init
    Ai_ = []
    # loop over j sources
    for j in range(i, Nsrcs):
        #
        xj_pos = X_pos[j] + int(math.floor(PSF_MAP_npix_x / 2))
        yj_pos = Y_pos[j] + int(math.floor(PSF_MAP_npix_y / 2))
        #
        # Source Recovery
        SR_x = PSF_MAP_npix_x - abs(xi_pos - xj_pos)
        SR_y = PSF_MAP_npix_y - abs(yi_pos - yj_pos)
        #
        A = 0.e0
        if ((SR_x > 0.e0) and (SR_y > 0.e0)):
            #
            # 4-/ Create a source model at position xj_pos,
            # yj_pos and extract the PSF_j_MAP at position xi_pos, yi_pos
            MODEL_MAP = create_source_at_pos(xj_pos, yj_pos, SC_MAP_npix_x,
                                             SC_MAP_npix_y, PSF_MAP,
                                             SC_MAP.mask)
            PSF_j_MAP = extract_source_at_pos(MODEL_MAP, xi_pos, yi_pos,
                                              PSF_MAP_npix_x, PSF_MAP_npix_y)
            #
            # Compute A_ij
            A = npy.ma.sum(PSF_i_MAP * PSF_j_MAP / SUB_NOISE_MAP**2.)
            #
        Ai_.append(A)
        #
    return i, Bi, Ai_, A_


def model_MAP(SC_MAP, PSF_MAP, Catalog):
    """
    Build a complete MODEL_MAP by injecting source (according to the PSF_MAP)
    At all source positions with the corresponding flux

    Parameters
    ----------
    SC_MAP : numpy masked array.
        The SCientific MAP. Allow to extract the shape
    PSF_MAP : numpy array
        The Point Spread Function MAP.
    Catalog : numpy scrutured and masked array
        The source catalog. To build the MODEL mask, this catalog has to
        contain at least in input the source positions and source fluxes

    Returns
    -------
    MODEL_MAP : npy.array
        The MODEL_MAP in which all sources of the Catalog have been injected according to the PSF_MAP
    """
    #
    # Extract informations
    SC_MAP_npix_x, SC_MAP_npix_y = SC_MAP.shape
    PSF_MAP_npix_x, PSF_MAP_npix_y = PSF_MAP.shape
    #
    edge_x = int(math.floor(PSF_MAP_npix_x / 2))
    edge_y = int(math.floor(PSF_MAP_npix_y / 2))
    x_i = edge_x
    x_f = x_i + SC_MAP_npix_x
    y_i = edge_y
    y_f = y_i + SC_MAP_npix_y
    #
    # Init MODEL_MAP
    MODEL_MAP = npy.zeros([SC_MAP_npix_x + 2 * edge_x, SC_MAP_npix_y + 2 * edge_y])  # create
    MOD_MAP_npix_x, MOD_MAP_npix_y = MODEL_MAP.shape
    #
    # Create the MASK
    MASK = (MODEL_MAP > 0.e0)
    #
    # Extract source position in the Catalog
    # To take into accound the edge, source position has to be shifted
    # by PSF_MAP_npix_x/2 and PSF_MAP_npix_y/2
    X_pos = npy.ma.compressed(Catalog['x_pos']) + edge_x
    Y_pos = npy.ma.compressed(Catalog['y_pos']) + edge_y
    # Extract source fluxes in the Catalog
    Flux = npy.ma.compressed(Catalog['flux'])
    #
    # Loop over sources
    for s in range(len(X_pos)):
        #
        MODEL_MAP = MODEL_MAP + create_source_at_pos(X_pos[s], Y_pos[s],
                                                     MOD_MAP_npix_x,
                                                     MOD_MAP_npix_y,
                                                     Flux[s] * PSF_MAP,
                                                     MASK=MASK)
    # Return the MODEL_MAP
    # The MODEL_MAP is resized on the SC_MAP shape
    return MODEL_MAP[x_i:x_f, y_i:y_f]


def save_pdf_MAP(MAP, map_name, src_cat, path=__test_path__):
    """
    Plot a SC_MAP with a cross at each source positions

    """
    # The figure size is setled accordingly to the MAP size
    sx, sy = MAP.shape
    px = min(int(math.floor(sy / 10)), 10)
    py = int(1.1 * math.floor(px * float(sy) / float(sx)))
    #
    fig = plt.figure(figsize=(px, py))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.84, bottom=0.1, top=0.92)
    #
    # define the colormaps
    cmap = cm.jet
    #
    # Create a mask
    MASK = (MAP == 0.e0)
    MAP_ = npy.ma.log10(npy.ma.array(MAP.T, mask=MASK))
    max_z = math.ceil(npy.ma.max(MAP_))
    min_z = max(max_z - 7, math.floor(npy.ma.min(MAP_)))
    #
    bound_z = npy.linspace(float(min_z), float(max_z), int(max_z - min_z) + 1)
    norm_z = colors.Normalize(vmin=min(bound_z), vmax=max(bound_z))
    # create a ScalarMappable and initialize a data structure
    cb_sm = cm.ScalarMappable(cmap=cmap, norm=norm_z)
    cb_sm.set_array([])
    #
    plt.title(map_name + ' MAP')
    plt.imshow(MAP_, extent=[0., sx, 0., sy],
               origin='lower', interpolation='nearest',
               aspect='auto', shape=MAP.shape, cmap=cmap)
    # In matplotlib.imshow()
    # the origin of the pixel is in the corner of the pixel
    # To obtain source positions in the center of their associated pixel
    # we apply a systematic shift of 0.5
    plt.scatter(src_cat['x_pos'] + 0.5, src_cat['y_pos'] + 0.5,
                marker='+', s=100, c='black')
    #
    plt.xlabel('x [pix]')  # label for x axis
    plt.ylabel('y [pix]')  # label for y axis
    #
    filename = map_name + '_MAP.pdf'
    plt.savefig(join(path, filename))
    plt.close()

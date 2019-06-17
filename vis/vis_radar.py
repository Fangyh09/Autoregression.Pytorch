import numpy as np
import sys
sys.path.append('./')
sys.path.append('./fft')

import torch
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ar_image import Corr
from skimage import color
from skimage import io
from boxx import *
from myfft.fft import fft_decompose, fft_recompose
from matplotlib import cm, colors, gridspec


def _dynamic_formatting_floats(floatArray, colorscale='pysteps'):
    """
    Function to format the floats defining the class limits of the colorbar.
    """
    floatArray = np.array(floatArray, dtype=float)

    labels = []
    for label in floatArray:
        if label >= 0.1 and label < 1:
            if colorscale == 'pysteps':
                formatting = ',.2f'
            else:
                formatting = ',.1f'
        elif label >= 0.01 and label < 0.1:
            formatting = ',.2f'
        elif label >= 0.001 and label < 0.01:
            formatting = ',.3f'
        elif label >= 0.0001 and label < 0.001:
            formatting = ',.4f'
        elif label >= 1 and label.is_integer():
            formatting = 'i'
        else:
            formatting = ',.1f'

        if formatting != 'i':
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))

    return labels


def _get_colorlist(units='mm/h', colorscale='pysteps'):
    """
    Function to get a list of colors to generate the colormap.
    Parameters
    ----------
    units : str
        Units of the input array (mm/h, mm or dBZ)
    colorscale : str
        Which colorscale to use (BOM-RF3, pysteps, STEPS-BE)
    Returns
    -------
    color_list : list(str)
        List of color strings.
    clevs : list(float)
        List of precipitation values defining the color limits.
    clevsStr : list(str)
        List of precipitation values defining the color limits
        (with correct number of decimals).
    """

    if colorscale == "BOM-RF3":
        color_list = np.array([(255, 255, 255),  # 0.0
                               (245, 245, 255),  # 0.2
                               (180, 180, 255),  # 0.5
                               (120, 120, 255),  # 1.5
                               (20,  20, 255),   # 2.5
                               (0, 216, 195),    # 4.0
                               (0, 150, 144),    # 6.0
                               (0, 102, 102),    # 10
                               (255, 255,   0),  # 15
                               (255, 200,   0),  # 20
                               (255, 150,   0),  # 30
                               (255, 100,   0),  # 40
                               (255,   0,   0),  # 50
                               (200,   0,   0),  # 60
                               (120,   0,   0),  # 75
                               (40,   0,   0)])  # > 100
        color_list = color_list/255.
        if units == 'mm/h':
            clevs = [0.,0.2, 0.5, 1.5, 2.5, 4, 6, 10, 15, 20, 30, 40, 50, 60, 75,
                    100, 150]
        elif units == "mm":
            clevs = [0.,0.2, 0.5, 1.5, 2.5, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40,
                    45, 50]
        else:
            raise ValueError('Wrong units in get_colorlist: %s' % units)
    elif colorscale == 'pysteps':
        pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgreyHex = '#%02x%02x%02x' % (156, 126, 148)
        color_list = [redgreyHex, "#640064","#AF00AF","#DC00DC","#3232C8","#0064FF","#009696","#00C832",
        "#64FF00","#96FF00","#C8FF00","#FFFF00","#FFC800","#FFA000","#FF7D00","#E11900"]
        if units in ['mm/h', 'mm']:
            # clevs= [0.08,0.16,0.25,0.40,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100,160]
            clevs= [0.08,0.16,0.25,0.40,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100,160]
        elif units == 'dBZ':
            clevs = np.arange(10,65,5)
        else:
            raise ValueError('Wrong units in get_colorlist: %s' % units)
    elif colorscale == 'STEPS-BE':
        color_list = ['cyan','deepskyblue','dodgerblue','blue','chartreuse','limegreen','green','darkgreen','yellow','gold','orange','red','magenta','darkmagenta']
        if units in ['mm/h', 'mm']:
            clevs = [0.1,0.25,0.4,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100]
        elif units == 'dBZ':
            clevs = np.arange(10,65,5)
        else:
            raise ValueError('Wrong units in get_colorlist: %s' % units)

    else:
        print('Invalid colorscale', colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Generate color level strings with correct amount of decimal places
    clevsStr = []
    clevsStr = _dynamic_formatting_floats(clevs, )

    return color_list, clevs, clevsStr


def get_colormap(type, units='mm/h', colorscale='pysteps'):
    """Function to generate a colormap (cmap) and norm.
    Parameters
    ----------
    type : {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If type is 'prob', this specifies the unit of 
        the intensity threshold.
    colorscale : {'pysteps', 'STEPS-BE', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.
    Returns
    -------
    cmap : Colormap instance
        colormap
    norm : colors.Normalize object
        Colors norm
    clevs: list(float)
        List of precipitation values defining the color limits.
    clevsStr: list(str)
        List of precipitation values defining the color limits (with correct
        number of decimals).
    """
    if type in ["intensity", "depth"]:
        # Get list of colors
        color_list,clevs,clevsStr = _get_colorlist(units, colorscale)

        cmap = colors.LinearSegmentedColormap.from_list("cmap", color_list, len(clevs)-1)
        # raise ValueError()
        if colorscale == 'BOM-RF3':
            cmap.set_over('black',1)
        if colorscale == 'pysteps':
            cmap.set_over('darkred',1)
        if colorscale == 'STEPS-BE':
            cmap.set_over('black',1)
        norm = colors.BoundaryNorm(clevs, cmap.N)

        return cmap, norm, clevs, clevsStr

    elif type == "prob":
        cmap = plt.get_cmap("OrRd", 10)
        return cmap, colors.Normalize(vmin=0, vmax=1), None, None
    else:
        return cm.jet, colors.Normalize(), None, None

        
def vis_radar(img, save_name=None):
    """img: numpy
    """
    # revert back from dBZ
    img = 10.0 ** (img / 10.0)
    print(">>>here is img")
    # loga(img)

    cmap, norm, clevs, clevsStr = get_colormap("intensity", units='mm/h', colorscale='pysteps')
    plt.imshow(img, cmap=cmap, norm=norm,
               interpolation='nearest')
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight')
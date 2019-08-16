import numpy as np
import scipy.ndimage as ndimage

from ecoglib.filt.space import pixel_corrections

def correct_for_channels(d, fname, **kwargs):
    if fname.find('test_40') >= 0 or fname.find('test_41') >= 0:

        # careful.. these are given in matlab indexing (1-based).
        # furthermore, the row-major interpretation
        # of the flat array swaps columns for rows

        bad_col = np.array([9, 12]) - 1
        bad_row = np.array([6]) - 1

        bad_single_row = np.array([16, 13, 9, 1, 4, 4,  1,  18, 3,  3,  3]) - 1
        bad_single_col = np.array([3, 4, 5, 6, 7, 11, 14, 11, 18, 19, 20]) - 1

    else:
        pass

    return pixel_corrections(
        d, (bad_row, bad_col), list(zip(bad_single_row, bad_single_col)),
        **kwargs
        )


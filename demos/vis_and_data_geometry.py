# This demo is to note down how an array can be visualized volumetrically
# without (necessarily) making memory copies.
import numpy as np
import ecoglib.vis.data_scroll as dscroll
nrow = 18; ncol = 20
npts = int(1e4)


xx, yy = np.meshgrid(np.arange(float(ncol)), np.arange(float(nrow)))
xx *= np.sqrt(0.5) / 17
yy *= np.sqrt(0.5) / 17
phs_map = np.sqrt(xx**2 + yy**2)

# Here's a case where we have data in shape (nchan, npts)
# There is only one unpacking of the 0th axis that makes sense,
# which is (nrow, ncol).
# Now unfortunately VTK wants this to be (X,Y), but whatevs
arr = np.sin(2*np.pi*(0.05*np.arange(npts) + phs_map.ravel()[:,None]))

# the right way
ds = dscroll.DataScroller(arr, arr[0], rowcol=(nrow, ncol))
ds.configure_traits()
# the wrong way
ds2 = dscroll.DataScroller(arr, arr[0], rowcol=(ncol, nrow))
ds2.configure_traits()

# should see that there has been no copy
print ds.arr_img_dsource.scalar_data.flags

# Now simulate case where data has come from MATLAB (column-major layout).
# The c-contiguous view of this data is shaped (npts, ncol, nrow),
# since rows are the fastest incrementing index in col-major

arr = np.sin(2*np.pi*(0.05*np.arange(npts)[:,None] + phs_map.T.ravel()))
ds3 = dscroll.DataScroller(arr, arr[:,0], rowcol=(nrow,ncol))
ds3.configure_traits()
# again there should not have been a copy
print ds2.arr_img_dsource.scalar_data.flags


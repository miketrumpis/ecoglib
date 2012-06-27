import os.path as osp
import subprocess
import numpy as np
import matplotlib.pyplot as pp
from matplotlib import animation

import nitime.timeseries as ts

class EcogAni(animation.TimedAnimation):

    def __init__(
            self, ecog_ts, nrow, ncol, 
            t0=0, alim=(), fps=20.0, repeat=False
            ):
        if not alim:
            alim = (ecog_ts.data.min(), ecog_ts.data.max())
        fps = float(fps)
        fig = pp.figure( figsize=(5.0,  7.5) )
        im_ax = fig.add_subplot(211)
        ts_ax = fig.add_subplot(212)

        fig.tight_layout()
        
        n_pts = len(ecog_ts)

        # plot the mean trace and time marker on the bottom
        scl = float(ts.time_unit_conversion[ecog_ts.time_unit])
        self.tx = np.array(ecog_ts.time/scl, dtype='d') + t0
        self.ts_plot = ts_ax.plot(
            self.tx, np.mean(ecog_ts.data, axis=0)
            )
        self.t_mark = ts_ax.axvline(
            x=self.tx[0], color='r', ls=':'
            )
        ts_ax.set_xlabel('s')
        ts_ax.set_ylim(alim)
        ts_ax.yaxis.set_visible(False)

        # plot the initial image above ... channel array is contiguous
        # in the row dimension, so the natural image shape in ndarray
        # C-order is (ncol, nrow)
        im_norm = pp.normalize(vmin=alim[0], vmax=alim[1])
        self.ecog_image = im_ax.imshow(
            ecog_ts.data[:,0].reshape(ncol, nrow).T,
            interpolation='nearest', norm=im_norm,
            cmap=pp.cm.jet
            )
        im_ax.set_xlabel('Cols')
        im_ax.xaxis.set_ticks([])
        im_ax.set_ylabel('Rows')
        im_ax.yaxis.set_ticks([])

        self.ecog_ts = ecog_ts
        self.nrow = nrow; self.ncol = ncol
        self.im_ax = im_ax
        self.ts_ax = ts_ax

        animation.TimedAnimation.__init__(
            self, fig, interval=1000/fps, blit=True, repeat=repeat
            )

    def new_frame_seq(self):
        return iter(range(len(self.ecog_ts)))
    
    def _draw_frame(self, framedata):
        self.ecog_image.set_data(
            self.ecog_ts.data[:,framedata].reshape(self.ncol, self.nrow).T
            )
        ti = float(self.tx[framedata])
        self.t_mark.set_data(( [ti, ti], [0, 1] ))
        self._drawn_artists = [self.ecog_image, self.t_mark]

    def _init_draw(self):
        self.ecog_image.set_data(
            np.zeros((self.nrow, self.ncol))
            )
        ti = float(self.tx[0])
        self.t_mark.set_data(( [ti, ti], [0, 1] ))
        
    def save(self, filename, **kwargs):
        fps = 1000.0 / self._interval
        super(EcogAni, self).save(filename, fps=fps, **kwargs)

    def _make_movie(self, fname, fps, codec, frame_prefix, **kwargs):
        fname = osp.splitext(fname)[0]
        ## r = subprocess.call(
        ##     ['ffmpeg', '-y', '-i', '%s%%04d.png'%frame_prefix,
        ##      '-an', '-vcodec', 'libx264',
        ##      '-x264opts', 'qp=1:bframes=1:fps=%1.2f'%fps, '%s.mp4'%fname]
        ##      )
        r = subprocess.call(
            ['ffmpeg', '-f', 'image2', '-r', str(fps), 
             '-i', '%s%%04d.png'%frame_prefix, 
             '-sameq', '%s.mp4'%fname, '-pass', '2']
             )
    
        
        

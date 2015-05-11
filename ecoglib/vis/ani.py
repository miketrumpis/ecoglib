import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as pp
import matplotlib.animation as _animation
import os
import ecoglib.vis.plot_modules as pm
import time

def write_frames(
        frames, fname='', title='Array Movie', fps=5, 
        quicktime=False, axis_toggle='on', **imshow_kw
        ):
    # most simple frame writer -- no tricks
    f = pp.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(frames[0], **imshow_kw)
    ax.axis('image')
    ax.axis(axis_toggle)
    def _step_time(num, frames, frame_im):
        frame_im.set_data(frames[num])
        return (frame_im,)
    func = lambda x: _step_time(x, frames, im)
    write_anim(
        fname, f, func, frames.shape[0], fps=fps, title=title,
        quicktime=quicktime
        )
        
def write_anim(
        fname, fig, func, n_frame,
        title='Array Movie', fps=5, quicktime=False
        ):

    FFMpegWriter = _animation.writers['ffmpeg']
    metadata = dict(title=title, artist='ecoglib')
    writer = FFMpegWriter(
        fps=fps, metadata=metadata, codec='libx264'
        )
    if quicktime:
        # do arguments that are quicktime compatible
        extra_args = ['-pix_fmt', 'yuv420p', '-qp', '1']
        # yuv420p looks a bit crappy, but upping the res helps
        #dpi = 300
        dpi = fig.dpi
    else:
        # yuv422p seems pretty good
        extra_args = ['-pix_fmt', 'yuv422p', '-qp', '0']
        dpi = fig.dpi
    writer.extra_args = extra_args
    fname = fname.split('.mp4')[0]
    with writer.saving(fig, fname+'.mp4', dpi):
        for n in xrange(n_frame):
            func(n)
            ## pp.draw()
            writer.grab_frame()

def dynamic_frames_and_series(
        frames, series, 
        tx=None, frame_times=None,
        xlabel='Epoch (s)', ylabel='V', 
        stack_traces=True, interp=1,
        imshow_kw={}, line_props={},
        title='Array Movie', vertical=True
        ):
    # returns a function that can be used to step through
    # figure frames
    if vertical:
        fig = pp.figure(figsize=(5, 10))
        frame_ax = fig.add_subplot(211)
        trace_ax = fig.add_subplot(212)
    else:
        fig = pp.figure(figsize=(8, 8))
        frame_ax = fig.add_subplot(121)
        trace_ax = fig.add_subplot(122)
    
    if tx is None:
        tx = np.arange(len(series))

    if interp > 1:
        n = len(tx)
        tx_plot = np.linspace(tx[0], tx[-1], (n-1)*interp)
        ifun = interp1d(tx, series, kind='cubic', axis=0)
        series = ifun(tx_plot)
    else:
        interp = max(interp, 1)
        tx_plot = tx
    if series.ndim == 2 and stack_traces:
        ptp = np.median(series.ptp(axis=0))
        series = series + np.arange(series.shape[1])*ptp
    ## Set up timeseries trace(s)
    ylim = line_props.pop('ylim', ())
    if not ylim:
        ylim = (series.min(), series.max())
    trace_ax.plot(tx_plot, series, **line_props)
    trace_ax.set_xlabel(xlabel)
    trace_ax.set_ylabel(ylabel)
    trace_ax.set_ylim(ylim)
    time_mark = trace_ax.axvline(x=tx[0], color='r', ls='-')
    
    ## Set up array frames
    f_img = frame_ax.imshow(frames[0], **imshow_kw)
    frame_ax.axis('image'); #frame_ax.axis('off')
    frame_ax.set_title(title, fontsize=18)
    
    def _step_time(num, tx, frames, frame_img, tm, f_idx=None):
        # frame index array f_idx encodes possible offsets and skips
        # of the frame times with respect to the time axis
        if f_idx is None:
            x = tx[num]
        else:
            x = tx[ f_idx[num] ]
        tm.set_data(( [x, x], [0, 1] ))
        frame_img.set_data(frames[num])
        return (frame_img, tm)

    func = lambda x: _step_time(
        x, tx, frames, f_img, time_mark, f_idx=frame_times
        )
    fig.tight_layout()
    return fig, func
        
            
def animate_frames_and_series(
        frames, series, **kwargs
        ):

    fps = kwargs.pop('fps', 5)
    fig, func = dynamic_frames_and_series(frames, series, **kwargs)
    # blit don't seem to work
    ani = _animation.FuncAnimation(
        fig, func, frames=frames.shape[0],
        interval=1000.0/fps, blit=False
        )
    return ani

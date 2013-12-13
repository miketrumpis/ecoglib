import numpy as np
import matplotlib.pyplot as pp
import matplotlib.animation as _animation
import os
import ecoglib.vis.plot_modules as pm
import time

def write_frames(
        frames, fname='', title='Array Movie', fps=5, 
        quicktime=False, **imshow_kw
        ):
    # most simple frame writer -- no tricks
    f = pp.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(frames[0], **imshow_kw)
    ax.axis('image')
    def _step_time(num, frames, frame_im):
        frame_im.set_data(frames[num])
        print np.linalg.norm(frame_im._A.ravel())
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
        dpi = 300
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
        frames, series, tx=None, title='Array Movie',
        xlabel='Epoch', ylabel='$\mu V$',
        imshow_kw={}, line_props={}
        ):
    # returns a function that can be used to step through
    # figure frames
    
    fig = pp.figure(figsize=(5, 10))
    frame_ax = fig.add_subplot(211)
    trace_ax = fig.add_subplot(212)
    
    if tx is None:
        tx = np.arange(len(stack_data))
    
    if series.ndim == 2:
        ptp = np.median(series.ptp(axis=0))
        series = series + np.arange(series.shape[1])*ptp
    ## Set up timeseries trace(s)
    ylim = (series.min(), series.max())
    trace_ax.plot(tx, series, **line_props)
    trace_ax.set_xlabel(xlabel)
    trace_ax.set_ylabel(ylabel)
    trace_ax.set_ylim(ylim)
    time_mark = trace_ax.axvline(x=tx[0], color='r', ls='-')
    
    ## Set up array frames
    f_img = frame_ax.imshow(frames[0], **imshow_kw)
    frame_ax.axis('image'); frame_ax.axis('off')
    frame_ax.set_title(title, fontsize=18)
    
    def _step_time(num, tx, frames, frame_img, tm):
        #tsp.time = tx[num]
        #tsp.draw_dynamic()
        tm.set_data(( [tx[num], tx[num]], [0, 1] ))
        frame_img.set_data(frames[num])
        return (frame_img, tm)

    func = lambda x: _step_time(x, tx, frames, f_img, time_mark)
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

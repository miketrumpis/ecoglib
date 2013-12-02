import numpy as np
import matplotlib.pyplot as pp
import matplotlib.animation as _animation
import os
import ecoglib.vis.plot_modules as pm
import time

def animate_frames(frames, movie_name='', fps=5, **imshow_kw):
    fig = pp.figure()
    ims = []
    for n, f in enumerate(frames):
        i = pp.imshow(f, **imshow_kw)
        ims.append([i])
    ani = _animation.ArtistAnimation(fig, ims)
    if movie_name:
        #pfx='-vcodec libx264 -vpre ultrafast -crf 15 -an'
        ani.save(
            movie_name+'.mp4', fps=fps,
            extra_args=['-vcodec', 'libx264']
            )
    return ani

def write_frames(frames, fname='', title='Array Movie', fps=5, **imshow_kw):
    FFMpegWriter = _animation.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib')
    #metadata = dict()
    ## writer = FFMpegWriter(
    ##     fps=fps, metadata=metadata,
    ##     extra_args=['-x264opts qp=1:bframes=1']
    ##     )
    writer = FFMpegWriter(
        fps=fps, metadata=metadata, codec='libx264'
        )
    fig = pp.figure()
    im = pp.imshow(np.empty_like(frames[0]), **imshow_kw)
    fig.axes[0].axis('image')
    fig.tight_layout()
    fig.axes[0].set_title(title)
    
    with writer.saving(fig, fname+'.mp4', 100):
        for frm in frames:
            im.set_data(frm)
            writer.grab_frame()
    
def write_anim(fname, fig, func, n_frame, title='Array Movie', fps=5):

    FFMpegWriter = _animation.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib')
    writer = FFMpegWriter(
        fps=fps, metadata=metadata, codec='libx264'
        )
    fname = fname.split('.mp4')[0]
    with writer.saving(fig, fname+'.mp4', 100):
        for n in xrange(n_frame):
            func(n)
            writer.grab_frame()


def dynamic_frames_and_series(
        frames, series, tx=None, title='Array Movie',
        imshow_kw={}, line_props={}
        ):
    
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
    trace_ax.set_xlabel('Session Interval')
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
        frames, series, tx=None, title='Array Movie',
        fps=5, imshow_kw={}, line_props={}
        ):
    
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
    ## tsp = pm.PagedTimeSeriesPlot(
    ##     tx, series, t0=tx[0], ylim=ylim,
    ##     figure=fig, axes=trace_ax,
    ##     line_props=line_props
    ##     )
    ## tsp.time = tx[0]
    trace_ax.plot(tx, series, **line_props)
    trace_ax.set_xlabel('Session Interval')
    #trace_ax.set_ylim(ylim)
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
        time.sleep(20/1000.)
        #return (tsp.time_mark, frame_img)
        return (frame_img, tm)
        
    
    ani = _animation.FuncAnimation(
        fig, _step_time, frames=len(tx),
        fargs=(tx, frames, f_img, time_mark),
        interval=1000.0/fps, blit=True
        )
    return ani

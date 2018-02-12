import numpy as np
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.animation as _animation
import os
import subprocess
from progressbar import ProgressBar, Percentage, Bar
import warnings

def _setup_animated_frames(
        frames, timer='ms', time=(), axis_toggle='on',
        figsize=None, colorbar=False, cbar_label='', figure_canvas=True,
        **imshow_kw
        ):
    if figure_canvas:
        from matplotlib.pyplot import figure
        f = figure(figsize=figsize)
    else:
        f = Figure(figsize=figsize)
    ax = f.add_subplot(111)
    im = ax.imshow(frames[0], **imshow_kw)
    ax.axis('image')
    ax.axis(axis_toggle)
    if isinstance(time, bool) and time:
        time = np.arange(len(frames))
        timer = 'samp'
    if len(time):
        ttl = ax.set_title('{0:.2f} {1}'.format(time[0], timer))
    else:
        ttl = None
    def _step_time(num, frames, frame_im):
        frame_im.set_data(frames[num])
        if ttl:
            ttl.set_text('{0:.2f} {1}'.format(time[num], timer))
            return (frame_im, ttl)
        return (frame_im,)
    func = lambda x: _step_time(x, frames, im)
    if colorbar:
        cb = f.colorbar(im, ax=ax, use_gridspec=True)
        cb.set_label(cbar_label)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f.tight_layout(pad=0.2)

    return f, func

def write_frames(
        frames, fname, fps=5, quicktime=False, qtdpi=300,
        title='Array movie', **anim_kwargs
        ):

    f, func = _setup_animated_frames(frames, figure_canvas=False, **anim_kwargs)
    write_anim(
        fname, f, func, frames.shape[0], fps=fps, title=title,
        quicktime=quicktime, qtdpi=qtdpi
        )

def animate_frames(frames, fps=5, blit=False, **anim_kwargs):
    f, func = _setup_animated_frames(frames, figure_canvas=True, **anim_kwargs)
    anim = _animation.FuncAnimation(
        f, func, frames=len(frames), interval=1000.0 / fps, blit=blit
        )
    return anim
    
def h264_encode_files(in_pattern, out, fps, quicktime=False):
    """Use ffmpeg to encode a list of files matching a pattern

    Parameters
    ----------
    in_pattern : str
        ffmpeg input pattern, e.g. "path/to/frame_%03d.png" for
        {frame_001.png, frame_002.png, ...}
    out : str
        Name of output video
    fps : int
        Frames per second of video
    quicktime : bool {False | True}
        Use encoding compatible with Quicktime playback (otherwise VNC
        seems to work)

    """

    if os.path.splitext(out)[1] != '.mp4':
        out = out + '.mp4'
    if quicktime:
        extra_args = ['-pix_fmt', 'yuv420p', '-x264opts', 'qp=1:bframes=1']
    else:
        extra_args = ['-pix_fmt', 'yuv422p', '-x264opts', 'qp=0:bframes=1']

    args = ['ffmpeg', '-y', '-r', fps, '-i', '%s'%in_pattern,
            '-an', '-vcodec', 'libx264']
    args = args + extra_args + [out]

    r = subprocess.call(args)
    
def write_anim(
        fname, fig, func, n_frame,
        title='Array Movie', fps=5, quicktime=False, qtdpi=300
        ):

    FFMpegWriter = _animation.writers['ffmpeg']
    metadata = dict(title=title, artist='ecoglib')
    writer = FFMpegWriter(
        fps=fps, metadata=metadata, codec='h264'
        )
    if quicktime:
        # do arguments that are quicktime compatible
        extra_args = ['-pix_fmt', 'yuv420p', '-qp', '1']
        # yuv420p looks a bit crappy, but upping the res helps
        dpi = qtdpi
    else:
        # yuv422p seems pretty good
        extra_args = ['-pix_fmt', 'yuv422p', '-qp', '0']
        dpi = fig.dpi
    writer.extra_args = extra_args
    fname = fname.split('.mp4')[0]
    with writer.saving(fig, fname+'.mp4', dpi):
        print 'Writing {0} frames'.format(n_frame)
        pbar = ProgressBar(
            widgets=[Percentage(), Bar()], maxval=n_frame
            ).start()
        for n in xrange(n_frame):
            func(n)
            writer.grab_frame()
            pbar.update(n)
        pbar.finish()

def dynamic_frames_and_series(
        frames, series, timer='ms',
        tx=None, frame_times=None,
        xlabel='Epoch (s)', ylabel='V', 
        stack_traces=True, interp=1,
        imshow_kw={}, line_props={},
        title='Array Movie', vertical=True,
        image_sz=0.5, figsize=(), pyplot=True
        ):
    # returns a function that can be used to step through
    # figure frames
    if pyplot:
        import matplotlib.pyplot as pp
        fig_fn = pp.figure
    else:
        fig_fn = Figure
        
    image_sz = int( 100 * image_sz )
    trace_sz = 100 - image_sz
    if vertical:
        figsize = figsize or (5, 10)
        fig = fig_fn(figsize=figsize)        
        s = GridSpec(100, 1).new_subplotspec((0, 0), rowspan=image_sz)
        frame_ax = fig.add_subplot(s)
        s = GridSpec(100, 1).new_subplotspec((image_sz, 0), rowspan=trace_sz)
        trace_ax = fig.add_subplot(s)
    else:
        figsize = figsize or (8, 8)
        fig = fig_fn(figsize=figsize)
        s = GridSpec(1, 100).new_subplotspec((0, 0), colspan=image_sz)
        frame_ax = fig.add_subplot(s)
        s = GridSpec(1, 100).new_subplotspec((0, image_sz), colspan=trace_sz)
        trace_ax = fig.add_subplot(s)
    
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
    trace_ax.set_xlim(tx[0], tx[-1])
    time_mark = trace_ax.axvline(x=tx[0], color='r', ls='-')
    if timer:
        ttl = frame_ax.set_title('{0:.2f} {1}'.format(tx[0], timer))
    else:
        if title:
            frame_ax.set_title(title, fontsize=18)
        ttl = None
    
    ## Set up array frames
    f_img = frame_ax.imshow(frames[0], **imshow_kw)
    frame_ax.axis('image'); #frame_ax.axis('off')
    
    def _step_time(num, tx, frames, frame_img, tm, f_idx=None):
        # frame index array f_idx encodes possible offsets and skips
        # of the frame times with respect to the time axis
        if f_idx is None:
            x = tx[num]
        else:
            x = tx[ f_idx[num] ]
        tm.set_data(( [x, x], [0, 1] ))
        frame_img.set_data(frames[num])
        if not ttl:
            return (frame_img, tm)
        ttl.set_text('{0:.2f} {1}'.format(x, timer))
        return (frame_img, tm, ttl)

    func = lambda x: _step_time(
        x, tx, frames, f_img, time_mark, f_idx=frame_times
        )
    if fig.canvas is not None:
        fig.tight_layout()
    return fig, func
        
            
def animate_frames_and_series(
        frames, series, blit=False, **kwargs
        ):

    fps = kwargs.pop('fps', 5)
    fig, func = dynamic_frames_and_series(frames, series, **kwargs)
    # blit don't seem to work
    ani = _animation.FuncAnimation(
        fig, func, frames=frames.shape[0],
        interval=1000.0/fps, blit=blit
        )
    return ani

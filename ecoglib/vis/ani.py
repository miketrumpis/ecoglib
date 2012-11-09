import numpy as np
import matplotlib.pyplot as pp
import matplotlib.animation as _animation

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

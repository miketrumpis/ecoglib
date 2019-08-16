
## if __name__=='__main__':
##     print 'shutting down splitting'
##     import sandbox.array_split
##     sandbox.array_split.SPLIT_DISABLED = True
import os
import numpy as np
import sandbox.array_split as arsp


# inplace mod
def f(x):
    x *= 2
    return

# returning new array
def g(x):
    y = x * 2
    return y

# returning two arrays
def h(x):
    y = x / 2
    b = y > 3
    return y, b

# one array split, another array shared broadcast in this case
def k(x, y):
    z = x + y[:] # quirk of my shared memory access
    return z

# method calls another method
def m(x):
    f(x)
    return k(x, g(x))

def test(n_jobs=-1):
    import ecogana.anacode.ep_scoring as ep_scoring
    s = np.random.randn(60, 13, 30)
    b = np.random.randn(60, 390)

    s_fun = arsp.split_at(
        split_arg=0, n_jobs=n_jobs, splice_at=(0,1)
        )(ep_scoring.active_sites)

    ## m1, scr1 = ep_scoring.active_sites(b, s)
    ## m2, scr2 = s_fun(b, s)

    m1, scr1 = ep_scoring.active_sites(s, b)
    m2, scr2 = s_fun(s, b)

    return (m1, m2), (scr1, scr2)

if __name__ == '__main__':
    #pass
    x = np.tile( np.arange(6), (30, 1) )
    sx = arsp.shared_copy(x)
    
    sf = arsp.split_at()(f)
    sg = arsp.split_at()(g)
    sh = arsp.split_at(splice_at=(0,1))(h)
    

    print('inplace:')
    f(x)
    print(x)
    sf(sx)
    print('----------------------------------------------------------------')
    print(sx)
    assert np.all(x==sx)

    print('simple split:')
    y1 = g(x)
    print(y1)
    print('----------------------------------------------------------------')
    y2 = sg(x)
    print(y2)
    assert np.all(y1==y2)

    print('returning two arrays')
    y1, b1 = h(x)
    print(y1)
    print(b1)
    print('----------------------------------------------------------------')
    y2, b2 = sh(x)
    print(y2)
    print(b2)
    assert np.all(y2==y1)
    assert np.all(b1==b2)
    
    print('sharing one argument')
    sk = arsp.split_at(shared_args=(1,))(k)
    y = np.arange(x.shape[-1])
    z1 = k(x, y)
    z2 = sk(x, y)
    print(z1)
    print('----------------------------------------------------------------')
    print(z2)
    assert np.all(z1==z2)

    print('nested functions')
    sm = arsp.split_at()(m)
    y1 = m(x)
    y2 = sm(x)
    print(y1)
    print('----------------------------------------------------------------')
    print(y2)
    assert np.all(y1*2 == y2)

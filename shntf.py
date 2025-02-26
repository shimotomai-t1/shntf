
import click
import numpy
import scipy
import matplotlib.pyplot as plt
import pandas


@click.group()
def cl():
    pass

@cl.command()
def eval2():
    n = 3
    ai = 3
    bi = 41
    ao = numpy.random.rand(ai, n)
    bo = numpy.random.rand(bi, n)
    print(ao)
    print(bo)
    x = numpy.einsum('ih,jh->ij', ao, bo)
    df = pandas.DataFrame(index=numpy.arange(1,10), columns=['rank', 'error'])
    for i in range(1,20):
        ax, bx, info = ntf2(x, i, iter=5)
        last = info.index[-1]
        df.loc[i, ['rank','error', 'LL', 'AIC']] = [i, info.loc[last,'error'], info.loc[last,'likelihood'], info.loc[last,'aic']]
    #df.index = df['rank']
    print('result')
    print(ax)
    print(bx)
    print(df)
    #df.drop('rank')
    df.plot()
    #plt.figure()
    #plt.plot(ax)
    #plt.figure()
    #plt.plot(bx)
    plt.show()
    return

@cl.command()
def eval3():
    n = 3
    ai = 3
    bi = 13
    ci = 37
    uo = numpy.random.randint(0, 10, ai*n).reshape(ai, n)
    vo = numpy.random.randint(0, 10, bi*n).reshape(bi, n)
    wo = numpy.random.randint(0, 10, ci*n).reshape(ci, n)
    print(uo)
    print(vo)
    print(wo)
    uovo = numpy.dot(uo, vo.T)
    print(uovo)
    uovowo = numpy.einsum('il,jl,kl->ijk', uo, vo, wo)
    print(uovowo)
    r1,r2,r3 = ntf3(uovowo, 3, iter=8)
    print('r1\n',r1)
    print('r2\n',r2)
    print('r3\n',r3)
    #print('loglikelihood', loglikelihood(uovowo, r1, r2, r3))
    print('loglikelihoodGamma', loglikelihoodGamma(uovowo, r1, r2, r3))
    print('aic', aic(uovowo, r1, r2, r3))
    plt.figure()
    plt.plot(uo)
    plt.figure()
    plt.plot(r1)
    plt.figure()
    plt.plot(r2)
    plt.figure()
    plt.plot(wo)
    plt.title('wo')
    plt.figure()
    plt.plot(r3)
    plt.title('r3')
    plt.show()

    return

def loglikelihood(m, r1, r2, r3):
    r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
    return numpy.sum(m*numpy.log(r123)-r123)

def loglikelihoodGamma(m, r1, r2, r3):
    """ Gamma Distribution Log Likelihood
    m: numpy.ndarray
    r1: numpy.ndarray
    r2: numpy.ndarray
    r3: numpy.ndarray
    """
    r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
    #lp = numpy.sum(numpy.log(r123)*m-(r123))
    k = 5
    mask = (m!=0) * (r123!=0)
    m0 = m[mask]
    r0 = r123[mask]
    lp = numpy.sum((k-1)*numpy.log(m0)-m0/r0 - numpy.log(scipy.special.gamma(m0)) - k*numpy.log(r0))
    return lp


def aic(m, r1, r2, r3):
    return -2*loglikelihood(m, r1, r2, r3)+2*(r1.size+r2.size+r3.size)

def ntf2(x:numpy.ndarray, n:int, iter:int=4) -> tuple:
    """
    x: 目的行列
    n: 分解数,rank
    iter: 繰り返し回数
    """
    a = numpy.random.rand(x.shape[0], n)
    b = numpy.random.rand(x.shape[1], n)
    xhat = numpy.einsum('ih,jh->ij', a, b)
    er = ((x - xhat)*(x - xhat)).sum()
    dfer = pandas.DataFrame(index=numpy.arange(iter+1), columns=['error'])
    parnum = x.shape[0]*n+ x.shape[1]*n
    dfer.loc[0,'error'] = er
    for i in range(1, iter+1):
        xhat = numpy.einsum('ih,jh->ij', a, b)
        lower = numpy.einsum('ij,jh->ih', xhat, b)
        upper = numpy.einsum('ij,jh->ih', x,    b)
        a = a * (upper/lower)
        xhat = numpy.einsum('ih,jh->ij', a, b)
        lower = numpy.einsum('ij,ih->jh', xhat, a)
        upper = numpy.einsum('ij,ih->jh', x,    a)
        b = b * (upper/lower)
        xhat = numpy.einsum('ih,jh->ij', a, b)
        er = ((x - xhat)*(x - xhat))
        s = er.mean()
        ll = - er.sum()/(2*s) - numpy.log(2*numpy.pi*s)
        aic = -2*ll + 2*parnum*0.003
        dfer.loc[i, ['error', 'variance', 'likelihood', 'aic']] = [er.sum(), s, ll, aic]
    print(dfer)
    return a, b, dfer


def ntf3(m, n, iter=4):
    r1 = numpy.zeros((m.shape[0],n))
    r2 = numpy.zeros((m.shape[1],n))
    r3 = numpy.zeros((m.shape[2],n))
    r1 = numpy.random.rand(m.shape[0],n)
    r2 = numpy.random.rand(m.shape[1],n)
    r3 = numpy.random.rand(m.shape[2],n)
    #r1[:,0] = m.mean(axis=(1,2))
    #r2[:,0] = m.mean(axis=(0,2))
    #r3[:,0] = m.mean(axis=(0,1))
    rr = []
    for i in range(iter):
        r123  = numpy.einsum('il,jl,kl->ijk', r1,   r2, r3)
        lower = numpy.einsum('ijk,jh,kh->ih', r123, r2, r3)
        upper = numpy.einsum('ijk,jh,kh->ih', m,    r2, r3)
        r1 = r1*(upper/lower)
        #print('ntf3:r1\n',r1)
        r123  = numpy.einsum('il,jl,kl->ijk', r1,   r2, r3)
        lower = numpy.einsum('ijk,ih,kh->jh', r123, r1, r3)
        upper = numpy.einsum('ijk,ih,kh->jh', m,    r1, r3)
        r2 = r2*(upper/lower)
        #print('ntf3:r2\n',r2)
        r123  = numpy.einsum('il,jl,kl->ijk', r1,   r2, r3)
        lower = numpy.einsum('ijk,ih,jh->kh', r123, r1, r2)
        upper = numpy.einsum('ijk,ih,jh->kh', m,    r1, r2)
        r3 = r3*(upper/lower)
        #print('ntf3:r3\n',r3)
        print(i)
        print(pandas.DataFrame(r1).corr())
        print(pandas.DataFrame(r2).corr())
        print(pandas.DataFrame(r3).corr())
        rr += [{'cor1':pandas.DataFrame(r1).corr(), 'cor3':pandas.DataFrame(r3).corr()}]
    pandas.DataFrame([r['cor3'].values.flatten() for r in rr ]).to_csv('cor.csv')
    return r1,r2,r3

# u_{r,k}=u_{r, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}v_{s,k}w_{t,k}}{\sum_s\sum_tv_{s,k}w_{t,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}
# v_{s,k}=v_{s, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}u_{r,k}w_{t,k}}{\sum_r\sum_tu_{r,k}w_{t,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}
# w_{t,k}=w_{t, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}v_{s,k}u_{r,k}}{\sum_s\sum_tv_{s,k}u_{r,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}

if __name__ == '__main__':
    cl()

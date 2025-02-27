
import click
import numpy
import scipy
import matplotlib.pyplot as plt
import pandas

@click.group()
def cl():
    pass

@cl.command()
@click.option('--rank', '-r', type=int, default=3, help='分解数を指定します。')
@click.option('--alength', '-a', type=int, default=3, help='行列Aの行数を指定します。')
@click.option('--blength', '-b', type=int, default=5, help='行列Bの行数を指定します。')
def eval2(rank:int, alength:int, blength:int):
    n = rank
    ai = alength
    bi = blength
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
    df.to_csv('ntf2_rank_error.csv', index=False)
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
    plt.savefig('ntf2_rank_vs_error.png')
    plt.savefig('ntf2_rank_vs_error.svg')
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
    r1,r2,r3, info = ntf3(uovowo, 3)
    print('r1\n',r1)
    print('r2\n',r2)
    print('r3\n',r3)
    #print('loglikelihood', loglikelihood(uovowo, r1, r2, r3))
    print('loglikelihoodGamma', loglikelihoodGamma(uovowo, r1, r2, r3))
    print('aic', aic(uovowo, r1, r2, r3))
    error2 = []
    samplen = 10
    maxrank = 12
    nvec = numpy.arange(1,maxrank+1)
    dfer = pandas.DataFrame(index=numpy.arange(samplen*maxrank), columns=['n','trial', 'MSE'])
    ind = 0
    for i in nvec:
        for j in range(samplen):
            uo = numpy.random.randint(0, 10, ai*n).reshape(ai, n)
            vo = numpy.random.randint(0, 10, bi*n).reshape(bi, n)
            wo = numpy.random.randint(0, 10, ci*n).reshape(ci, n)
            #uovowo = numpy.einsum('il,jl,kl->ijk', uo, vo, wo)
            r1,r2,r3, info = ntf3(uovowo, i)
            dfer.loc[ind, ['n','trial','MSE']] = [i, j, info['error2'][-1]]
            ind += 1
        error2 += [info['error2'][-1]]
    dfer.to_csv('ntf3_rank_error2.csv', index=False)
    print(dfer)
    dfer.boxplot(by='n', column='MSE')
    plt.savefig('ntf3_rank_error2_boxplot.png')
    plt.savefig('ntf3_rank_error2_boxplot.svg')
    plt.figure()
    plt.plot(error2)
    plt.grid()
    plt.yscale('log')
    plt.savefig('ntf3_rank_error2.png')
    plt.savefig('ntf3_rank_error2.svg')
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

def ntf2(x:numpy.ndarray, n:int, iter:int=2048, error_break:float=1e-4) -> tuple:
    """
    x: 目的行列
    n: 分解数,rank
    iter: 繰り返し回数
    """
    med = numpy.median(x)
    x = x/med
    a = numpy.random.rand(x.shape[0], n)
    b = numpy.random.rand(x.shape[1], n)
    xhat = numpy.einsum('ih,jh->ij', a, b)
    er = ((x - xhat)*(x - xhat)).sum()
    dfer = pandas.DataFrame(index=numpy.arange(iter+1), columns=['error'])
    parnum = numpy.sum(x.shape)*n
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
        if s < error_break:
            break
    print(dfer)
    b = b*med
    return a, b, dfer


def ntf3(x:numpy.ndarray, n:int, iter:int=2048, error_break:float=1e-4) -> tuple:
    """３階テンソル分解
    x: 目的行列
    n: 分解数,rank
    iter: 繰り返し回数
    error_break: 誤差がこの値以下になったら終了
    """
    med = numpy.median(x)
    x = x/med
    a = numpy.random.rand(x.shape[0], n)
    b = numpy.random.rand(x.shape[1], n)
    c = numpy.random.rand(x.shape[2], n)
    xhat = numpy.einsum('ih,jh,kh->ijk', a, b,c)
    er = ((x - xhat)*(x - xhat)).sum()
    dfer = pandas.DataFrame(index=numpy.arange(iter+1), columns=['error'])
    parnum = numpy.sum(x.shape)*n
    dfer.loc[0,'error'] = er
    for i in range(1, iter+1):
        xhat  = numpy.einsum('ih,jh,kh->ijk', a, b,c)
        lower = numpy.einsum('ijk,jh,kh->ih', xhat, b, c)
        upper = numpy.einsum('ijk,jh,kh->ih', x,    b, c)
        a = a * (upper/lower)
        xhat  = numpy.einsum('ih,jh,kh->ijk', a, b, c)
        lower = numpy.einsum('ijk,ih,kh->jh', xhat, a, c)
        upper = numpy.einsum('ijk,ih,kh->jh', x,    a, c)
        b = b * (upper/lower)
        xhat  = numpy.einsum('ih,jh,kh->ijk', a, b, c)
        lower = numpy.einsum('ijk,ih,jh->kh', xhat, a, b)
        upper = numpy.einsum('ijk,ih,jh->kh', x,    a, b)
        c = c * (upper/lower)
        xhat  = numpy.einsum('ih,jh,kh->ijk', a, b, c)
        er = ((x - xhat)*(x - xhat))
        s = er.mean()
        ll = - er.sum()/(2*s) - numpy.log(2*numpy.pi*s)
        aic = -2*ll + 2*parnum*0.003
        dfer.loc[i, ['error', 'variance', 'likelihood', 'aic']] = [er.sum(), s, ll, aic]
        if s < error_break:
            break
    print(dfer)
    c = c*med
    return a, b, c, dfer

def nntf(x:numpy.ndarray, n:int, iter:int=2048, error_break:float=1e-4) -> tuple:
    """n階テンソル分解 26次元まで
    x: 目的行列
    n: 分解数,rank
    iter: 繰り返し回数
    error_break: 誤差がこの値以下になったら終了
    """
    med = numpy.median(x)
    x = x/med
    dim = len(x.shape)
    index_list = [chr(i) for i in range(ord('a'),ord('a')+dim)]
    vec = [numpy.random.rand(x.shape[i], n) for i in range(dim)]
    ind = ','.join([f'{c}z' for c in index_list])
    indr= ind.replace('z','').replace(',','')
    statement = f'{ind}->{indr}'
    print(statement)
    xhat = numpy.einsum(statement, *vec)
    er = ((x - xhat)*(x - xhat)).mean()
    dfer = pandas.DataFrame(index=numpy.arange(iter+1), columns=['error'])
    parnum = numpy.sum(x.shape)*n
    dfer.loc[0,'error'] = er
    for i in range(1, iter+1):
        for j in range(dim):
            c = index_list[j]
            xhat = numpy.einsum(statement, *vec)
            exlist = [f'{ci}z' for ci in index_list if ci != c]
            ulstatement = (''.join(index_list))+','+ (','.join(exlist)) + '->' + c + 'z'
            exvec = [vec[k] for k in range(dim) if k != j]
            lower = numpy.einsum(ulstatement, xhat, *exvec)
            upper = numpy.einsum(ulstatement, x, *exvec)
            vec[j] = vec[j] * (upper/lower)
        xhat = numpy.einsum(statement, *vec)
        er = ((x - xhat)*(x - xhat))
        s = er.mean()
        ll = - er.sum()/(2*s) - numpy.log(2*numpy.pi*s)
        aic = -2*ll + 2*parnum*0.003
        dfer.loc[i, ['error', 'variance', 'likelihood', 'aic']] = [er.sum(), s, ll, aic]
    return vec, dfer

def ntf3old(m:numpy.ndarray, n:int, iter:int=4, error_break:float=1e-4) -> tuple:
    """
    m: 目的行列
    n: 分解数,rank
    iter: 繰り返し回数
    """
    med = numpy.median(m)
    m = m/med
    r1 = numpy.zeros((m.shape[0],n))
    r2 = numpy.zeros((m.shape[1],n))
    r3 = numpy.zeros((m.shape[2],n))
    r1 = numpy.random.rand(m.shape[0],n)
    r2 = numpy.random.rand(m.shape[1],n)
    r3 = numpy.random.rand(m.shape[2],n)
    #r1[:,0] = m.mean(axis=(1,2))
    #r2[:,0] = m.mean(axis=(0,2))
    #r3[:,0] = m.mean(axis=(0,1))
    error1 = []
    error2 = []
    for i in range(iter):
        r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
        error1 += [numpy.mean(numpy.abs(m-r123))]
        error2 += [numpy.mean(r123*r123)]
        print(f'ntf3:error1={error1[-1]} error2={error2[-1]}') 
        lower = numpy.einsum('sk,tk,rst->rk',r2,r3,r123)
        upper = numpy.einsum('rst,sk,tk->rk',m,r2,r3)
        r1 = r1*(upper/lower)
        print('ntf3:r1\n',r1)
        r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
        lower = numpy.einsum('rk,tk,rst->sk',r1,r3,r123)
        upper = numpy.einsum('rst,rk,tk->sk',m,r1,r3)
        r2 = r2*(upper/lower)
        print('ntf3:r2\n',r2)
        r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
        lower = numpy.einsum('rk,sk,rst->tk',r1,r2,r123)
        upper = numpy.einsum('rst,rk,sk->tk',m,r1,r2)
        r3 = r3*(upper/lower)
        print('ntf3:r3\n',r3)
    r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
    info = {'error1':error1, 'error2':error2, 'r123':r123}
    return r1,r2,r3, info

# u_{r,k}=u_{r, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}v_{s,k}w_{t,k}}{\sum_s\sum_tv_{s,k}w_{t,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}
# v_{s,k}=v_{s, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}u_{r,k}w_{t,k}}{\sum_r\sum_tu_{r,k}w_{t,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}
# w_{t,k}=w_{t, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}v_{s,k}u_{r,k}}{\sum_s\sum_tv_{s,k}u_{r,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}

if __name__ == '__main__':
    cl()

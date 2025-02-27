# Non-negative tensor factorization

## 使い方

```
pip install -r shntf/requirements.txt
```


### 例
ランダム行列で試行：
```
import numpy
import shntf
x = numpy.random.rand(3,4,5)
vec, info = shntf.nntf(x, 3)
# vec: list[numpy.ndarray]
# info: pandas.DataFrame
```
結果：
```
>>> info
           error  variance  likelihood         aic
0       0.739134       NaN         NaN         NaN
1     120.272243  0.334090 -180.741531  361.807062
2     113.970693  0.316585 -180.687714  361.699429
3     111.292356  0.309145 -180.663934  361.651867
4     109.712854  0.304758 -180.649640  361.623279
...          ...       ...         ...         ...
2044  102.091816  0.283588 -180.577646  361.479291
2045  102.091816  0.283588 -180.577646  361.479291
2046  102.091816  0.283588 -180.577646  361.479291
2047  102.091816  0.283588 -180.577646  361.479291
2048  102.091816  0.283588 -180.577646  361.479291
>>> vec
[array([[1.03602167, 0.84861983, 0.72514372],
       [0.38456956, 1.03549075, 0.65604731],
       [0.43539307, 0.64764285, 1.53985898]]), array([[2.20690291e+00, 6.62679407e-01, 4.08983694e-76],
       [7.06186257e-01, 1.31401157e-01, 1.20739414e+00],
       [6.28668276e-13, 1.25780934e+00, 6.79734550e-01],
       [5.11372501e-01, 5.64258585e-01, 9.13502838e-01]]), array([[9.29228937e-001, 1.38883289e-016, 3.89326111e-001],
       [6.26649249e-001, 1.15623247e+000, 1.23535530e-072],
       [3.46110334e-001, 1.53159563e+000, 3.90456845e-001],
       [1.18923643e+000, 2.54573120e-001, 5.37152112e-001],
       [4.68730077e-001, 1.06922253e-175, 1.20642205e+000]])]
>>> info
          error  variance  likelihood        aic
0      1.020678       NaN         NaN        NaN
1     24.699999  0.411667  -30.950336  62.116671
2     21.175136  0.352919  -30.796360  61.808720
3     19.275682  0.321261  -30.702377  61.620754
4     17.676645  0.294611  -30.615777  61.447554
...         ...       ...         ...        ...
2044   9.343006  0.155717  -29.978161  60.172321
2045   9.343006  0.155717  -29.978161  60.172321
2046   9.343006  0.155717  -29.978161  60.172321
2047   9.343006  0.155717  -29.978161  60.172321
2048   9.343006  0.155717  -29.978161  60.172321
```



## 2階テンソル分解
### ユークリッド距離ベース

ユークリッド距離で定義：

$$E=\sum_{ij}{(x_{ij}-\hat{x}_{ij})^2}$$

ただし $\hat{x}$ は次のように分解される。

$$\hat{x}_{ij} = \sum_{h}{a_{ih}b_{jh}}$$

xはデータ、a,bは分解ベクトルで、hがランク数の添え字。

更新式は

$$a_{ih} = a_{ih} \frac{\sum_{j}{x_{ij} b_{jh}}}{\sum_{j}{\hat{x}_{ij} b_{jh}}}$$

$$b_{jh}=b_{jh} \frac{\sum_{i}{x_{ij} a_{ih}}}{\sum_{i}{\hat{x}_{ij} a_{ih}}}$$

#### 途中式

Jensenの不等式を用いて補助関数 $\tilde{E}$ を定義。

$$E\le \tilde{E} = \sum_{ij}{\left(x_{ij}^2 - 2x_{ij}\sum_h{a_{ih}b_{jh}}+\sum_h{\frac{a_{ih}^2b_{jh}^2c_{kh}^2}{r_{ijh}}}\right)}$$

$$\sum_hr_{ijh} = 1$$

未定乗数法を使って、

$$F=\tilde{E} + \sum_{ij}{\lambda_{ij}(\sum_h{r_{ijh}}-1)}$$

Fをrで偏微分して、更に和が１という条件を使って、

$$r_{ijh} = \frac{a_{ih}b_{jh}}{\hat{x}_{ij}}$$

aで偏微分

$$\frac{\partial F}{\partial a_{ig}}=\sum_{ij}{-2x_{ij}b_{jg}+2\frac{a_{ig}b^2_{jg}}{r_{ijg}}}=0$$

$$a_{ih} = \frac{\sum_j{x_{ij}b_{jh}}}{\sum_j{\frac{b^2_{jh}}{r_{ijh}}}}$$

ここにrの式を代入すると上の更新式になる。


## 3階テンソル分解

### ユークリッド距離ベース

$$E=\sum_{ijk}{(x_{ijk}-\hat{x}_{ijk})^2}$$

ただし

$$\hat{x}_{rst} = \sum_{l}{a_{rl} b_{sl} c_{tl}}$$


分解されたベクトルは下の更新式で繰り返し計算される。

```math
a_{rk} = a_{rk}\displaystyle\frac{\sum_{st}{x_{rst}b_{sk}c_{tk}}}{\sum_{st}{\hat{x}_{rst}b_{sk}c_{tk}}}
```


```math
b_{sk}=b_{sk}\displaystyle\frac{\sum_{rt}x_{rst}a_{rk}c_{tk}}{\sum_{rt}{\hat{x}_{rst}}a_{rk}c_{tk}}
```

```math
c_{tk}=c_{tk}\displaystyle\frac{\sum_{rs}x_{rst}c_{tk}a_{rk}}{\sum_{rs}{\hat{x}_{rst}a_{rk}b_{sk}}}
```


## 4階テンソル分解
### ユークリッド距離ベース

$$E=\sum_{ijkl}{(x_{ijkl} - \hat{x}_{ijkl})^2}$$


```math
\hat{x}_{ijkl} = \sum_{h}{a_{ih} b_{jh} c_{kh} d_{lh}}
```

```math
a_{ih} = a_{ih} \frac{\sum_{jkl}{x_{ijkl} b_{jh} c_{kh} d_{lh}}}{\sum_{jkl}{\hat{x}_{ijkl} b_{jh} c_{kh} d_{lh}}}
```

```math
b_{jh} = b_{jh} \frac{\sum_{ikl}{x_{ijkl}a_{ih}c_{kh}d_{lh}}}{\sum_{ikl}{\hat{x}_{ijkl}a_{ih}c_{kh}d_{lh}}}
```

```math
c_{kh} = c_{kh} \frac{\sum_{ijl}{x_{ijkl}a_{ih}b_{jh}d_{lh}}}{\sum_{ijl}{\hat{x}_{ijkl}a_{ih}b_{jh}d_{lh}}}
```

```math
d_{lh} = d_{lh} \frac{\sum_{ijk}{x_{ijkl}a_{ih}b_{jh}c_{kh}}}{\sum_{ijk}{\hat{x}_{ijkl}a_{ih}b_{jh}c_{kh}}}
```



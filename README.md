# Non-negative tensor factorization


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

$$a_{rk} = a_{rk}\displaystyle\frac{\sum_{st}{x_{rst}b_{sk}c_{tk}}}{\sum_{st}{\hat{x}_{rst}b_{sk}c_{tk}}}$$

$$b_{sk}=b_{sk}\displaystyle\frac{\sum_{rt}x_{rst}a_{rk}c_{tk}}{\sum_{rt}{\hat{x}_{rst}}a_{rk}c_{tk}}$$

$$c_{tk}=c_{tk}\displaystyle\frac{\sum_{rs}x_{rst}c_{tk}a_{rk}}{\sum_{rs}{\hat{x}_{rst}a_{rk}b_{sk}}}$$


## 4階テンソル分解
### ユークリッド距離ベース

$$E=\sum_{ijkl}{(x_{ijkl} - \hat{x}_{ijkl})^2}$$

$$\hat{x}_{ijkl} = \sum_{h}{a_{ih} b_{jh} c_{kh} d_{lh}}$$

$$a_{ih} = a_{ih} \frac{\sum_{jkl}{x_{ijkl} b_{jh} c_{kh} d_{lh}}}{\sum_{jkl}{\hat{x}_{ijkl} b_{jh} c_{kh} d_{lh}}}$$

$$b_{jh} = b_{jh} \frac{\sum_{ikl}{x_{ijkl}a_{ih}c_{kh}d_{lh}}}{\sum_{ikl}{\hat{x}_{ijkl}a_{ih}c_{kh}d_{lh}}}$$

$$c_{kh} = c_{kh} \frac{\sum_{ijl}{x_{ijkl}a_{ih}b_{jh}d_{lh}}}{\sum_{ijl}{\hat{x}_{ijkl}a_{ih}b_{jh}d_{lh}}}$$

$$d_{lh} = d_{lh} \frac{\sum_{ijk}{x_{ijkl}a_{ih}b_{jh}c_{kh}}}{\sum_{ijk}{\hat{x}_{ijkl}a_{ih}b_{jh}c_{kh}}}$$



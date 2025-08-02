设$D_{\text{max}}$为待测海域的最大深度，则

$$D_{\text{max}} = D_0 + \frac{L_{es}}{2} \tan \alpha$$

其中$D_0$为待测海域中心处的海水深度，$L_{es}$为待测海域东西的宽度。  
第1条测线的坐标为  

$$x_1 = D_{\text{max}} \cdot \tan \frac{\theta}{2}$$  

由问题1知，在计算覆盖宽度时，需要知道测线上的海水深度  

$$D_1 = D_{\text{max}} - x_1 \cdot \tan \alpha$$  


设$d_1$为第2条测线与第1条测线之间的间距，测线深度与测线间距的关系式为  

$$D_1 - D_2 = d_1 \tan \alpha, \, D_2 = D_1 - d_1 \tan \alpha$$  

考虑到覆盖的重复率为$\eta$，由式(2)得到  

$$d_1 = (W_{12} + W_{21})(1 - \eta) \cos \alpha \tag{5}$$  

其中  

$$W_{12} = \frac{D_1 \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} - \alpha \right)}, \quad W_{21} = \frac{D_2 \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} + \alpha \right)} \tag{6}$$  

由式(5)和式(6)得到  

$$
\begin{align*} 
d_1 &= \left( W_{12} + \frac{D_2 \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} + \alpha \right)} \right) (1 - \eta) \cos \alpha \\ 
&= \left( W_{12} + \frac{(D_1 - d_1 \tan \alpha) \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} + \alpha \right)} \right) (1 - \eta) \cos \alpha \\ 
&= (W_{12} + W_{11})(1 - \eta) \cos \alpha - \frac{d_1 \tan \alpha \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} + \alpha \right)} (1 - \eta) \cos \alpha \\ 
&= (W_{12} + W_{11})(1 - \eta) \cos \alpha - \frac{d_1 \sin \alpha \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} + \alpha \right)} (1 - \eta) \\ 
\end{align*}
$$  


所以  

$$\left( 1 + \frac{\sin \alpha \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} + \alpha \right)} (1 - \eta) \right) d_1 = (W_{12} + W_{11})(1 - \eta) \cos \alpha$$  

即  

$$d_1 = \frac{(W_{12} + W_{11})(1 - \eta) \cos \alpha}{1 + \frac{\sin \alpha \sin \frac{\theta}{2}}{\cos \left( \frac{\theta}{2} + \alpha \right)} (1 - \eta)}$$  

再计算出条带的左、右线的坐标  

$$l_1 = x_1 - W_{11} \cdot \cos \alpha, \quad r_1 = x_1 + W_{12} \cdot \cos \alpha$$  


在计算第2条条带时，其坐标为  

$$x_2 = x_1 + d_1$$  

$x_2$处的海水深度为  

$$D_2 = D_1 - d_1 \tan \alpha$$  

再按上述推导过程计算出$d_2$、$x_3$等。  

第3条以后测线的坐标计算类推，可用迭代的方法，直到  

$$x_k + W_2 \cos \alpha > 4 \, (\text{海里})$$  

时，计算结束。  

最终的计算结果如表5所示，测线布设图如图14所示。

---

### 迭代公式推导

为了布设整个测区的测线，需要从第一条测线开始，依次计算后续测线的位置。这是一个迭代的过程。我们来推导计算第 $k+1$ 条测线位置的通用公式。

设第 $k$ 条测线的位置为 $x_k$，该点的水深为 $D_k$。我们需要计算与下一条测线（第 $k+1$ 条）之间的距离 $d_k$。

第 $k+1$ 条测线的位置和水深分别为：
$$x_{k+1} = x_k + d_k$$
$$D_{k+1} = D_k - d_k \tan \alpha$$

相邻两条测线（第 $k$ 条和第 $k+1$ 条）的间距 $d_k$ 由两者之间的覆盖宽度和重叠率 $\eta$ 决定：
$$d_k = (W_{k, \text{右}} + W_{k+1, \text{左}})(1 - \eta) \cos \alpha$$
其中 $W_{k, \text{右}}$ 是第 $k$ 条测线在水深 $D_k$ 处的右侧覆盖宽度，$W_{k+1, \text{左}}$ 是第 $k+1$ 条测线在水深 $D_{k+1}$ 处的左侧覆盖宽度。

根据问题一的模型，覆盖宽度的计算公式为：
$$W_{k, \text{右}} = \frac{D_k \sin(\theta/2)}{\cos(\theta/2 - \alpha)}$$
$$W_{k+1, \text{左}} = \frac{D_{k+1} \sin(\theta/2)}{\cos(\theta/2 + \alpha)}$$

将 $D_{k+1}$ 的表达式代入 $W_{k+1, \text{左}}$：
$$W_{k+1, \text{左}} = \frac{(D_k - d_k \tan \alpha) \sin(\theta/2)}{\cos(\theta/2 + \alpha)}$$

再将 $W_{k, \text{右}}$ 和 $W_{k+1, \text{左}}$ 的表达式代入 $d_k$ 的计算式中，得到一个关于 $d_k$ 的方程：
$$d_k = \left( \frac{D_k \sin(\theta/2)}{\cos(\theta/2 - \alpha)} + \frac{(D_k - d_k \tan \alpha) \sin(\theta/2)}{\cos(\theta/2 + \alpha)} \right) (1-\eta) \cos \alpha$$

为了方便求解，我们定义在水深 $D_k$ 处的左、右覆盖宽度：
$$W_{k1} = \frac{D_k \sin(\theta/2)}{\cos(\theta/2 + \alpha)} \quad (\text{左})$$
$$W_{k2} = \frac{D_k \sin(\theta/2)}{\cos(\theta/2 - \alpha)} \quad (\text{右})$$

代入后整理可得：
$$d_k = \frac{(W_{k1} + W_{k2})(1-\eta)\cos\alpha}{1 + \frac{(1-\eta)\sin\alpha\sin(\theta/2)}{\cos(\theta/2+\alpha)}}$$

这是一个迭代公式。从 $k=1$ 开始，已知 $x_1$ 和 $D_1$，我们可以通过此公式计算出 $d_1$。然后便可得到 $x_2$ 和 $D_2$。重复此过程，即可计算出所有的 $d_k, x_{k+1}, D_{k+1}$，直到测线覆盖整个区域。

在向西（水深增加）布线时，迭代公式中的符号需要相应调整：
$$x_{k+1} = x_k - d_k$$
$$D_{k+1} = D_k + d_k \tan \alpha$$
间距 $d_k$ 的计算公式形式保持不变。
# geo-bearing-direaction
用于计算路网路段的有向夹角以及八方向属性

## Background And Target
1. 给定中心点(定点,经纬度0点或市政府位置),有序经纬度串,算经纬度串的有向夹角
(与正北还是哪个方向夹角由算法看通用一般算跟哪条线的夹角).走向(横or竖).八方向属性(正北东北东...).<Br/>
用于路名生成,路段切分(确定路段切分的固定始发点).<Br/>
2. 给定中心点(路口),有序但可能反向的经纬度串,先根据距离判断是否反转,再算这三个属性.<Br/>
用于判断路是在路口的哪个方向<Br/>
ps: 在实际路网可能不需要中心点，因默认的有向夹角为正北向<Br/>
3. 根据计算得到的夹角来对应路段的方向属性<Br/>
4. 在osm路网中对路段进行测试

## Methods
1. 计算两个经纬度点方位bearing的公式为<Br/>
θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )<Br/>
其中，λ1 φ1分别是起点的经纬度，λ2 φ2分别是终点经纬度<Br/>
计算得到的方位还需要换算成角度<Br/>
2. 计算两个经纬度点距离的公式为<Br/>
首先利用haversine公式计算地球间两点的最短距离，<Br/>
a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)<Br/>
然后计算弧长，<Br/>
c = 2 ⋅ atan2( √a, √(1−a) )<Br/>
d = R ⋅ c<Br/>
其中，λ φ分别是起点的经纬度，R为地球半径<Br/>
在计算过程中可以用geopy包来计算<Br/>
3. 对一条路上的所有点组成的不同方位取加权平均<Br/>
假设由n个点组成，则有n-1条向量组成，则一条路的平均方位为：<Br/>
![](http://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D0%7D%5E%7Bn-1%7Dd_i%5Ctheta_i)
<Br/>
4. 将计算的方位换算成方向属性<Br/>
考虑方位换算的度数落于哪个度数范围区间内，并以相应的数字对以标识：<Br/>
{1:East, 2:Northeast, 3:North, 4:Northwest,
5:West, 6:Southwest, 7:South, 8:Southeast}


## Reference
1. 如何计算球面坐标距离、方位<Br/>
http://www.movable-type.co.uk/scripts/latlong.html<Br/>
2. geopy<Br/>
https://github.com/geopy/geopy<Br/>
3. gis <Br/>
http://www.nickeubank.com/gis-in-python/<Br/>
4. geopandas<Br/>
http://geopandas.org/<Br/>





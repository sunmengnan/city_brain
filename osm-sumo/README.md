# 调研osm，osm与sumo转换关系，以便达到简化路网的目的

## 调研内容：
## 1. 不同程度简化后的osm转化为sumo路网带来的差异
1） 取滨江区路网为例，由于osm由于node，way，relation组成，node定义了点，way定义了点之间的组成，relation
定义了node、way之间的组成关系，因relation占比比较大，为了比较不同的简化程度的影响，分别将原路网删除了全部relation部分，
和删除relation以及对应的node，删除relation以及对应的way<Br/>
2） 将这些osm文件转化为sumo格式的路网，然后比较转化后的文件差异和路网图<Br/>
3） 以简化程度为最低的为例（仅完全去除osm的relation部分，但没有去除relation中递归的node和way），路网对比如下：<Br/>
![转换后路网对比](http://git.tianrang-inc.com/mn.sun/osm-sumo/blob/master/osm_degree/net_difference.png)<Br/>
4） 根据路网对比图可以大致判断去除relation之后删除了区域外的所有节点和道路，以及区域内的一部分节点和路，但在区域内的删除逻辑还需要具体分析。

## 2. 简化的osm是否等价.way+node,way+node+relation是否等价(即在转换中是否以来relation做推断)，osm文件中relation对osm转化为sumo格式的影响.
1）将sumo netconvert source code中涉及到从osm导入的代码取出<Br/>
2）用正则表达将涉及到key value的处理取出<Br/>
3）在feature map中对比这些key value是否对实际路网有影响

## 调研结果
在找出的key中，restriction是转向限制，route和public_transport是公共交通相关的.这些tags暂时都不影响路网生成，目前可以先将osm中relation部分删除

## Reference
1) netconvert source：<Br/>
https://github.com/eclipse/sumo.git<Br/>
netconvert main函数在sumo/src/netconvert_main.cpp，<Br/>
相关头文件在sumo/src/netimport之类的目录下<Br/>
https://github.com/eclipse/sumo<Br/>
2) osm 的路网构建介绍，osm wiki<Br/>
https://wiki.openstreetmap.org/wiki/Elements<Br/>
深入理解map feature相关内容<Br/>
https://wiki.openstreetmap.org/wiki/Map_Features
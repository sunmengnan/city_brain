# 根据无序点确定简单多边形

## 根据无序多边形的点找出能组成简单多边形的有序点，从而找到中心点

### Algorithm:
1. 找到最左边点 p<Br/>
2. 找到最右边点 q<Br/>
3. 将pq之上的点分到a, pq之下的点分到b<Br/> 
4. 对a中的点按x轴升序<Br/>
5. 对b中的点按x轴降序<Br/>
6. 返回点p，a中的点，b中的点，q<Br/>

### Runtime:
1. 第1,2,3 步 O(n)<Br/>
2. 第4,5步 O(nlogn)<Br/>
3. 第6步 O(n)<Br/>
4. 总时间复杂度O(nlogn)<Br/>
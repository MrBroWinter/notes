


## 常用语法：
# 标题

## 加粗

**加粗** 

## 斜体

*斜体* 

## 删除线

~~删除线~~ 

## 有序列表
1. aa
	1. bb
2. cc
	1. dd
3. ee
4. ff

## 无序列表
* a
	* b
* c
	* d
* e

## 公式
$$E=mc^2$$


## 引用
> 我是zdongdong。
> 
> 我是**赵冬冬**

## 分割线
___

## 注释
<!-- 这行不显示-->

---
## 链接
[markdown学习链接](https://blog.csdn.net/afei__/article/details/80717153) 

[cmd markdown](https://www.zybuluo.com/mdeditor)
## 代码
``` python
from functools import reduce  
  
"""filter"""  
ns = range(1,10)  
#filter(condition, list)  
def func(x):  
    if x % 2 == 0:  
        return True  
    else:  
        return False  
print(*filter(func, ns))  
  
print(*filter(lambda x: x % 2 == 0, ns))    #筛选偶数  
  
ns_1 = [1,2,3,"a", "b", "c", 2.8]  
print(*filter(lambda x: isinstance(x,int), ns_1))   # 筛选整形  
  
"""reduce"""  
# 筛选出100以内所有质数的平方根和  
import math  
N = 100  
print(reduce(lambda x,y: x+y, map(lambda d: math.sqrt(d), filter(lambda p : 0 not in list(map(lambda q: p % q, range(2, int(math.sqrt(p) + 1)))), range(1, 100)))))
```
## 任务列表
- [ ] 入门  <!-- 快捷键Ctrl + Enter-->
- [ ] 进阶

## 表格
| 项目        | 价格   |  数量  |
| --------   | -----  | ----  |
| 计算机     | \$1600 |   5     |
| 手机        |   \$12   |   12   |
| 管线        |    \$1    |  234  |


| xiangmu | jiage | shuliang |
| --- | --- | --- |
| jisuan | \$20   | 100 |




## 高阶语法




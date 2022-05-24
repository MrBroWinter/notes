# filter
**filter**: 通过对函数中返回的bool值确定对iter筛选后的结果
```python
ns = range(1,10)  
res_1 = filter(lambda x: x % 2 == 0, ns)  
print("res_1: ", *res_1)    # '*'有解包的作用
  
def func(x):  
    if x % 2 == 0:  
        return True  
    else:  
        return False  
  
res_2 = filter(func, ns)  
print("res_2: ", *res_2)

```

> res_1:  2 4 6 8 
> 
> res_2:  2 4 6 8 

# reduce
**reduce**:对参数序列中元素进行累积
```python
"""筛选出list中的整形求和"""
from functools import reduce
ns = [1,2,3,4,"a", "b", 2.8]
res = reduce(lambda x, y: x + y, filter(lambda n: isinstance(n, int), ns))
print("res:", res)

```
> 10




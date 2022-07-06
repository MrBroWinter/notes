#  [Python进阶链接](https://tairraos.github.io/IntermediatePython/)  
	
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

# 迭代器
```python
"""生成器"""  
def fibon(n):  
   a = b = 1  
   for i in range(n):  
       yield a  
       a, b = b, a + b  
N = 5  
# 1  
gen = fibon(N)  
for i in range(N):  
    print(next(gen))  
print("\n")
# 2.  
for x in fibon(N):  
    print(x)  
```

>1
1
2
3
5

>
1
1
2
3
5

```python
"""iter将可迭代对象返回一个迭代器对象"""  
my_string = "Yasoob"  
my_iter = iter(my_string)  
for i in range(len(my_string)):  
    print(next(my_iter))
```
>
Y
a
s
o
o
b


# set
```python
# 筛选出列表中重复的元素
some_list = ['a', 'b', 'c', 'b', 'd', 'm', 'n', 'n']  
print(set([x for x in some_list if some_list.count(x) > 1]))
```
> {'n', 'b'}

## 交集(intersection)&差集(difference)
```python
valid = set(['yellow', 'red', 'blue', 'green', 'black'])  
input_set = set(['red', 'brown'])  
print(valid.intersection(input_set))
print(valid.difference(input_set))
```

> {'red'}
> 
> {'green', 'yellow', 'blue', 'black'}

# 装饰器
```python
from functools import wraps  
def decorator(a_func):  
    @wraps(a_func)  # 修改函数__name__的参数  
    def wrapTheFunction():  
        print("I am doing some boring work before executing a_func()")  
        a_func()  
        print("I am doing some boring work after executing a_func()")  
    return wrapTheFunction  
  
@decorator  
def a_function_requiring_decoration(): # 将本函数当做参数传入到decorator函数中并实现decorator的功能  
    """Hey you! Decorate me!"""  
    print("I am the function which needs some decoration to "  
          "remove my foul smell")  
a_function_requiring_decoration()  
print('\n')  
print(a_function_requiring_decoration.__name__)
```

> I am doing some boring work before executing a_func()
I am the function which needs some decoration to remove my foul smell
I am doing some boring work after executing a_func()

> a_function_requiring_decoration


# collections

## defaultdict(dict有的它都有)
* **defaultdict**不需要检查key是否存在
```python
"""默认value类型为list"""
from collections import defaultdict 
colours = (  
    ('Yasoob', 'Yellow'),  
    ('Ali', 'Blue'),  
    ('Arham', 'Green'),  
    ('Ali', 'Black'),  
    ('Yasoob', 'Red'),  
    ('Ahmed', 'Silver'),  
)  
favourite_colours = defaultdict(list)  
for name, colour in colours:  
    favourite_colours[name].append(colour)  
print(favourite_colours)
```
> defaultdict(<class 'list'>, {'Yasoob': ['Yellow', 'Red'], 'Ali': ['Blue', 'Black'], 'Arham': ['Green'], 'Ahmed': ['Silver']})

```python
from collections import defaultdict
bag = ['apple', 'orange', 'cherry', 'apple','apple', 'cherry', 'blueberry']

# 计数方法1
count_dict = defaultdict(int)
for sub in bag:
	count_dict[sub] += 1
print(count_dict)

# 计数方法2
count_dict2 = defaultdict(int)
for sub in set(bag):
	count_dict2[sub] = bag.count(sub)
```
> defaultdict(<class 'int'>, {'apple': 3, 'orange': 1, 'cherry': 2, 'blueberry': 1})

## counter
* 返回一个字典，分别对应该元素在iterator出现的次数
```python
from collections import defaultdict, Counter  
bag = ['apple', 'orange', 'cherry', 'apple','apple', 'cherry', 'blueberry'] 
print(Counter(bag))
```

> Counter({'apple': 3, 'cherry': 2, 'orange': 1, 'blueberry': 1})

## deque(list有的它都有)
* **deque**提供了一个双端队列，你可以从头/尾两端添加或删除元素
```python
d = deque(['apple', 'orange', 'cherry', 'apple','apple', 'cherry', 'blueberry'], maxlen=None)  
print(d)  
print(d.pop())  
print(d.popleft())  
d.appendleft("aa")  
print(d)
```

> deque(['apple', 'orange', 'cherry', 'apple', 'apple', 'cherry', 'blueberry'])
>
> blueberry
> 
> apple
> 
> deque(['aa', 'orange', 'cherry', 'apple', 'apple', 'cherry'])



# 多进程
```python
from multiprocessing import Pool
def func(x):
	return x**2

inputs = range(1,60)
pool = Pool(5)  # 5个进程
outputs = pool.map(func, inputs)
print(outputs)
```

> [1, 4, 9, 16, 25]

# python 脚本
```python
import os
os.system("python 要执行的python文件.py")
```
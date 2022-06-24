1. 写好c++函数源码
```C
// 文件名: test.cpp
#include<iostream>
using namespace std;
  
extern "C"{
	int sum_func(int a, int b)
	{
		int result = 0;
		for (int i = 0; i < a; i++)
			{
				for (int j = 0;j < b; j++)
					result = result + 1;
			}
	return result;
	}
}

  
extern "C"{
	void hello()
	{
		cout << "hello world!" << endl;
	}
}
```

2. 编译cpp代码为so动态库
> g++ -o sum_func.so -shared -fPIC test.cpp

3. python调用so动态库
```python
import ctypes  
import time  
dll = ctypes.cdll.LoadLibrary  
lib = dll('./sum_func.so')  
lib.hello()  
  
def sum_func_py(a, b):  
    result = 0  
    for i in range(a):  
        for j in range(b):  
            result += 1  
    return result  
  
c_start_time = time.time()  
res_cpp = lib.sum_func(1000, 2000)  
c_end_time = time.time()  
py_start_time = time.time()  
res_py = sum_func_py(1000, 2000)  
py_end_time = time.time()  
print("c++所花的时间：", c_end_time - c_start_time)  
print("python所花的时间：", py_end_time - py_start_time)

```

    hello world!
	c++所花的时间： 0.0019528865814208984
	python所花的时间： 0.06929302215576172
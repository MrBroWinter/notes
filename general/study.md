# scatter
```python
import torch
import numpy as np
input = torch.FloatTensor(np.array([[11,12,13,14],
									[21,22,23,24],
									[31,32,33,34],
									[41,42,43,44]]))
output = torch.zeros(5, 5)
index = torch.tensor([[3, 1, 2, 0],
					  [1, 2, 0, 3]])
output = output.scatter(1, index, input)
# index[1][2]的值为0 ，那么output[1][0]=input[1][2]=23；index的坐标为[i][j],[i][j]对应的值为k，那么就将input中的[i][j]对应的值拿出来放在output的位置上[i][k]
print(output)
# tensor([[14., 12., 13., 11., 0.],
# [23., 21., 22., 24., 0.],
# [ 0., 0., 0., 0., 0.],
# [ 0., 0., 0., 0., 0.],
# [ 0., 0., 0., 0., 0.]])
output = torch.zeros(5, 5)
output = output.scatter(0, index, input) # dim对应的就是k属于哪个维度进行替换
# index[1][2]的值为0 ，那么output[0][2]=input[1][2]=23；index的坐标为[i][j],[i][j]对应的值为k，那么就将input中的[i][j]对应的值拿出来放在output的位置上[k][j]
print(output)
# tensor([[14., 12., 23., 14., 0.],
# [21., 12., 22., 24., 0.],
# [ 0., 22., 13., 0., 0.],
# [11., 0., 0., 24., 0.],
# [ 0., 0., 0., 0., 0.]])
```

# frangi滤波
```python
def objectness(arr, radius=None, dimension=1):
arrIm = sitk.GetImageFromArray(arr.astype('float32'))
if radius is not None:
	arrIm = sitk.DiscreteGaussian(arrIm, radius)
arrIm = sitk.ObjectnessMeasure(arrIm, objectDimension=dimension)
res = sitk.GetArrayFromImage(arrIm)
return res
```
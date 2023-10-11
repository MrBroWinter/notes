[sitk链接](https://blog.csdn.net/sinat_30618203/article/details/117464465)
# 连通域处理
```python
mask_itk_relab = sitk.RelabelComponent(sitk.ConnectedComponent(mask_itk), minimumObjectSize=900)
########################################################################
relabel_filter = sitk.RelabelComponentImageFilter()
relabel_filter.SetMinimumObjectSize(10)
relabeled_img = relabel_filter.Execute(ori_img)
```

# 形态学处理
```python
processed_img = sitk.LabelSetDilate(img, [3,3,3])    # 膨胀

processed_img = sitk.LabelSetErode(img, [3,3,3])     # 腐蚀

sitk.BinaryMorphologicalOpening(img, [3,3,3]) # 开

sitk.BinaryMorphologicalClosing(img, [3,3,3]) # 闭
```

# 标签映射
```python
new_img = sitk.ChangeLabel(img, label_change_dict)
```

# 标签统计
```python
label_stat = sitk.LabelShapeStatisticsImageFilter()

label_stat.Execute(imSeg)

labels = label_stat.GetLabels()

for cur_label in labels:

    pixels_num = label_stat.GetNumberOfPixels(cur_label)   # 获取label的像素数量
    
```

# 计算dice
```python
Filter = sitk.LabelOverlapMeasuresImageFilter()
Filter.Execute(itkLabel == label, itkPred == label)
dice = Filter.GetDiceCoefficient()
```

# 高斯滤波
```python
Filter = sitk.DiscreteGaussianImageFilter()
 
Filter.SetVariance(3)
 
Filter.SetMaximumError(0.2)
 
output_itk = Filter.Execute(img_itk)
```

# 像素值缩放
```python
Filter = sitk.RescaleIntensityImageFilter() 
 
Filter.SetOutputMaximum(255)
 
Filter.SetOutputMinimum(0)
 
output_itk = Filter.Execute(img_itk)
```

# 物理-像素坐标转换
```python
img_itk.TransformContinuousIndexToPhysicalPoint(point[::-1])
img_itk.TransformPhysicalPointToContinuousIndex(point)[::-1]
```

# 获取bbox
```python
def get_bbox_from_mask(bin_mask):  
        label_filter = sitk.LabelShapeStatisticsImageFilter()  
        label_filter.Execute(bin_mask)  
        bbox = label_filter.GetBoundingBox(1)  # 或者GetRegion也是一样的  
  
        return bbox
```
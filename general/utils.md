```python
import random
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform,data,exposure
import scipy.ndimage as ndimage
import cv2
import SimpleITK as sitk
import skimage
import os
from skimage.measure import label, regionprops
from skimage.morphology import closing


def visualization_CT(ct, mask, img_save_path):
    ct_flatten = ct.flatten()
    mask_flatten = mask.flatten()
    ct_res = ct_flatten[np.where(mask_flatten > 0)[0]]
    plt.hist(ct_res, bins=400)
    plt.savefig(img_save_path)
    plt.close()

# -------------------------
# get boundaries box
# -------------------------
def get_boundaries_from_mask(mask):

    mask_voxel_coords = np.where(mask != 0)
    zmin = int(np.min(mask_voxel_coords[0]))
    zmax = int(np.max(mask_voxel_coords[0])) + 1
    ymin = int(np.min(mask_voxel_coords[1]))
    ymax = int(np.max(mask_voxel_coords[1])) + 1
    xmin = int(np.min(mask_voxel_coords[2]))
    xmax = int(np.max(mask_voxel_coords[2])) + 1
    out_bbox = {'zmin': zmin,
                'zmax': zmax,
                'ymin': ymin,
                'ymax': ymax,
                'xmin': xmin,
                'xmax': xmax}
    return out_bbox



# **********************************
# -------------------------
# read raw file
# -------------------------
def read_dicom(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    raw = reader.Execute()
    return raw




# -------------------------
# 获取有效切片区域范围
# -------------------------
def getRangeImageDepth(seg_array):

	firstflag = True
	startposition = 0
	endposition = 0
	for z in range(seg_array.shape[0]):
		notzeroflag = np.max(seg_array[z])
		if notzeroflag and firstflag:
			startposition = z
			firstflag = False
		if notzeroflag:
			endposition = z
	return startposition, endposition

# **********************************
# **********************************



# # -------------------------
# # 数据增强, 增强�?d数据输入
# # -------------------------
# The data randomly selects an augment method in the dataloader
def data_augment(ct, seg):
    index = random.randint(0, 15)
    global ct_res, seg_res
    # if 0 <= index < 1:     # probability = 0.05  高斯噪声
    #     var = random.uniform(0, 0.01)
    #     ct_res = skimage.util.random_noise(ct.copy(), mode='gaussian', var=var)
    #     seg_res = seg.copy()

    if 0 <= index < 2:    # probability = 0.1  高斯模糊
        ct_res = cv2.GaussianBlur(ct.copy(), (3, 3), 0)
        seg_res = seg.copy()

    elif 2 <= index < 3:    # # probability = 0.05  数据随机乘以[0.65，1.5]内的值
        mn = ct.min()
        coefficient = random.uniform(0.65, 1.5)
        ct_res = (ct.copy() - mn) * coefficient + mn
        seg_res = seg.copy()

    elif 3 <= index < 4:   # probability = 0.1  数据随机角度旋转
        angle = random.uniform(-30, 30)
        ct_res = ndimage.rotate(ct.copy(), angle, axes=(1, 2), reshape=False, cval=0)
        seg_res = np.round(ndimage.rotate(seg.copy(), angle, axes=(1, 2), reshape=False, cval=0))

    elif 4 <= index < 6:    # probability = 0.1  y轴镜像
        ct_res = ct[:, ::-1, :].copy()
        seg_res = seg[:, ::-1, :].copy()

    elif 6 <= index < 8:    # probability = 0.1  z轴镜像
        ct_res = ct[::-1, :, :].copy()
        seg_res = seg[::-1, :, :].copy()

    elif 8 <= index < 10:    # probability = 0.1  x轴镜像
        ct_res = ct[:, :, ::-1].copy()
        seg_res = seg[:, :, ::-1].copy()


    elif 10 <= index < 11:   # # probability = 0.05  gamma
        seg_res = seg.copy()
        mn = ct.mean()
        sd = ct.std()
        gamma = np.random.uniform(0.5, 2)
        minm = ct.min()
        rnge = ct.max() - minm
        ct_res = np.power((ct.copy() - minm) / float(rnge + 1e-7), gamma) * rnge + minm

        ct_res = ct_res - ct_res.mean()
        ct_res = ct_res / (ct_res.std() + 1e-8) * sd
        ct_res = ct_res + mn

    elif 11 <= index < 12:   # probability = 0.05  crop
        random_x = random.randint(0, ct.shape[2] - ct.shape[2] // 2)
        random_y = random.randint(0, ct.shape[1] - ct.shape[1] // 2)
        random_z = random.randint(0, ct.shape[0] - ct.shape[0] // 2)
        ct_res = ct.copy()[random_z:random_z + ct.shape[0] // 2, random_y:random_y + ct.shape[1] // 2, random_x:random_x + ct.shape[2] // 2]
        seg_res = seg.copy()[random_z:random_z + ct.shape[0] // 2, random_y:random_y + ct.shape[1] // 2, random_x:random_x + ct.shape[2] // 2]

        ct_res = ndimage.zoom(ct_res, (2, 2, 2), order=3)
        seg_res = ndimage.zoom(seg_res, (2, 2, 2), order=0)

    else:
        ct_res = ct
        seg_res = seg
    return ct_res, seg_res.astype(np.uint8)

# # **********************************
# # **********************************


# # **********************************
# # **********************************

# # -------------------------
# # 重采样到 new_spacing [x(mm) * y(mm) * z(mm)]
# # -------------------------

def Resampling(img, new_spacing, label = False):
    original_size = img.GetSize() #获取图像原始尺寸
    original_spacing = img.GetSpacing() #获取图像原始分辨�?
    # new_spacing = [1, 1, 1] #设置图像新的分辨率为1*1*1

    new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))] #计算图像在新的分辨率下尺寸大�?

    resampleSliceFilter = sitk.ResampleImageFilter() #初始�?

    if label == False:
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkBSpline,
                                                img.GetOrigin(), new_spacing, img.GetDirection(), 0, img.GetPixelIDValue())


    else:# for label, should use sitk.sitkNearestNeighbor to make sure the original and resampled label are the same!!!
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())


    return Resampleimage

# # **********************************
# # **********************************

def resampleVolume(vol, outspacing, mask=False):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]

    # 读取文件的size和spacing信息

    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    if mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol

# -------------------------
# 1/2重叠滑块预测
# -------------------------
import math
import torch
def overlap_predict(net, ct, patch_size, rate=2):
    """
    :param net: 网络模型
    :param data_for_predict:   输入的待预测矩阵
    :param patch_size:     切块大小
    :param rate=2   1/2重叠滑块
    :return: 预测输出结果   与输入大小一致
    """

    # slide_rate = patch_size // rate  # 滑块的步长，默认为patch_size的一半（64）
    slide_rate = list(map(lambda x: x // rate, patch_size))
    pad_num_z = (patch_size[0] - ct.shape[0] % patch_size[0]) if (
            ct.shape[0] <= patch_size[0]) else (slide_rate[0] - ct.shape[0] % slide_rate[0])
    pad_num_y = (patch_size[1] - ct.shape[1] % patch_size[1]) if (
            ct.shape[1] <= patch_size[1]) else (slide_rate[1] - ct.shape[1] % slide_rate[1])
    pad_num_x = (patch_size[2] - ct.shape[2] % patch_size[2]) if (
            ct.shape[2] <= patch_size[2]) else (slide_rate[2] - ct.shape[2] % slide_rate[2])

    tmp_ct = np.pad(ct, ((0, pad_num_z), (0, pad_num_y), (0, pad_num_x)), 'constant')


    z_slide_num = math.ceil((tmp_ct.shape[0] - patch_size[0]) / slide_rate[0]) + 1
    y_slide_num = math.ceil((tmp_ct.shape[1] - patch_size[1]) / slide_rate[1]) + 1
    x_slide_num = math.ceil((tmp_ct.shape[2] - patch_size[2]) / slide_rate[2]) + 1
    # 保存最终的预测结果
    tmp_res = np.zeros(tmp_ct.shape)

    # 用于计算滑块过程中每个像素重叠的次数
    num_array = np.zeros(tmp_res.shape)
    with torch.no_grad():
        for xx in range(x_slide_num):
            for yy in range(y_slide_num):
                for zz in range(z_slide_num):
                    ct_part = tmp_ct[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                              yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                              xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]]


                    # 当下滑块对应的像素点滑块次数+1
                    num_array[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                    yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                    xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]] += 1

                    ct_tensor = torch.FloatTensor(ct_part).cuda()
                    ct_tensor = ct_tensor.unsqueeze(dim=0)
                    ct_tensor = ct_tensor.unsqueeze(dim=0)

                    # print(ct_tensor.shape)
                    outputs = net(ct_tensor)
                    outputs = outputs.squeeze().cpu().detach().numpy()
                    # 将预测的结果加入到对应的位置上
                    tmp_res[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                    yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                    xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]] += outputs
    tmp_res = tmp_res / num_array  # 对应像素点叠加预测的结果除以相对应的次数，得到预测平均值
    return tmp_res[0:ct.shape[0], 0:ct.shape[1], 0:ct.shape[2]]

---------------------------------------------------------------------------
def keep_maximum_volume(mask):
    """
    :param mask:  二值图像（3D）
    :param thread: 保留体积大于指定阈值的连通域
    :return: 主要连通域
    """
    mask = closing(mask)
    label_mask = label(mask, connectivity=1)
    valid_label = set()
    properties = regionprops(label_mask)
    volume = []
    for sub in properties:
        volume.append(sub.area)
    for i, prop in enumerate(properties):
        if prop.area == max(volume):
            valid_label.add(prop.label)

    res = np.in1d(label_mask, list(valid_label)).reshape(label_mask.shape).astype(int)
    # res = closing(res)
    return res
---------------------------------------------------------------------------
def remove_small_volume(mask, threshold=800):
    """
    :param mask:  二值图像（3D）
    :param thread: 保留体积大于指定阈值的连通域
    :return: 主要连通域
    """
    label_mask = label(mask)
    valid_label = set()
    properties = regionprops(label_mask)
    for i, prop in enumerate(properties):
        if prop.area > threshold:
            valid_label.add(prop.label)
        #print(i)
    res = np.in1d(label_mask, list(valid_label)).reshape(label_mask.shape).astype(int)
    return res
---------------------------------------------------------------------------
import itertools
def detect_endpoint(center_line):
    """
    :param center_line: 骨架提取后的中心线
    :return:
        endpoint:三维矩阵形式的坐标显示
        coordinates：坐标点
        num_endpoint：端点个数
    """
    end_point = np.zeros_like(center_line)
    z_shape, y_shape, x_shape = np.shape(end_point)
    items = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    coordinates = []
    num_endpoint = 0
    for i in range(1, x_shape - 1):
        for j in range(1, y_shape - 1):
            for k in range(1, z_shape - 1):
                # 当前中心点为1时，则计算其所有领域和自身点的和，若和为2，则说明是端点
                if center_line[k, j, i] == 1:
                    sum = 0
                    for item in items:
                        sum += center_line[k + item[0]][j + item[1]][i + item[2]]
                    if sum == 2:   # 说明为端点
                        end_point[k][j][i] = 1
                        coordinates.append([k, j, i])
                        num_endpoint += 1

    return end_point, coordinates, num_endpoint
---------------------------------------------------------------------------
def seed_grow(CT_array, main_mask_data, seed, thread=80):

    seedList = []

    # 因为seed不一定准确，具有一定的偏差，seed的周围部分领域可能都是已经标记的mask区域，因此我们可以对目标seed一定范围内进行搜索，将符合条件的加入到seedList中
    # seedList中的坐标对应的mask不一定是1

    num = 3   # seed周围5*5*5个像素点搜索候选种子
    tmp_points = list(itertools.product(range(-num, num + 1), range(-num, num + 1), range(-num, num + 1)))
    for tmp_point in tmp_points:
        tmp_z = seed[0] + tmp_point[0]
        tmp_y = seed[1] + tmp_point[1]
        tmp_x = seed[2] + tmp_point[2]

        # 筛选条件
        if CT_array[tmp_z][tmp_y][tmp_x] > 110 and main_mask_data[tmp_z][tmp_y][tmp_x] == 0:
            seedList.append([tmp_z, tmp_y, tmp_x])
    # seedList.append(seed)
    label = 1
    # 26连通域+中心点
    items = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    iter_num = 3000
    while len(seedList) > 0:

        z, y, x = seedList.pop(0)

        # main_mask_data[z][y][x] = label
        for item in items:
            try:
                # 如果没超过边界，则计算目标前和种子的差值，计算目标是否跟种子是同一类
                object_z = z + item[0]   # 待定目标的坐标
                object_y = y + item[1]
                object_x = x + item[2]
                object_point = CT_array[object_z][object_y][object_x]
                diff = abs(CT_array[z][y][x] - object_point)
                if diff < thread and main_mask_data[object_z][object_y][object_x] == 0 and iter_num > 0 and CT_array[object_z][object_y][object_x] > 130:
                    main_mask_data[object_z][object_y][object_x] = label
                    seedList.append((object_z, object_y, object_x))
                    iter_num -= 1
            except:
                pass
---------------------------------------------------------------------------
from skimage import morphology
def point_for_visualization(coordinates, thined_line_array, num=1):
    """
    :param coordinates: 端点坐标(z, y, x)格式
    :param thined_line_array: 骨架提取后的数组
    :param num:  显示端点的半径
    :return: 可以利用itk_snap可视化的端点数组
    """
    line_array = morphology.dilation(thined_line_array)
    res = np.zeros_like(thined_line_array)
    items = list(itertools.product(range(-num, num + 1), range(-num, num + 1), range(-num, num + 1)))

    for coordinate in coordinates:
        z, y, x = coordinate
        try:
            for item in items:
                res[z + item[0]][y + item[1]][x + item[2]] = 3
        except:
            pass
    res = line_array + res
    res[res > 1] = 2
    return res

---------------------------------------------------------------------------
# -------------------------
# 标签转为onehot
# -------------------------
def label2onehot(mask, num_classes):
    """
    :param mask: 五维Tensor数据 N*1*D*W*H
    :param num_classes: 类别数，包括背景
    :return: 五维数据 N*C*D*W*H
    """
    N, _, D, W, H = mask.shape
    result_onehot = torch.zeros((N, num_classes, D, W, H))
    for cur_batch in range(N):
        for label_index in range(num_classes):
            tmp = torch.zeros((D, W, H))
            tmp[mask[cur_batch][0] == label_index] = 1
            result_onehot[cur_batch][label_index] = tmp
    return result_onehot


--------------------------------------------------------------------------
# 在读取该函数保存的文件时可能需要进行数据类型的转换
def itk_keep_maximum_connect_domain(itk_label):
    cc = sitk.ConnectedComponent(itk_label)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, itk_label)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0
    outmasksitk = sitk.GetImageFromArray(outmask)
    outmasksitk.SetSpacing(itk_label.GetSpacing())
    outmasksitk.SetOrigin(itk_label.GetOrigin())
    outmasksitk.SetDirection(itk_label.GetDirection())
    return outmasksitk

---------------------------------------------------------------------------
#pytorch指数滑动平均
# 使用方法参考 https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3
class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
---------------------------------------------------------------------------
def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # //获取原图size
    originSpacing = itkimage.GetSpacing()  # //获取原图spacing
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)   # //spacing格式转换
    resampler.SetReferenceImage(itkimage)   # //指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # //得到重新采样后的图像
    return itkimgResampled
---------------------------------------------------------------------------
# 负样本难样本挖掘
def neg_soft_mining(self, neg_loss, tp_seg):   # 筛选出负样本中概率较大的若干个预测像素点当做难样本
    batch_size = tp_seg.shape[0]
    neg_loss = neg_loss.view(batch_size, -1)

    pos_point_num = torch.clamp(torch.sum(tp_seg) / batch_size, min=10)
    neg_point_num = torch.clamp(pos_point_num * 3, max=neg_loss.shape[1])
    neg_point_num2 = int(torch.clamp(neg_point_num * 1.5, max=neg_loss.shape[1]))

    _, idxs1 = torch.topk(neg_loss, neg_point_num2, dim=1, largest=True)
    perm = torch.randperm(neg_point_num2)
    idxs1 = idxs1[:, perm[: int(neg_point_num)]]
    neg_mask = torch.zeros_like(neg_loss).scatter_(1, idxs1, 1)

    return neg_mask.view(tp_seg.shape)
---------------------------------------------------------------------------
def Focus_HistogramEqualization(dcm_npy, seg_npy):
    masked_dcm = dcm_npy * seg_npy
    dcm_min, dcm_max = masked_dcm.min(), masked_dcm.max()
    print(dcm_min, dcm_max)
    focus_points = np.argwhere(seg_npy > 0)
    HuStatic = [0 for _ in range(dcm_min, dcm_max + 1)]   # 存放dcm_min到dcm_max HU值出现的个数
    for i in range(len(focus_points)):
        point = focus_points[i]
        point_HU_value = dcm_npy[point[0], point[1], point[2]]
        HuStatic[point_HU_value - dcm_min] += 1
    gyh = [i / (focus_points.shape[0]) for i in HuStatic]

    cdf = gyh.copy()
    for j in range(1, dcm_max + 1):
        cdf[j] = cdf[j] + cdf[j - 1]

    # 重新调整FocusHU值
    for i in range(len(focus_points)):
        subpoint = focus_points[i]
        dcm_npy[subpoint[0], subpoint[1], subpoint[2]] = dcm_npy[subpoint[0], subpoint[1], subpoint[2]] * cdf[dcm_npy[subpoint[0], subpoint[1], subpoint[2]] - dcm_min]
    return dcm_npy
---------------------------------------------------------------------------
def rotate_3D(img, seg, degree=[-45, 45]):  
    """  
    围绕中心旋转  
    """    angle_xy = random.randint(degree[0], degree[1])  
    img = ndimage.rotate(img, angle_xy, axes=(1, 2), reshape=False, cval=0)  
    seg = np.round(ndimage.rotate(seg, angle_xy, axes=(1, 2), reshape=False, cval=0))  
  
    angle_xz = random.randint(degree[0], degree[1])  
    img = ndimage.rotate(img, angle_xz, axes=(0, 2), reshape=False, cval=0)  
    seg = np.round(ndimage.rotate(seg, angle_xz, axes=(0, 2), reshape=False, cval=0))  
  
    angle_yz = random.randint(degree[0], degree[1])  
    img = ndimage.rotate(img, angle_yz, axes=(0, 1), reshape=False, cval=0)  
    seg = np.round(ndimage.rotate(seg, angle_yz, axes=(0, 1), reshape=False, cval=0))  
  
    return img, seg

def appearance_aug(dicom, seg):  
    img_size = dicom.shape  
    points = np.argwhere(sm.skeletonize_3d(seg) > 0)  
  	if len(points) > 0:
		random_times = random.randint(1, 5)  
		for _ in range(random_times):  
			random_point = points[random.randint(0, len(points))]  
			print(random_point)  
			z_half_len = min(random_point[0], img_size[0] - random_point[0])  
			y_half_len = min(random_point[1], img_size[1] - random_point[1])  
			x_half_len = min(random_point[2], img_size[2] - random_point[2])  

			dicom_patch = dicom[(random_point[0] - z_half_len):(random_point[0] + z_half_len),  
						  (random_point[1] - y_half_len):(random_point[1] + y_half_len),  
						  (random_point[2] - x_half_len):(random_point[2] + x_half_len)]  

			seg_patch = seg[(random_point[0] - z_half_len):(random_point[0] + z_half_len),  
						(random_point[1] - y_half_len):(random_point[1] + y_half_len),  
						(random_point[2] - x_half_len):(random_point[2] + x_half_len)]  
			dicom_patch_rotate, seg_patch_rotate = rotate_3D(dicom_patch, seg_patch, degree=[-60, 60])  

			# 腐蚀一下，防止边界旋转带来的杂点影响  
			seg_patch_rotate = erosion(seg_patch_rotate)  
			dicom_rotate_res = np.zeros_like(dicom)  
			seg_rotate_res = np.zeros_like(seg)  
			dicom_rotate_res[(random_point[0] - z_half_len):(random_point[0] + z_half_len),  
			(random_point[1] - y_half_len):(random_point[1] + y_half_len),  
			(random_point[2] - x_half_len):(random_point[2] + x_half_len)] = dicom_patch_rotate  
			seg_rotate_res[(random_point[0] - z_half_len):(random_point[0] + z_half_len),  
			(random_point[1] - y_half_len):(random_point[1] + y_half_len),  
			(random_point[2] - x_half_len):(random_point[2] + x_half_len)] = seg_patch_rotate  
			intersection = seg * seg_rotate_res  
			dicom = (1 - intersection) * dicom + intersection * dicom_rotate_res  
    return dicom





if __name__ == '__main__':
    dcm_path = r"/media/data/stent_0422/val_test/v2.0/test/dicom/CE021004-70336786-Cardiac-8"
    seg_path = r"/media/data/stent_0422/val_test/v2.0/test/nii/CE021004-70336786-Cardiac-8-seg.nii.gz"
    dcm_nii = read_dicom(dcm_path)
    dcm_npy = sitk.GetArrayFromImage(dcm_nii)
    seg_npy = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    res = Focus_HistogramEqualization(dcm_npy, (seg_npy > 2).astype(np.uint8))
    res_itk = sitk.GetImageFromArray(res)
    res_itk.CopyInformation(dcm_nii)
    sitk.WriteImage(res_itk, r"/home/zdongdong/桌面/tmp.nii.gz")
	

```

## torch 仿射变换
```python
"""  
https://blog.csdn.net/weixin_42917352/article/details/118730410   # 2D  
https://blog.csdn.net/shenquanyue/article/details/103262512   # 3D  
"""  

"""
alpha = 20  
beta =20  
gamma =20  
rotate_Z_mtx = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],  
                         [np.sin(gamma), np.cos(gamma),  0, 0],  
                         [0,             0,              1, 0],  
                         [0,             0,              0, 1]])  
  
rotate_Y_mtx = np.array([[np.cos(beta),  0, np.sin(beta), 0],  
                         [0,             1, 0,            0],  
                         [-np.sin(beta), 0, np.cos(beta), 0],  
                         [0,             0, 0,            1]])  
  
rotate_X_mtx = np.array([[1, 0,             0,              0],  
                         [0, np.cos(alpha), -np.sin(alpha), 0],  
                         [0, np.sin(alpha), np.cos(alpha),  0],  
                         [0, 0,             0,              1]])  
rotate_mtx = rotate_Z_mtx @ rotate_Y_mtx @ rotate_X_mtx
"""
import SimpleITK as sitk  
import numpy as np  
import cv2  
import torch  
  
  
# 给grid增加旋转信息  
def get_rotate_grid(img,  
             degree=45,  
             rotate_point=[256, 256]):  
    rows = img.shape[0]  
    cols = img.shape[1]  
    grid_y = np.arange(0, rows, 1)  
    grid_x = np.arange(0, cols, 1)  
    grid = np.meshgrid(grid_y, grid_x)  
    grid = [g[:, :, None] for g in grid]  
    grid = np.concatenate(grid, axis=-1)  
  
    alpha = degree / 180 * np.pi  
    rotate_mtx = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])  
    center_point = np.array([[rotate_point]])  
    grid -= center_point  
    grid = np.matmul(grid, np.linalg.inv(rotate_mtx))  
    grid += center_point  
  
    grid *= 2  
    grid /= np.array([[[cols-1, rows-1]]])  
    grid -= 1   # 缩放到[-1， 1]    grid = torch.from_numpy(grid.astype(np.float32))[None]  
    return grid  
   
  
img = cv2.imread(r"/home/zdongdong/桌面/test.jpg", 0)  
img = cv2.resize(img, (512, 512))  
img_res = torch.nn.functional.grid_sample(torch.from_numpy(img[None, None, ...]).float(), get_rotate_grid(img), align_corners=False)  
img_res = img_res.squeeze().numpy().astype(np.uint8)  
  
########################################################################  
rows = img.shape[0]  
cols = img.shape[1]  
alpha = 45 / 180 * np.pi  # 度数  
rotate_mtx = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])  
scale_w = 1  
scale_h = 1  
scale_mtx = np.array([[scale_w, 0], [0, scale_h]])  
mtx = scale_mtx @ rotate_mtx  
affine_mtx = np.zeros([3,3])  
affine_mtx[:2, :2] = mtx  
affine_mtx[2,2] = 1  
affine_mtx = torch.from_numpy(affine_mtx).float()  
grid_torch = torch.nn.functional.affine_grid(affine_mtx[None, :2], [1, 1, rows, cols], align_corners=True)  
img_res_torch = torch.nn.functional.grid_sample(torch.from_numpy(img[None, None, ...]).float(), grid_torch, align_corners=False)  
  
img_res_torch = torch.squeeze(img_res_torch).numpy().astype(np.uint8)  
  
imgs = np.hstack([img, img_res, img_res_torch])  
cv2.imshow("res image", imgs)  
cv2.waitKey()  
  

```


## 弹性形变
```python
def _affine_elastic_transform_3d(
        self, vol, alpha=[2.0, 2.0, 2.0], smooth_num=4, win=[5, 5, 5], field_size=[17, 17, 17]
    ):
        batch_size = vol.size(0)
        aff_matrix = torch.zeros([batch_size, 3, 4])
        aff_matrix[:, 0, 0] = 1
        aff_matrix[:, 1, 1] = 1
        aff_matrix[:, 2, 2] = 1

        grid = torch.nn.functional.affine_grid(aff_matrix, vol.size())

        pad = [win[i] // 2 for i in range(3)]
        fs = field_size
        dz = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2)
        dy = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2)
        dx = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2)
        dz = (dz - 0.5) * 2.0 * alpha[0]
        dy = (dy - 0.5) * 2.0 * alpha[1]
        dx = (dx - 0.5) * 2.0 * alpha[2]

        for _ in range(smooth_num):
            dz = self.__smooth_3d__(dz, win)
            dy = self.__smooth_3d__(dy, win)
            dx = self.__smooth_3d__(dx, win)

        dz = dz[:, :, pad[0] : pad[0] + fs[0], pad[1] : pad[1] + fs[1], pad[2] : pad[2] + fs[2]]
        dy = dy[:, :, pad[0] : pad[0] + fs[0], pad[1] : pad[1] + fs[1], pad[2] : pad[2] + fs[2]]
        dx = dx[:, :, pad[0] : pad[0] + fs[0], pad[1] : pad[1] + fs[1], pad[2] : pad[2] + fs[2]]

        size_3d = [vol.size(2), vol.size(3), vol.size(4)]
        dz = torch.nn.functional.interpolate(dz, size=size_3d, mode="trilinear", align_corners=False).repeat(
            batch_size, 1, 1, 1, 1
        )
        dy = torch.nn.functional.interpolate(dy, size=size_3d, mode="trilinear", align_corners=False).repeat(
            batch_size, 1, 1, 1, 1
        )
        dx = torch.nn.functional.interpolate(dx, size=size_3d, mode="trilinear", align_corners=False).repeat(
            batch_size, 1, 1, 1, 1
        )

        grid[:, :, :, :, 0] += dz[:, 0, :, :, :]
        grid[:, :, :, :, 1] += dy[:, 0, :, :, :]
        grid[:, :, :, :, 2] += dx[:, 0, :, :, :]

        vol_o = torch.nn.functional.grid_sample(vol, grid, mode="bilinear")

        return vol_o

    def __smooth_3d__(self, vol, win):
        kernel = torch.ones([1, vol.size(1), win[0], win[1], win[2]])
        pad_size = [
            (int)((win[2] - 1) / 2),
            (int)((win[2] - 1) / 2),
            (int)((win[1] - 1) / 2),
            (int)((win[1] - 1) / 2),
            (int)((win[0] - 1) / 2),
            (int)((win[0] - 1) / 2),
        ]
        vol = torch.nn.functional.pad(vol, pad_size, "replicate")
        vol_s = torch.nn.functional.conv3d(vol, kernel, stride=(1, 1, 1)) / torch.sum(kernel)
        return vol_s

```

# 自定义膨胀腐蚀
```python
from skimage.morphology import binary_dilation
res_arr = binary_dilation(arr, np.ones([3,6,6], bool))
```
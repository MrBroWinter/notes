```python
"""https://antspyx.readthedocs.io/en/latest/registration.html"""
import SimpleITK as sitk
import numpy as np
from skimage.morphology import binary_erosion, binary_opening, erosion
import torch
import os
import matplotlib.pyplot as plt
import tqdm
import functools
import threading
import random
import logging
import subprocess, shlex
import glob
import traceback
from starship import sitktools as st
import time

def simpleRegistration(
    fixed,
    moving,
    fixedMask=None,
    movingMask=None,
    transform=None,
    initialize=True,
    iterations=100,
    lr=0.1,
    samplingPercentage=0.1,
    histogramBins=25,
    multiLevel=None
):
    """Simple simpleitk based registration with mattes MI metric and gradient
    descent optimizer.

    :param fixed: Fixed Image.
    :param moving: Moving Image.
    :param fixedMask: Fixed mask. default to None.
    :param movingMask: Moving mask. default to None.
    :param transform: Initial moving to fixed transform. default to None = AffineTransform.
    :param initialize: Initialize transform by sitk.CenteredTransformInitializer. default to True.
    :param iterations: Iterations, default to 100.
    :param lr: Learning rate, default to 0.1.
    :param samplingPercentage: Sampling percentage to mattes MI metric. default to 0.1.
    :param histogramBins: Histogram bins to calculate mattes MI metric. default to 25.
    :param multiLevel: Multilevel shrink & smooth, None: disabled,
                           if an int (k) is provided: shrink factor: [2**k for k in range(k)], smooth sigma: range(k),
                           if a tuple is provided: ([shrink factor], [smooth sigma]).
                           default to None.
    :return: moving to fixed transform.
    """
    if transform is None:
        transform = sitk.AffineTransform(fixed.GetDimension())
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    if initialize:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, transform, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:
        initial_transform = transform

    registration_method = sitk.ImageRegistrationMethod()
    if fixedMask is not None:
        registration_method.SetMetricFixedMask(fixedMask)
    if movingMask is not None:
        registration_method.SetMetricMovingMask(movingMask)
        
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=histogramBins)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(samplingPercentage, seed=49999)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iterations)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    if multiLevel is not None:
        if type(multiLevel) is int:
            shrinkFactor = [2**k for k in range(multiLevel)]
            smoothFactor = [k for k in range(multiLevel)]
        elif type(multiLevel) is tuple:
            shrinkFactor = multiLevel[0]
            smoothFactor = multiLevel[1]
        else:
            raise ValueError('Unrecognized multiLevel parameter: %s' % str(multiLevel))
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinkFactor)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothFactor)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    translation = registration_method.Execute(fixed, moving)
    return translation

def getAntsCommandline(origPath, targPath, resPath, origMaskPath="NULL", targMaskPath="NULL", antsPath=None):
    if antsPath is None:
        antsPath = os.path.dirname(__file__) + '/'
    cmdLine = 'antsRegistration --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 ' \
                '--output {res} --interpolation Linear --use-histogram-matching 1 ' \
                '--winsorize-image-intensities "[0.005,0.995]" -x [{origmask}, {targmask}]' \
                ' --initial-moving-transform "[{orig},{targ},1]" --transform Affine"[0.1]" ' \
                '--metric GC"[{orig},{targ},1,32,Regular,0.25]" --random-seed 49999 --convergence "[1000x500x250x0,1e-6,10]"' \
                ' --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x2x1vox '.format(orig=origPath, targ=targPath, res=resPath,origmask=origMaskPath, targmask=targMaskPath)

    return antsPath + cmdLine

class ANTsRegistrationException(Exception):
    pass


class ANTsAsyncThread(threading.Thread):
    def __init__(self, runFunction, name=""):
        super(ANTsAsyncThread, self).__init__(name=name)
        self.runFunction = runFunction
        self.result = None

    def run(self):
        self.result = self.runFunction()


class ANTsRegistration(object):

    def __init__(self, tmpPath=None, antsPath=None, logr=None):
        
        self.tmpPath = tmpPath
        self.antsPath = antsPath
        if logr is None:
            self.logr = logging.getLogger('ANTsRegistration')
        else:
            self.logr = logr

    def getFileName(self, path=None):
        if path is None:
            path = self.tmpPath
        nameGenerator = lambda x: "".join([chr(random.randrange(65, 91)) for i in range(x)])
        curFileName = nameGenerator(8)
        while len(glob.glob(path + "/" + curFileName + "*")) > 0:
            curFileName = nameGenerator(8)
        self.logr.debug('tmpFilename: ' + curFileName)
        return curFileName

    def exec_run(self, cmd):
        self.logr.debug('Run command: ' + cmd)
        try:
            p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = b"------\n".join([i if i is not None else b"\n" for i in p.communicate()]) 
        except:
            res = traceback.format_exc()
        return res

    def runByPath(self, origPath, targPath, curFileName="", maskPath="NULL", return_log=False, coerce_errors=True):

        if curFileName == "":
            curFileName = self.getFileName()
        self.logr.info('ANTsRegistration start.')
        res = self.exec_run(getAntsCommandline(origPath, targPath, os.path.join(self.tmpPath, curFileName), maskPath, antsPath=self.antsPath))
        self.logr.debug(res)
        resMatFile = os.path.join(self.tmpPath, curFileName + "0GenericAffine.mat")
        if os.path.isfile(resMatFile):
            resAffine = sitk.ReadTransform(resMatFile)
            self.exec_run("rm %s" % os.path.join(self.tmpPath, curFileName + "0GenericAffine.mat"))
            self.logr.info('ANTsRegistration finished. %s' % (str([list(resAffine.GetFixedParameters()), list(resAffine.GetParameters())])))
            if return_log:
                return resAffine, res
            else:
                return resAffine
        else:
            if coerce_errors:
                self.logr.warning('Error while executing ANTsRegistration.')
                if return_log:
                    return sitk.AffineTransform(3), res
                else:
                    return sitk.AffineTransform(3)
            else:
                raise ANTsRegistrationException()

    def runAsyncByPath(self, origPath, targPath, maskPath="NULL", **kwargs):
        if not "curFileName" in kwargs:
            curFileName = self.getFileName()
        else:
            curFileName = kwargs["curFileName"]
        thread = ANTsAsyncThread(
            functools.partial(self.runByPath, origPath=origPath, targPath=targPath, curFileName=curFileName,
                                maskPath=maskPath, **kwargs), curFileName)
        thread.start()
        return thread

    def runByImage(self, imTemplate, imTarget, imMask=None, curFileName=None, **kwargs):
        if curFileName is None:
            curFileName = self.getFileName()

        templatePath = os.path.join(self.tmpPath, curFileName + "_template.nii.gz")
        targetPath = os.path.join(self.tmpPath, curFileName + "_target.nii.gz")
        maskPath = "NULL"
        sitk.WriteImage(imTemplate, templatePath)
        sitk.WriteImage(imTarget, targetPath)
        if not imMask is None:
            maskPath = os.path.join(self.tmpPath, curFileName + "_template_mask.nii.gz")
            sitk.WriteImage(imMask, maskPath)
        res = self.runByPath(templatePath, targetPath, curFileName=curFileName, maskPath=maskPath, **kwargs)
        for i in [templatePath, targetPath, maskPath]:
            if os.path.isfile(i):
                self.exec_run('rm %s' % i)
        return res

    def runAsyncByImage(self, imTemplate, imTarget, imMask=None, curFileName=None, **kwargs):
        if curFileName is None:
            curFileName = self.getFileName()
        thread = ANTsAsyncThread(functools.partial(self.runByImage, imTemplate=imTemplate, imTarget=imTarget, imMask=imMask,
                                                    curFileName=curFileName, **kwargs), curFileName)
        thread.start()
        return thread



fix_image = sitk.ReadImage(r"/media/tx-deepocean/Data/data/lymph_node/test/registration/fix_image/410545-seg.nii.gz", sitk.sitkFloat32)
moving_image = sitk.ReadImage(r"/media/tx-deepocean/Data/data/lymph_node/test/registration/moving_image/vessels_airway_seg/426741-seg.nii.gz", sitk.sitkFloat32)
t1 = time.time()

model_dirname = r"/media/tx-deepocean/Data/data"
ants_registration = ANTsRegistration(tmpPath="/tmp", antsPath=model_dirname + '/')
mtyx = ants_registration.runByImage(st.resampleByRef(fix_image, spacing=[1.5] * 3), st.resampleByRef(moving_image, spacing=[1.5] * 3))

res = st.resampleByRef(moving_image, fix_image, transform=mtyx, interpolator=sitk.sitkNearestNeighbor)
print(time.time() - t1)
sitk.WriteImage(res, r"/media/tx-deepocean/Data/data/lymph_node/test/registration/moving_image/res.nii.gz")
```
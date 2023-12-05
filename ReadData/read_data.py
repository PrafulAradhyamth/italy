import numpy as np
import nibabel as nib
import yaml


def read_dicoms(currentDataDir, patient=None, dataset=None, database=None):
    try:
        dicomDataset = nib.load(currentDataDir)
    except IOError:
        print("Could not read file:", currentDataDir)
    voxel_ndarray = dicomDataset.get_fdata()
    voxel_ndarray = np.array(voxel_ndarray.real)
    voxel_ndarray = voxel_ndarray.astype(float)
    voxel_ndarray = np.swapaxes(voxel_ndarray, 0, 1)

    # normalization of DICOM voxel
    rangeNorm = [0, 1]
    voxel_ndarray = (voxel_ndarray - np.min(voxel_ndarray)) * (rangeNorm[1] - rangeNorm[0]) / (
                np.max(voxel_ndarray) - np.min(voxel_ndarray))

    # Global equalization
    # voxel_ndarray = exposure.equalize_hist(voxel_ndarray)
    # print('after equalization shape', voxel_ndarray.shape)
    ##########adaptive equalization#########
    # voxel_ndarray = exposure.equalize_adapthist(voxel_ndarray)
    ########################################
    # Display results
    # fig = plt.figure()
    # plt.imshow(img[:,:,25], cmap='gray')
    # plt.imshow(ni_mask.get_fdata()[:,:,15], cmap='jet', alpha=0.5)
    # plt.show()
    #
    # ni_mask = nib.load(labelfile)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.imshow(ni_img.get_fdata()[:,:,0], cmap='gray')
    # plt.imshow(voxel_ndarray[:,:,25], cmap='gray')
    # plt.show()
    ###################################################

    # sort array
    newnparray = np.zeros(shape=voxel_ndarray.shape)
    for i in range(voxel_ndarray.shape[-1]):
        newnparray[:, :, voxel_ndarray.shape[-1] - 1 - i] = voxel_ndarray[:, :, i]

    return newnparray


def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg



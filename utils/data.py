from torch.utils.data import Dataset
import torch
import nibabel as nib
import numpy as np
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
import nibabel as nib
from ReadData.create_ID_list_selected_dataset import *
from ReadData.read_data import read_dicoms
import matplotlib.pyplot as plt
import matplotlib.colors
from torch.utils.data import DataLoader
from monai.transforms import Compose, ScaleIntensityd, RandFlipd, NormalizeIntensityd, RandCoarseDropoutd, RandScaleIntensityd, RandShiftIntensityd, ToTensord, RandRotated, RandZoomd, RandAdjustContrastd
from torchio import ZNormalization


########################################################################################################################
# Pre-processing transformation for training and evaluation/test

transform_training = Compose(
    [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotated(keys=["image", "label"], prob=0.5, range_x=(0.7, 0.7), padding_mode='zeros'),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1, padding_mode='constant'),
        # RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 4.5)),
        # RandCoarseDropoutd(keys=["image", "label"], prob=0.5, spatial_size=(15, 15), holes=10, fill_value=0),
        # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        NormalizeIntensityd(keys=["image"]),
        # RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.4),
        # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
        ToTensord(keys=["image", "label"]),
    ]
)

transform_test = Compose(
    [
        # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        NormalizeIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ]
)


########################################################################################################################
# Custom Dataset for MRPhysics and NAKO to pre-process data for dataloader

class MRPDataset(Dataset):
    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess

    def __getitem__(self, i):
        data = self.data[i]
        data["label"][data["label"] == 2] = 1
        # plt.imshow(data["image"].transpose((1, 2, 0)), cmap='gray')
        # plt.show()
        # plt.imshow(data["label"].transpose((1, 2, 0)), cmap='jet')
        # plt.show()
        transformed = self.preprocess(data)
        transformed_image = transformed["image"]
        transformed_target = transformed["label"]
        # plt.imshow(transformed_image.cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')
        # plt.show()
        # plt.imshow(transformed_target.cpu().detach().numpy().transpose((1, 2, 0)), cmap='jet')
        # plt.show()
        return transformed_image, transformed_target

    def __len__(self):
        return len(self.data)


########################################################################################################################
# data class, for padding, patching, splitting and saving the data

class Data:
    def __init__(self, cfg):
        self.batch_size = cfg['BatchSize']
        self.mode = cfg['Mode']
        self.database = cfg['Database']
        self.patchsize = cfg['PatchSize']
        self.patch_overlap_nako = cfg['PatchOverlap_NAKO']
        self.patch_overlap_mrp = cfg['PatchOverlap_MRP']
        self.ratio_testdata = cfg['Train_test_ratio']

        self.normalized_data_path = cfg['normalized_data_path']
        self.checkpoint = cfg['Checkpoint']
        self.selectedPatient = cfg['SelectedPatient']
        self.data_preprocessing = cfg['dataPreprocessing']

        self.cfg = cfg
        self.selectedDatasets = dict()

        for i, _ in enumerate(self.database):
            for j, selData in enumerate(cfg[self.database[i]]):

                if selData == 'datasets':
                    for k, data in enumerate(cfg[self.database[i]][selData]):
                        dat = cfg[self.database[i]][selData][data]

                        self.selectedDatasets.update(
                            {data: Dataset(data, dat[1], dat[2], dat[3], [dat[4], dat[5], dat[6]])})

        print('selectedDatasets', self.selectedDatasets)

        # --------------------------------------------------------------------------------------------------------------
        self.patchSizeX, self.patchSizeY = self.patchsize[0], self.patchsize[1]

        # # parsable list of data files
        self.new_get_ID_list()
        print(self.list_ID)
        self.n_samples = len(self.list_ID)

        if self.data_preprocessing:
            self.load_data()


    def new_get_ID_list(self):
        # create data file list, find required data files, and each file absolute path will be stored
        # in this list. This path list will be used in load_data function

        self.list_ID = []

        for database in self.cfg['Database']:
            path = self.cfg[database]['path']
            for pat in os.listdir(path):
                pat_path = path + os.sep + pat

                for dataset in self.cfg[database]['datasets']:
                    dataset_path = pat_path + os.sep + dataset
                    try:
                        for file in os.listdir(dataset_path):
                            if 'dicom_' in file:
                                print('dataset path', dataset_path)
                                self.list_ID.append(dataset_path + os.sep + file)
                    except:
                        print('no ', dataset, ' under path ', dataset_path)

        return self.list_ID


    def load_data(self):
        def preload(i):
            filepath = i
            print('filepath', filepath)

            # load data and store normalized data
            # if data is not stored before, we should load and preprocess those data
            image_array, label_array = self.load_file(
                filepath)  # load the whole patient -> for patient/volume-wise scaling
            # store preprocessed data
            self.store_normalized_data_motion_detection(image_array, label_array)

        print('loading original data')
        lids = len(self.list_ID)
        print('lids', lids)  # return file number
        Parallel(n_jobs=8)(delayed(preload)(i) for i in tqdm(self.list_ID))
        print('load_data done')


    def load_file(self, filepath):
        'load original data from .nii file for later normalization and preprocessing'
        # data extension specific loading
        self.currentDataDir = filepath
        splitted_filepath = filepath.split('/')
        # print(splitted_filepath)
        self.patient = splitted_filepath[-3]
        # print(self.patient)
        dataset = splitted_filepath[-2]
        # print(''dataset)
        self.dataset = self.selectedDatasets[dataset]
        # print(self.dataset)
        self.database = splitted_filepath[-5]
        # print(self.database)
        self.filename = splitted_filepath[-1].split('.')[0]
        # print(self.filename)

        image_voxelarray = read_dicoms(self.currentDataDir, self.patient, self.dataset,
                                       self.database)  # read real part and normalize the pixels
        label_path = os.path.abspath(os.path.join(self.currentDataDir, os.pardir))

        for file in os.listdir(label_path):
            if 'label_' in file:
                print('data path of image', self.currentDataDir)
                print('data path of label', label_path + os.sep + file)
                label_voxelarray = nib.load(label_path + os.sep + file)

                label_voxelarray = np.swapaxes(np.array(label_voxelarray.get_fdata()), 0, 1)

        print('image size', image_voxelarray.shape)
        print('label shape', label_voxelarray.shape)

        return image_voxelarray, label_voxelarray


    def store_normalized_data_motion_detection(self, image_array, label_array):
        # only for image-based motion detection
        'store the loaded and normalized data for later operation'
        if not os.path.exists(self.normalized_data_path):
            try:
                os.makedirs(self.normalized_data_path)
            except:
                pass
            # os.mkdir(self.normalized_data_path)

        normalized_path = self.normalized_data_path + os.sep + self.database + '_' + \
                          self.patient + '_' + self.dataset.pathdata + '_' + self.filename
        print('normalized data path', normalized_path)

        with h5py.File(normalized_path + ".hdf5", 'w') as hf:
            hf.create_dataset('image', data=image_array)
            hf.create_dataset('label', data=label_array)
            hf.close()


    def fetch_image_data_from_ID(self, ID):
        # here ID is file path of the original data
        # print('ID', ID)
        splitted_ID = ID.split('/')

        f = splitted_ID[-1].split('.')

        filename = splitted_ID[-5] + '_' + splitted_ID[-3] + '_' + splitted_ID[-2] + '_' + \
                   f[0] + '.hdf5'  # highly storage method dependent

        filepath = self.normalized_data_path + os.sep + filename

        with h5py.File(filepath, 'r') as hf:
            X = np.array(hf.get('image'))
            Y = np.array(hf.get('label'))

            hf.close()

        return X, Y


    def process_image_data(self, list_ID):
        # for image motion detection NN
        # patching images and create a list for all patches
        def process(ID):
            # load hdf5 format data
            img, label = self.fetch_image_data_from_ID(ID)
            # converge the label to int
            label = np.round(label).astype(int)

            # fig = plt.figure()
            # plt.imshow(img[:,:,20], cmap='gray')
            # plt.imshow(label[:,:,20], cmap='jet', alpha=0.4)
            # plt.show()

            image_patches, label_patches = self.patching(img, label, ID)

            if self.cfg['Mode'] == 'training':
                return image_patches, label_patches
                # return image_patches_ones, label_patches_ones
            if self.cfg['Mode'] == 'prediction':
                return image_patches, label_patches, img, label

        ############just for test NN###########
        # list_ID = list_ID[0:1]  #when use prediction, select an image to predict
        # print('list_IDs', list_ID)
        #######################################

        # patching data before training and create a patch list for image and label

        if self.cfg['Mode'] == 'training':
            images, labels = zip(*Parallel(n_jobs=1)(delayed(process)(i) for i in tqdm(list_ID)))
        if self.cfg['Mode'] == 'prediction':
            images, labels, img, label = zip(*Parallel(n_jobs=1)(delayed(process)(i) for i in tqdm(list_ID)))

        images = np.vstack(images)
        labels = np.vstack(labels)

        if self.cfg['Mode'] == 'training':
            return images, labels
        if self.cfg['Mode'] == 'prediction':
            return images, labels, img, label


    def padding(self, image, label, ID):
        # padding the input image to get exact size for patching, only 2D

        str = 'NAKO'
        if str in ID:
            stride = int(self.patchSizeX * (1 - self.patch_overlap_nako))
        else:
            stride = int(self.patchSizeX * (1 - self.patch_overlap_mrp))
        numberPatchesX = int(max((image.shape[0] - self.patchSizeX), 1) / stride + 2)
        numberPatchesY = int(max((image.shape[1] - self.patchSizeY), 1) / stride + 2)
        paddingX = int(((numberPatchesX - 1) * stride - image.shape[0] + self.patchSizeX) + 1)
        paddingY = int(((numberPatchesY - 1) * stride - image.shape[1] + self.patchSizeY) + 1)
        if (paddingX % 2):
            paddingX += 1
        if (paddingY % 2):
            paddingY += 1
        padding = ((int(paddingX / 2), int(paddingX / 2)), (int(paddingY / 2), int(paddingY / 2)), (int(0), int(0)))
        image_pad = np.pad(image, padding)
        label_pad = np.pad(label, padding)

        return image_pad, label_pad


    def patching(self, images, labels, ID):
        # patching of one image slice, extract n patches dependent on overlap and patch size, only 2D

        image_list = []
        label_list = []
        base_tensor_list = []
        t_size_list = []
        dicom_depth = images.shape[2]
        if labels.shape[2] < images.shape[2]:
            dicom_depth = labels.shape[2]
        for i in range(dicom_depth):
            images_pad, labels_pad = self.padding(np.expand_dims(images[:, :, i], 2), np.expand_dims(labels[:, :, i], 2), ID)
            images_t, labels_t = torch.Tensor(images_pad), torch.Tensor(labels_pad)
            images_t, labels_t = images_t.unsqueeze(0).permute((0, 3, 1, 2)), labels_t.unsqueeze(0).permute((0, 3, 1, 2))

            image_patches, label_patches, mask_p, base_tensor, label_base, t_size = self.split_tensor(images_t, labels_t, ID)

            image_patches = np.stack(image_patches)
            label_patches = np.stack(label_patches)
            image_list.append(image_patches)
            label_list.append(label_patches)

        image_patches_list = np.vstack(image_list)
        label_patches_list = np.vstack(label_list)

        return image_patches_list, label_patches_list


    def split_tensor(self, image, label, ID, tile_size=224):
        # split the image slices into patches, only 2D

        mask = torch.ones_like(image)
        # use torch.nn.Unfold
        str = 'NAKO'
        if str in ID:
            stride = int(self.patchSizeX * (1 - self.patch_overlap_nako))
        else:
            stride = int(self.patchSizeX * (1 - self.patch_overlap_mrp))
        unfold = torch.nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
        # Apply to mask and original image
        mask_p = unfold(mask)
        image_patches = unfold(image)
        label_patches = unfold(label)

        image_patches = image_patches.reshape(1, tile_size, tile_size, -1).permute(3, 0, 1, 2)
        label_patches = label_patches.reshape(1, tile_size, tile_size, -1).permute(3, 0, 1, 2)

        if image.is_cuda:
            patches_base = torch.zeros(image_patches.size(), device=image.get_device())
            label_base = torch.zeros(label_patches.size(), device=label.get_device())
        else:
            patches_base = torch.zeros(image_patches.size())
            label_base = torch.zeros(label_patches.size())

        label_patches[label_patches == 2] = 1

        image_tiles = []
        for i in range(image_patches.size(0)):
            image_tiles.append(image_patches[i, :, :, :])

        label_tiles = []
        for i in range(image_patches.size(0)):
            label_tiles.append(label_patches[i, :, :, :])

        return image_tiles, label_tiles, mask_p, patches_base, label_base, (image.size(2), image.size(3))


    def rebuild_tensor(self, tensor_list, label_list, mask_t, base_tensor, label_base, t_size, ID, tile_size=224):
        # rebuild image slices from patches to visualize output segmentation mask, only 2D

        str = 'NAKO'
        if str in ID:
            stride = int(self.patchSizeX * (1 - self.patch_overlap_nako))
        else:
            stride = int(self.patchSizeX * (1 - self.patch_overlap_mrp))

        for t, tile in enumerate(tensor_list):
            base_tensor[[t], :, :] = tile

        for t, tile in enumerate(label_list):
            label_base[[t], :, :] = tile.type(torch.FloatTensor)

        base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(1 * tile_size * tile_size, base_tensor.size(0)).unsqueeze(0)
        label_base = label_base.permute(1, 2, 3, 0).reshape(1 * tile_size * tile_size, label_base.size(0)).unsqueeze(0)
        fold = torch.nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
        output_tensor = fold(base_tensor) / fold(mask_t)
        output_label = fold(label_base) / fold(mask_t)

        return output_tensor, output_label


    def create_image_space_train_dataset(self, data):
        # creation of training dataset
        dataset_local_label = MRPDataset(data, transform_training)
        return dataset_local_label


    def create_image_space_test_dataset(self, data):
        # creation of validation/test dataset
        dataset_local_label = MRPDataset(data, transform_test)
        return dataset_local_label


    def plot_patient(self, image_list, label_list, partition, model, device):
        # plot image slice, ground truth segmentation mask and prediction segmentation mask as hard mask and probability
        # heat map
        if self.database[0] == 'NAKO194':
            image_list = []
            label_list = []
            patient = []
            for file in partition['test']:
                img = nib.load(file)
                image = np.array(img.dataobj)
                label = np.zeros((320, 260, 316))
                image.astype(float)
                image_list.append(image)
                label_list.append(label)
                split = file.split('/')
                split = split[-1]
                splitted_filepath = split.split('_')
                sub = splitted_filepath[0]
                sub = str(sub)[1:]
                patient.append(sub)

        # UBK patients (plotting)
        # patient = []
        # img = nib.load('/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw/1006431/fat.nii.gz')
        # image = np.array(img.dataobj)
        # label = np.zeros((224, 168, 363))
        # image.astype(float)
        # image_list.append(image)
        # label_list.append(label)
        # img = nib.load('/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw/3016406/fat.nii.gz')
        # image = np.array(img.dataobj)
        # label = np.zeros((224, 168, 363))
        # image.astype(float)
        # image_list.append(image)
        # label_list.append(label)
        # img = nib.load('/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw/3373770/fat.nii.gz')
        # image = np.array(img.dataobj)
        # label = np.zeros((224, 168, 363))
        # image.astype(float)
        # image_list.append(image)
        # label_list.append(label)
        # img = nib.load('/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw/4202972/fat.nii.gz')
        # image = np.array(img.dataobj)
        # label = np.zeros((224, 168, 363))
        # image.astype(float)
        # image_list.append(image)
        # label_list.append(label)
        # img = nib.load('/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw/5603553/fat.nii.gz')
        # image = np.array(img.dataobj)
        # label = np.zeros((224, 168, 363))
        # image.astype(float)
        # image_list.append(image)
        # label_list.append(label)
        # patient.append('1006431')
        # patient.append('3016406')
        # patient.append('3373770')
        # patient.append('4202972')
        # patient.append('5603553')

        for idx in range(len(image_list)):
            dataset = partition['test'][idx].split('/')[-2]
            pat = partition['test'][idx].split('/')[-3]
            if self.database[0] == 'NAKO194':
                ID = 'NAKO'
            else:
                ID = partition['test'][idx]

            for i in range(image_list[idx].shape[2]):
                # please choose which slices you want to plot and save ans png image
                #if i==3 or i == 16 or i==28 or i == 47 or i==59 or i == 70 or i == 82 or i==89 or i == 96 or i == 105 or i == 113 or i == 123 or i==130 or i == 137 or i==148 or i==154 or i==161 or i==179 or i==190 or i == 206:
                if i == 3 or i == 28 or i == 47 or i == 70 or i == 89 or i == 96 or i == 105 or i == 123 or i == 137 or i==140 or i == 148 or i == 154 or i == 161 or i==168 or i == 179 or i==185 or i == 190 or i==200 or i == 206 or i==111 or i==118 or i==223 or i==228 or i==235 or i==240 or i==248 or i == 256 or i==264 or i==270 or i==277 or i==284 or i==290 or i==300 or i==314:
                #if i == 3 or i == 28 or i == 47 or i == 70 or i == 89 or i == 96 or i == 105 or i == 123 or i == 137 or i == 140 or i == 148 or i == 154 or i == 161 or i == 168 or i == 179 or i == 185 or i == 190 or i == 200 or i == 206 or i == 111 or i == 118 or i == 223 or i == 228 or i == 235 or i == 240 or i == 248 or i == 256 or i == 264 or i == 270 or i == 277 or i == 284 or i == 290 or i == 300 or i==308 or i == 314 or i==320 or i==328 or i==337 or i==351:
                #if i == 3 or i == 15 or i == 27:
                    image_pad, label_pad = self.padding(np.expand_dims(image_list[idx][:, :, i], 2),
                                                        np.expand_dims(label_list[idx][:, :, i], 2), ID)

                    if self.database[0] == 'NAKO194':
                        image_pad = image_pad.astype('float64')
                        rangeNorm = [0, 1]
                        image_pad = (image_pad - np.min(image_pad)) * (rangeNorm[1] - rangeNorm[0]) / (
                            np.max(image_pad) - np.min(image_pad))

                    image_t, label_t = torch.Tensor(image_pad), torch.Tensor(label_pad)
                    image_t, label_t = image_t.unsqueeze(0).permute((0, 3, 1, 2)), label_t.unsqueeze(0).permute(
                        (0, 3, 1, 2))

                    if self.database[0] == 'NAKO194':
                        image_t = torch.rot90(image_t, k=3, dims=[2, 3])
                        label_t = torch.rot90(label_t, k=3, dims=[2, 3])

                    image_patches, label_patches, mask_p, base_tensor, label_base, t_size = self.split_tensor(image_t,
                                                                                                              label_t,
                                                                                                              ID)
                    image_patches_stack = torch.stack(image_patches).to(device)
                    label_patches_stack = torch.stack(label_patches).to(device)

                    test_data = [{"image": img, "label": label} for img, label in
                                 zip(image_patches_stack, label_patches_stack)]
                    test_dataset_local = self.create_image_space_test_dataset(test_data)
                    test_loader = DataLoader(test_dataset_local, batch_size=len(test_dataset_local), shuffle=False,
                                             num_workers=0)

                    model.eval()
                    for images, labels in tqdm(test_loader):
                        with torch.no_grad():
                            images, labels = images.to(device), labels.to(device)
                            output_one_hot = model(images)
                            output_one_hot = torch.nn.functional.softmax(output_one_hot)

                    output = torch.argmax(output_one_hot, dim=1)
                    output_probs = output_one_hot.data[:, 1, :, :]

                    image_rebuild, label_rebuild = self.rebuild_tensor(image_patches, label_patches, mask_p,
                                                                       base_tensor, label_base, t_size, ID)
                    _, prediction_rebuild = self.rebuild_tensor(image_patches, output, mask_p, base_tensor, label_base,
                                                                t_size, ID)
                    _, prediction_rebuild_probs = self.rebuild_tensor(image_patches, output_probs, mask_p, base_tensor,
                                                                      label_base, t_size, ID)

                    cmap = matplotlib.colors.ListedColormap(['darkgreen', 'darkred'])
                    c = ["darkgreen", "green", "yellowgreen", "greenyellow", "yellow", "gold", "orange", "darkorange",
                         "orangered", "red", "darkred"]
                    v = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
                    l = list(zip(v, c))
                    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('rg', l, N=256)
                    # label_rebuild[label_rebuild != label_rebuild] = 0
                    f, axarr = plt.subplots(2, 2)
                    im1 = axarr[0, 0].imshow(image_rebuild.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)),
                                             cmap='gray', vmin=0.0, vmax=1.0)
                    axarr[0, 1].axis('off')
                    if self.database[0] != 'NAKO194':
                        axarr[0, 1].imshow(image_rebuild.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)),
                                        cmap='gray', vmin=0.0, vmax=1.0)
                        im2 = axarr[0, 1].imshow(label_rebuild.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)),
                                                cmap=cmap, alpha=1.0, vmin=0.0, vmax=1.0)
                    axarr[1, 0].imshow(image_rebuild.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)),
                                      cmap='gray', vmin=0.0, vmax=1.0)
                    im3 = axarr[1, 0].imshow(prediction_rebuild.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)),
                                             cmap=cmap, alpha=1.0, vmin=0.0, vmax=1.0)
                    axarr[1, 1].imshow(image_rebuild.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)),
                                       cmap='gray', vmin=0.0, vmax=1.0)
                    im4 = axarr[1, 1].imshow(
                        prediction_rebuild_probs.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0)), cmap=cmap2,
                        alpha=1.0, interpolation='bilinear', vmin=0.0, vmax=1.0)
                    if self.database[0] != 'NAKO194':
                        f.suptitle('patient: ' + str(pat) + ' - dataset: ' + str(
                            dataset) + ' - ' + 'slice: ' + str(i), fontsize=12)
                    else:
                        f.suptitle('NAKO194_patient: ' + str(patient[idx]) + ' - ' + 'slice: ' + str(i), fontsize=12)
                    f.tight_layout()
                    plt.colorbar(im1, ax=axarr[0, 0])
                    axarr[0, 0].title.set_text('image')
                    if self.database[0] != 'NAKO194':
                        plt.colorbar(im2, ax=axarr[0, 1])
                        axarr[0, 1].title.set_text('ground truth')
                    plt.colorbar(im3, ax=axarr[1, 0])
                    axarr[1, 0].title.set_text('hard prediction mask')
                    plt.colorbar(im4, ax=axarr[1, 1])
                    axarr[1, 1].title.set_text('probability heat map')
                    if self.database[0] != 'NAKO194':
                        plt.savefig('/home/students/studborst1/media/figures/Supervised/Trained_on_NAKO/Evaluated_on_NAKO/patient_' + str(pat) + '_dataset_' + str(
                            dataset) + '_' + 'slice_' + str(i))
                    else:
                        plt.savefig('/home/students/studborst1/media/figures/Supervised/Trained_on_MRPhysics_NAKO/Evaluated_on_NAKO194/NAKO194_patient_' + str(patient[idx]) + '_' + 'slice_' + str(i))
                    plt.show()


class Dataset:
    def __init__(self, pathdata, artefact=None, bodyregion=None, tWeighting=None, imageShape=None):
        'created class dataset for store image information like bodyregion...'
        self.pathdata = pathdata
        self.artefact = artefact
        self.bodyregion = bodyregion
        self.mrtWeighting = tWeighting
        self.imageShape = imageShape

    def getPathdata(self):
        return self.pathdata

    def getArtefact(self):
        return self.artefact

    def getBodyRegion(self):
        return self.bodyregion

    def getMRTWeighting(self):
        return self.mrtWeighting

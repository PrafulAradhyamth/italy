import os    

def create_list_ID_training(cfg):
    #create a directory to store train and validation/test data ID (absolute path)
    partition = {}
    train_ID_list = []
    test_ID_list = []

    #here, I use file_path as ID
    #TODO: use it as a function

    if cfg['Database'][0] != 'NAKO194':
        for database in cfg['Database']:
            path = cfg[database]['path']
            for pat in os.listdir(path):
                pat_path = path + os.sep + pat
                if pat in cfg['SelectedPatient']:
                    # print('test pat', pat)

                    for dataset in cfg[database]['datasets']:
                        dataset_path = pat_path + os.sep + dataset
                        try:
                            for file in os.listdir(dataset_path):
                                if 'dicom_' in file:
                                    # print(file)

                                    test_ID_list.append(dataset_path+os.sep+file)
                        except:

                            print('no ', dataset, ' under path ', dataset_path)
                else:
                    for dataset in cfg[database]['datasets']:
                        dataset_path = pat_path + os.sep + dataset
                        try:
                            for file in os.listdir(dataset_path):
                                if 'dicom_' in file:

                                    train_ID_list.append(dataset_path+os.sep+file)
                        except:
                             print('no ', dataset, ' under path ', dataset_path)

    else:
        for database in cfg['Database']:

            path = cfg[database]['path']
            for pat in os.listdir(path):
                pat_path = path + os.sep + pat
                for overall_dataset in os.listdir(pat_path):
                    pat_path = pat_path + os.sep + overall_dataset
                    sub_list = []
                    for subject in cfg['SelectedPatient']:
                        for file in os.listdir(pat_path):
                            splitted_filepath = file.split('_')
                            sub = splitted_filepath[0]
                            sub = str(sub)[1:]
                            subject = str(subject).zfill(5)
                            if sub == subject:
                                filepath = pat_path + os.sep + file
                                sub_list.append(filepath)

                    for dataset in cfg[database]['datasets']:
                        splitted_dataset = dataset.split('_')
                        dat = splitted_dataset[-1]
                        splitted_dataset = dat.split('.')
                        dat = splitted_dataset[0]
                        for i in sub_list:
                            split = i.split('/')
                            split = split[-1]
                            split = split.split('_')
                            split = split[-1]
                            split = split.split('.')
                            split = int(split[0])

                            if split == 0:
                                seq = 'FAT'
                            elif split == 1:
                                seq = 'WATER'
                            elif split == 2:
                                seq = 'IN'
                            else:
                                seq = 'OOP'

                            if dat == seq:
                                test_ID_list.append(i)

    partition.update({'train': train_ID_list})
    partition.update({'test': test_ID_list})
    return partition

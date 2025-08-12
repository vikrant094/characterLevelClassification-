import os
from characterLevelClassification.logging import logger
from characterLevelClassification.entity import DataTransformationConfig
from characterLevelClassification.utils.common import (lineToTensor,
                                                      get_device )



from io import open
import glob
import time
import torch
import os
from torch.utils.data import (Dataset,
                              random_split
                            )

class NamesDataset(Dataset):

    def __init__(
                    self,
                    data,
                    data_tensors,
                    labels,
                    labels_tensors):
        
        self.load_time = time.localtime #for provenance of the dataset

        self.data = data
        self.data_tensors = data_tensors
        self.labels = labels
        self.labels_tensors = labels_tensors
        self.labels_uniq = list(set(labels))
        # #read all the ``.txt`` files in the specified directory
        # text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        # for filename in text_files:
        #     label = os.path.splitext(os.path.basename(filename))[0]
        #     labels_set.add(label)
        #     lines = open(filename, encoding='utf-8').read().strip().split('\n')
        #     for name in lines:
        #         self.data.append(name)
        #         self.data_tensors.append(lineToTensor(name))
        #         self.labels.append(label)

        # #Cache the tensor representation of the labels
        # self.labels_uniq = list(labels_set)
        # for idx in range(len(self.labels)):
        #     temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
        #     self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

 
    def get_transformed_data(self, data_dir):

        self.data_dir = data_dir #for provenance of the dataset
        self.load_time = time.localtime #for provenance of the dataset
        labels_set = set() #set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []
        #read all the ``.txt`` files in the specified directory
        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)

        #Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

        transformed_dataset = [self.data,self.data_tensors, self.labels,self.labels_tensors  ]
        
        validation_status = False
        try:
            torch.save(transformed_dataset, os.path.join(self.config.root_dir, 'dataset.pt')) 
            validation_status = True
        except Exception as e:
            raise e
        
        with open(self.config.STATUS_FILE, 'w') as f:
            f.write(f"Validation status: {validation_status}")



        





    


    
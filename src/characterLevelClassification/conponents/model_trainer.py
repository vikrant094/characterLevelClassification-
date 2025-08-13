import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from characterLevelClassification.entity import ModelTrainerConfig
from characterLevelClassification.constants import ALLOWED_CHARACTERS
from characterLevelClassification.logging import logger
from characterLevelClassification.conponents.data_transformation import NamesDataset
from characterLevelClassification.utils.common import get_device 
from torch.utils.data import random_split
from characterLevelClassification.utils.common import label_from_output
from characterLevelClassification.utils.common import plot_result
from operator import truediv
import numpy as np
import pandas as pd

class CharRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear  = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)


    def forward(self, line_tensor):

        rnn_out, hidden = self.rnn(line_tensor)
        output = self.linear(hidden[0])
        output = self.softmax(output)
        return output


        
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def get_model(self,output_size):
        model = CharRNN(len(ALLOWED_CHARACTERS), self.config.n_hidden, output_size )
        logger.info(f'model architecture :: {model}')
        return model
    
    def get_train_test_data(self,dataset, test_prct= 0.2):

        device = get_device()
        logger.info(f">>>>>> device :: {device}  <<<<<<\n\nx==========x")
        train_set, test_set = random_split(dataset, [1-test_prct,test_prct], generator=torch.Generator(device=device).manual_seed(2024))
        logger.info(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")
        return train_set, test_set
    
    def load_data(self):

        
        data,data_tensors,labels,labels_tensors = torch.load(os.path.join('artifacts','data_transformation', 'dataset.pt'))
        all_data  = NamesDataset(data,data_tensors,labels,labels_tensors)
        train_set, test_set = self.get_train_test_data(all_data, test_prct=self.config.test_prct)
        return all_data, train_set, test_set


    def train(
            self,
            model, 
            training_data,
            n_epoch = 10,
            n_batch_size =64, 
            report_every = 1, 
            learning_rate = 0.2, 
            criterion = nn.NLLLoss()
            ):
        n_epoch = self.config.num_train_epochs
        n_batch_size = self.config.per_device_train_batch_size
        
        current_loss = 0
        all_losses = []
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        

        start = time.time()

        logger.info(f"training on data set with n = {len(training_data)}")

        for iter in range(1, n_epoch + 1):
            model.zero_grad() # clear the gradients

            # create some minibatches
             # we cannot use dataloaders because each of our names is a different length
            batches = list(range(len(training_data)))
            random.shuffle(batches)
            batches = np.array_split(batches, len(batches) //n_batch_size )

            for idx, batch in enumerate(batches):
                batch_loss = 0
                for i in batch: #for each example in this batch
                    (label_tensor, text_tensor, label, text) = training_data[i]
                    output = model.forward(text_tensor)
                    loss = criterion(output, label_tensor)
                    batch_loss += loss

                # optimize parameters
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                optimizer.zero_grad()

                current_loss += batch_loss.item() / len(batch)

            all_losses.append(current_loss / len(batches) )
            if iter % report_every == 0:
                logger.info(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
            current_loss = 0
        self.all_losses = all_losses
        torch.save(model.state_dict(),self.config.model_ckpt)
        plot_result(self.all_losses, os.path.join(self.config.root_dir, 'training_loss.png'))

    

        
        
        









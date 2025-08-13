
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from characterLevelClassification.entity import ModelEvaluationConfig
from characterLevelClassification.conponents.model_trainer import ModelTrainer
from characterLevelClassification.utils.common import label_from_output
from operator import truediv
import numpy as np

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def get_metrics(self, cm, classes):

        metrics = pd.DataFrame(columns = ["metric"] + classes)

        sp = []
        f1 = []
        gm = []
        sens = []
        acc= []
        final = []
        for c in range(len(classes)):
            tp = cm[c,c]
            fp = sum(cm[:,c]) - cm[c,c]
            fn = sum(cm[c,:]) - cm[c,c]
            tn = sum(np.delete(sum(cm)-cm[c,:],c))

            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            accuracy = (tp+tn)/(tp+fp+fn+tn)
            specificity = tn/(tn+fp)
            f1_score = 2*((precision*recall)/(precision+recall))
            g_mean = np.sqrt(recall * specificity)
            sp.append(specificity)
            f1.append(f1_score)
            gm.append(g_mean)
            sens.append(recall)
            acc.append(accuracy)

        metrics["metric"] = ["recall", "specificity", "accuracy", "f1", "gmean"]
        for index, class_ in enumerate(classes):
            metrics[class_] = [sens[index], sp[index], acc[index], f1[index], gm[index]]
        
        metrics.to_csv(os.path.join(self.config.root_dir, 'metrics.csv'))




    def evaluate(self, model, testing_data, classes):


        confusion = torch.zeros(len(classes), len(classes))

        model.eval() #set to eval mode
        with torch.no_grad(): # do not record the gradients during eval phase
            for i in range(len(testing_data)):
                (label_tensor, text_tensor, label, text) = testing_data[i]
                output = model(text_tensor)
                guess, guess_i = label_from_output(output, classes)
                label_i = classes.index(label)
                confusion[label_i][guess_i] += 1
        cm = confusion.clone().numpy()
        # Normalize by dividing every row by its sum
        for i in range(len(classes)):
            denom = confusion[i].sum()
            if denom > 0:
                confusion[i] = confusion[i] / denom

        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
        ax.set_yticks(np.arange(len(classes)), labels=classes)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # sphinx_gallery_thumbnail_number = 2
        plt.savefig(os.path.join(self.config.root_dir, 'confusion_matrix.png'))
        df = pd.DataFrame(cm).astype(int)
        df.to_csv(os.path.join(self.config.root_dir, 'confusion_matrix.csv'))
        self.get_metrics(cm, classes)


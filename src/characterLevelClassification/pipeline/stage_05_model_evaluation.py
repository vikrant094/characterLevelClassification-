from characterLevelClassification.config.configuration import ConfigurationManager
from characterLevelClassification.conponents.data_transformation import DataTransformation
from characterLevelClassification.conponents.model_trainer import ModelTrainer
from characterLevelClassification.conponents.model_evaluation import ModelEvaluation
from characterLevelClassification.logging import logger
import torch
import os



class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        ### initiate evaluation
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        #### get the classes
        data_transformation_config = config.get_data_transformation_config()
        classes = data_transformation_config.classes
        number_of_classes = len(classes)

        #### get the evaluation data
        model_trainer_config = config.get_model_trainer_config()
        train_obj = ModelTrainer(config=model_trainer_config)
        all_data, train_set, test_set = train_obj.load_data()

        #### get the model
        model = train_obj.get_model(number_of_classes)
        model.load_state_dict(torch.load(model_evaluation_config.model_path, weights_only=True))


        
        model_evaluation.evaluate(model, test_set, classes=classes)
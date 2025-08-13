from characterLevelClassification.config.configuration import ConfigurationManager
from characterLevelClassification.conponents.model_trainer import ModelTrainer
from characterLevelClassification.logging import logger

import time
class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        train_obj = ModelTrainer(config=model_trainer_config)
        all_data, train_set, test_set = train_obj.load_data()
        number_of_classes = len(all_data.labels_uniq)
        model = train_obj.get_model(number_of_classes)
        start = time.time()
        train_obj.train(model, train_set)
        end = time.time()
        logger.info(f"training took {end-start}s")

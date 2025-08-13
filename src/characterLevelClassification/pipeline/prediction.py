from characterLevelClassification.config.configuration import ConfigurationManager
from characterLevelClassification.conponents.model_trainer import ModelTrainer
import torch
from characterLevelClassification.utils.common import (lineToTensor,
                                                       label_from_output,
                                                      get_device )

class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()

        data_transformation_config = config.get_data_transformation_config()
        self.classes = data_transformation_config.classes
        self.number_of_classes = len(self.classes)

        model_trainer_config = config.get_model_trainer_config()
        train_obj = ModelTrainer(config=model_trainer_config)
        #### get the model
        self.model = train_obj.get_model(self.number_of_classes)
        self.model.load_state_dict(torch.load(eval_config.model_path, weights_only=True))



    def predict(self, name):

        input = name.strip().lower()
        input_tensor = lineToTensor(input)
        output =self.model(input_tensor)
        predict_class, predict_class_index = label_from_output(output, self.classes)        
        return predict_class
    
if __name__ == "__main__":
    name  = "vikrant"
    pred_obj = PredictionPipeline()
    predict_class = pred_obj.predict(name)

from characterLevelClassification.config.configuration import ConfigurationManager
from characterLevelClassification.conponents.data_transformation import DataTransformation
from characterLevelClassification.logging import logger
import os

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.get_transformed_data(os.path.join("artifacts","data_ingestion","data/names"))
        except Exception as e:
            raise e
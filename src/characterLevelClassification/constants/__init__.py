from pathlib import Path
import string

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
ALLOWED_CHARACTERS = string.ascii_letters + " .,;'" + "_"
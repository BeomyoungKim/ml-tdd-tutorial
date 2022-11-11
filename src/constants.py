import os


ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR_PATH = os.path.join(ROOT_DIR, 'data')
MODEL_DIR_PATH = os.path.join(ROOT_DIR, 'models')

RAW_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, 'raw')
PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, 'processed')

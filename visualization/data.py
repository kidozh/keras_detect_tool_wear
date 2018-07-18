import os.path

KERAS_MODEL_PATH = "%s/PHM_regression/residual_logs_adam_adjust_kernel_depth_20_shuffle_manually.kerasmodel" %(os.path.dirname(os.path.dirname(__file__)),)

from keras.models import load_model

def get_model():
    return load_model(KERAS_MODEL_PATH)
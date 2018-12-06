import os.path

KERAS_MODEL_NAME = "RUL_TRANSVERSE_PYRAMID_DEPTH_20_LR_0.01"

KERAS_MODEL_PATH = "%s/visualization/model_h5/%s.kerasmodel" %(os.path.dirname(os.path.dirname(__file__)),KERAS_MODEL_NAME)

# KERAS_MODEL_NAME = "residual_logs_adam_adjust_kernel_depth_20_shuffle_manually"
#
# KERAS_MODEL_PATH = "%s/PHM_regression/%s.kerasmodel" %(os.path.dirname(os.path.dirname(__file__)),KERAS_MODEL_NAME)

from keras.models import load_model

def get_model():
    return load_model(KERAS_MODEL_PATH)
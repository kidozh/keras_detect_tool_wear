from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
KERAS_MODEL_NAME = "TRANS_classic_modl_regression_depth_18_0.1_conv_swish"
model = load_model(KERAS_MODEL_NAME)

print(model.summary())
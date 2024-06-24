# License
#  This program is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published 
#  by the Free Software Foundation, either version 3 of the License, 
#  or (at your option) any later version.

#  This program is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

#  See the GNU General Public License for more details. You should have 
#  received a copy of the GNU General Public License along with this 
#  program. If not, see <https://www.gnu.org/licenses/>. 

# Description
#  Linear isotropic Hookean law. Implemented via a TensorFlow neural network 
#  model. The inputs are 3 strains, the outputs are 3 stresses.

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import tensorflow as tf
from tensorflow import keras
from joblib import load

ML_model= keras.models.load_model("ML_model.h5")

#Load the scalers
x_scaler = load('x_scaler.joblib')
y_scaler = load('y_scaler.joblib')

def predict():
    epsilon_scaled = x_scaler.transform(epsilon)
    sigma_scaled = ML_model.predict(epsilon_scaled[np.newaxis, :, :])
    sigma[:, :] = y_scaler.inverse_transform(sigma_scaled.reshape(epsilon_scaled.shape))

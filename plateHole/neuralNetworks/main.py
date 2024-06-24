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
#  This routine trains a neural network with linear elastic data, where the 
#  strains are taken from a simulation that used the Hookean law. 

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import numpy as np
from distutils.dir_util import mkpath
from functions import *
from input_file import *
from sklearn.preprocessing import MinMaxScaler
import random
from joblib import dump
import pickle

# Seed everything 
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create the scalers 
x_scaler =  MinMaxScaler()
y_scaler  =  MinMaxScaler()    

# Create the dataset 
# Read the expected strains
with open('original_x_training_set.npy', 'rb') as f:
    original_x_training_set = np.load(f)
f.close()

# Calculate stresses 
#Initialise original_y_training_set
original_y_training_set = np.zeros(original_x_training_set.shape)

#Populate original_y_training_set
original_y_training_set[:, 0] = 2 * LAME_2 * original_x_training_set[:, 0] \
    + LAME_1 * (original_x_training_set[:, 0] \
    + original_x_training_set[:, 3] + original_x_training_set[:, 5])
        
original_y_training_set[:, 1] = 2 * LAME_2 * original_x_training_set[:, 1]

original_y_training_set[:, 2] = 2 * LAME_2 * original_x_training_set[:, 2]

original_y_training_set[:, 3] = 2 * LAME_2 * original_x_training_set[:, 3] \
    + LAME_1 * (original_x_training_set[:, 0] \
    + original_x_training_set[:, 3] + original_x_training_set[:, 5])
        
original_y_training_set[:, 4] = 2 * LAME_2 * original_x_training_set[:, 4]

original_y_training_set[:, 5] = 2 * LAME_2 * original_x_training_set[:, 5] \
    + LAME_1 * (original_x_training_set[:, 0] \
    + original_x_training_set[:, 3] + original_x_training_set[:, 5])

x_train = original_x_training_set
y_train = original_y_training_set

x_scaler.fit(x_train)
y_scaler.fit(y_train)

x_train_normalised = x_scaler.transform(x_train)
y_train_normalised = y_scaler.transform(y_train)
        
if (SUBSAMPLE_ORIGINAL_STRAINS):
    idx = np.random.randint(0, int(x_train.shape[0]), 
                            size = int(0.01 * x_train.shape[0]))

# Create the ML_Model 
ML_model = createNN(ML_MODEL_IS_3X3, 7)
compileNN(ML_model, False)       
history = ML_model.fit(x_train_normalised[idx, np.newaxis, :], 
               y_train_normalised[idx, np.newaxis, :], 
                epochs = NUMBER_OF_EPOCHS, 
                validation_split=0.01)

x_train = x_train[idx, :]
y_train = y_train[idx, :]

# Save the ML model
ML_model.save("ML_model.h5")

# Serialise the dataset and the scalers
with open('x_train.npy', 'wb') as f:
    pickle.dump(x_train, f)    
f.close()
with open('y_train.npy', 'wb') as f:
    pickle.dump(y_train, f)
f.close()

# Serialise the scalers
dump(x_scaler, 'x_scaler.joblib')
dump(y_scaler, 'y_scaler.joblib')
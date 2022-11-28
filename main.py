import pandas as pd
import numpy as np
import preprocessing as p
import matplotlib.pyplot as plt


features = ["Latitude", "Longitude"]
res_var = ["Speed Array"]


# Loads data and splits into train valid and test
data = pd.read_excel('test.xlsx')
X, Y = p.load_data(data, max_depth=10, sampling_rate=0.1)
X_train, X_valid, X_test = p.split_data(X, 0.7, 0.15)
Y_train, Y_valid, Y_test = p.split_data(Y, 0.7, 0.15)



""" Feed Forward Neural Network """

# Makes tuples with train validate and test data
train_tuple=(X_train, Y_train)
valid_tuple=(X_valid, Y_valid)
test_tuple=(X_test, Y_test)

# Optimizes the hyperparameters on the validation data
best_path = "/Users/edvardronglan/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/2022 Fall semester/1.125 Software Engineering/Project/python/datafiles/mlp_model"
NN, best_hp =p.optimize_hyperparams(dir=best_path,
                                      train_tuple=train_tuple,
                                      valid_tuple=valid_tuple)

# Predicts the values for the validation and testing data
NN_pred_valid = NN.predict(valid_tuple[0])
NN_pred_test = NN.predict(test_tuple[0])
NN_pred_train = NN.predict(train_tuple[0])


# Calculates the predicted values for the validation data and compares them with the actual output
NN_train_err = p.error_calc(pred_vals=NN_pred_train,
                            targets=Y_train,
                            target_tags=res_var,
                            data_type='Validation data')

# Calculates the predicted values for the validation data and compares them with the actual output
NN_valid_err = p.error_calc(pred_vals=NN_pred_valid,
                              targets=Y_valid,
                              target_tags=res_var,
                              data_type='Validation data')


# Calcualtes the predicted values for the test data and compares them with the actual output
NN_test_err = p.error_calc(pred_vals=NN_pred_test,
                             targets=Y_test,
                             target_tags=res_var,
                             data_type='Test data')



plt.plot(range(len(NN_pred_test[1])), NN_pred_test[1])
plt.show()

plt.plot(range(len(NN_pred_test[1])), Y_test[1])
plt.show()

l = np.linspace(-4,4,10)
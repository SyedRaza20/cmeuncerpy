# author meta data:
__name__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

# the import statements:
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import sys

import sw_data_process as sw

# the knn function:
def knn_neighbors(train_features, train_target, test_features):
    """
    Perform K-Nearest Neighbors regression with hyperparameter optimization using grid search.

    Parameters:
    train_features : array-like, shape (n_samples, n_features)
        Training input samples where n_samples is the number of samples and n_features is the number of features.
        
    train_target : array-like, shape (n_samples,)
        Target values (real numbers) for the training input samples.
    
    test_features : array-like, shape (m_samples, n_features)
        Testing input samples where m_samples is the number of samples to predict and n_features is the number of features.

    Returns:
    tuple
        - predictions: array, shape (m_samples,)
          Predicted target values for the test data.
        - best_model: KNeighborsRegressor object
          The best K-Nearest Neighbors regressor model found by grid search.
    
    The function performs a grid search over a pre-defined range of hyperparameters to find the optimal K-Nearest Neighbors
    regressor model. It then predicts target values for the provided test feature data using the best found model.
    """
    # Hyperparameter optimization:
    param_grid = {
        'n_neighbors': range(1, 30),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Fit the KNN model to make predictions:
    knn_model = KNeighborsRegressor()
    grid_search = GridSearchCV(knn_model, param_grid, cv=5)
    grid_search.fit(train_features, train_target)
    knn_best = grid_search.best_estimator_
    
    # Make the predictions:
    predictions = knn_best.predict(test_features)
    
    # Optionally, you can return the best estimator and its parameters along with the predictions
    return predictions, knn_best

# the Support vector machine function:
def svm_regression(train_features, train_target, test_features):
    """
    Perform Support Vector Machine regression with hyperparameter optimization using grid search.

    Parameters:
    train_features : array-like, shape (n_samples, n_features)
        Training input samples where n_samples is the number of samples and n_features is the number of features.
        
    train_target : array-like, shape (n_samples,)
        Target values (real numbers) for the training input samples.
    
    test_features : array-like, shape (m_samples, n_features)
        Testing input samples where m_samples is the number of samples to predict and n_features is the number of features.

    Returns:
    tuple
        - predictions: array, shape (m_samples,)
          Predicted target values for the test data.
        - best_model: SVR object
          The best Support Vector Regression model found by grid search.
    
    The function performs a grid search over a pre-defined range of hyperparameters to find the optimal Support Vector
    Regression model. It then predicts target values for the provided test feature data using the best found model.
    """
    # Hyperparameter optimization:
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000], 
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'rbf']
    }

    # Fit the SVM model to make predictions:
    svm_model = SVR()
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(train_features, train_target)
    svm_best = grid_search.best_estimator_
    
    # Make the predictions:
    predictions = svm_best.predict(test_features)
    
    return predictions, svm_best

# the linear regression function:
def linear_regression(train_features, train_target, test_features):
    """
    Perform Linear Regression to predict target values based on input features.

    Parameters:
    train_features : array-like, shape (n_samples, n_features)
        Training input samples where n_samples is the number of samples and n_features is the number of features.
        
    train_target : array-like, shape (n_samples,)
        Target values for the training input samples.
    
    test_features : array-like, shape (m_samples, n_features)
        Testing input samples where m_samples is the number of samples to predict and n_features is the number of features.

    Returns:
    tuple
        - predictions: array, shape (m_samples,)
          Predicted target values for the test data.
        - best_model: LinearRegression object
          The Linear Regression model used for prediction.
    
    The function fits a Linear Regression model using the provided training data and predicts target values for the provided test feature data.
    """
    # Fit the Linear Regression model:
    linear_model = LinearRegression()
    linear_model.fit(train_features, train_target)
    
    # Make the predictions:
    predictions = linear_model.predict(test_features)
    
    return predictions, linear_model

# a function for normalizing data:
def normalize_data(data, columns_to_exclude):
    """
    This function takes the data set and returns it after normalizing it,
    excluding specified columns.
    
    Params:
        data (pd.DataFrame): The original DataFrame.
        columns_to_exclude (list): List of column names to exclude from normalization.
    
    Returns:
        pd.DataFrame: The DataFrame with normalized values, excluding the specified columns.
    """
    scaler = MinMaxScaler()
    
    # Exclude the columns that you don't want to normalize
    data_to_normalize = data.drop(columns=columns_to_exclude)
    
    # Fit the scaler on the data to normalize and transform it
    normalized_data = pd.DataFrame(scaler.fit_transform(data_to_normalize),
                                   columns=data_to_normalize.columns,
                                   index=data.index)
    
    # Combine the excluded columns back into the DataFrame
    data_normalized = pd.concat([data[columns_to_exclude], normalized_data], axis=1)
    
    return data_normalized

# make a function for leave-one-out cross validation:
def LOOCV(data, features, target):
    """
    This function performs leave-one-out cross validation on the data using the machine learning models.

    Parameters:
    data : DataFrame
        The input data 

    features : list
        The list of features to be used for training and testing the machine learning models. 

    target : str
        The target variable to be predicted using the machine learning models. This will be "PE" in our case.

    Returns:
    mae : dictionary
        The dictionary containing the mean absolute error for each machine learning model, along with the mae of PE
        (basically the test target).

    """
    # get the Machine Learning data:
    data_ml = data[features + [target]]

    # make an empty list to store the predictions:
    knn_pred = []
    svm_pred = []
    lr_pred = []

    # make the for loop for the LOOCV:
    for index, row in data_ml.iterrows():

        # make the training and testing sets:
        train = data_ml.drop(index, axis=0)
        test = row

        # separating the features and the target variable in both training and testing sets:
        X_train = train[features]
        X_test = test[features].to_frame().T  # This keeps X_test as a DataFrame with column names
        y_train = train[target]
        y_test = test[target]

        # call the machine learning models:
        knn = knn_neighbors(X_train, y_train, X_test)
        svm = svm_regression(X_train, y_train, X_test)
        lr = linear_regression(X_train, y_train, X_test)

        # make the predictions:
        knn_pred.append(y_test - knn[0])
        svm_pred.append(y_test - svm[0])
        lr_pred.append(y_test - lr[0])

        # print the progress:
        print(f"Completed LOOCV for index: {index}")
    
    # Convert each element in the array into one number in case they are themselves in lists/arrays:
    flatten = lambda x: [item[0] if isinstance(item, (list, np.ndarray)) else item for item in x]

    if len(features) == 1:
        feature_name = features[0]
        predictions_df = pd.DataFrame({
            f"PE_{feature_name}_knn": flatten(knn_pred),
            f"PE_{feature_name}_svm": flatten(svm_pred),
            f"PE_{feature_name}_lr": flatten(lr_pred)
        })

    else:
        predictions_df = pd.DataFrame({
            "PE_multi_knn": flatten(knn_pred),
            "PE_multi_svm": flatten(svm_pred),
            "PE_multi_lr": flatten(lr_pred)
        })

    # Call the function that stores everything in an excel file:
    save_predictions(predictions_df)

    # calculate the mean absolute error:
    mae = {
        "knn": np.mean(np.abs(knn_pred)),
        "svm": np.mean(np.abs(svm_pred)),
        "lr": np.mean(np.abs(lr_pred))
    }

    return mae

# make a function for univariate analysis:
def uni_LOOCV(cme_data, features, target="PE"):
    """
    This function performs leave-one-out cross validation on the data using the machine learning models for univariate
    analysis.

    Parameters:
    cme_data : DataFrame
        The input data containing 9 parameters. 4 CME CONE parameters, and 4 SW parameters.

    features: list
        The list of features to be used for training and testing the machine learning models. 

    target : str

    Returns:
    mae_uni: dictionary
        The nested dictionary containing the mean absolute error for each feature and each ml technique. This structure
        will look something like:

        {
            "feature 1": {
                "knn": 0.5,
                "svm": 0.6,
                "lr": 0.7
            },
            "feature 2": {
                "knn": 0.5,
                "svm": 0.6,
                "lr": 0.7
            },
            ...
        }
    
    """
    # make an empty dictionary to store the results:
    mae_uni = {}

    # make the for loop for the LOOCV:
    for feature in features:
        # get the result:
        mae_uni[feature] = LOOCV(cme_data, [feature], target)

    return mae_uni

# make a function for multivariate analysis:
def multi_LOOCV(data, features, target="PE"):
    """
    This function performs leave-one-out cross validation on the data using the machine learning models for multivariate
    inputs

    Parameters:
    data : DataFrame
        The input data containing 9 parameters. 4 CME CONE parameters, and 5 SW parameters.

    features : list
        The list of features to be used for training and testing the machine learning models. 

    target : str
        The target variable to be predicted using the machine learning models. This will be "PE" in our case.

    Returns:
    mae_multi : dictionary
        This is a dictionary of dictionaries. The keys are # of best features: "2 best features", "3 best features", etc. The 
        value is another dictionary. The key here is the ml technique and the value is mae for that "# best features" and 
        "ml technique" 
    """

    # the multi_mae returning dictionary:

    """
    mae_multi = {}
    for i in range(1, len(features) + 1):
        mae_multi[f"{i} best features" if i != len(features) else "all features multivariate"] = LOOCV(data, features[:i], target)
    """

    return LOOCV(data, features, target)

def uni_mae_plots(mae, save_path, cme_data):
    """
    This function plots the mean-absolute-errors of the univariate model.
    
    Parameters:
    mae: dictionary
        mae is a dictionary whose key-value pairs are (string, dictionary). The keys represent the features
        of the ml models and the values are dictionaries of key-value pairs (string, float). The strings in
        this dictionary are names of the techniques used (e.g svm, knn, or lr). The values in this case are 
        the reduced MAEs.
        
    save_path: string
        This is the path where the resulting figure will be stored.
        
    enlil_mae:
        This parameter is the enlil mae which I have to show on the plot, for comparison about how much the 
        error was reduced. 
        
    Returns:
    None
    """
    
    # Import statements:
    import numpy as np
    import matplotlib.pyplot as plt

    # get the enlil mae:
    enlil_mae = np.mean(np.abs(cme_data["PE"]))
    
    # Making the plot data dictionary:
    plot_data = {'KNN': [], 'LR': [], 'SVM': []}
    for feature in list(mae.keys()):
        for method in plot_data.keys():
            method_lower = method.lower()  # Convert method to lowercase to match keys in the dictionary
            plot_data[method].append(mae[feature][method_lower])
    
    # Define bar parameters
    colors = {'KNN': 'red', 'LR': 'green', 'SVM': 'blue'}  # Placeholder colors, replace with your own
    bar_width = 0.2
    index = np.arange(len(list(mae.keys())))  # Match index length with the number of features
    
    # Plot features and data
    for i, ml_technique in enumerate(plot_data.keys()):
        plt.bar(index + i * bar_width, plot_data[ml_technique], color=colors[ml_technique], width=bar_width, label=ml_technique)

    feature_labels = list(mae.keys())  # Create feature labels
    plt.xticks(index + bar_width / 2 * len(plot_data), feature_labels, rotation=45, fontsize=16)

    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    plt.title('MAE vs Selected Features', fontsize=12)
    plt.axhline(y=enlil_mae, color='black', linestyle='--', linewidth=3, label='ENLIL MAE')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, markerscale=1.5, labelspacing=0.8)
    plt.grid(axis='y')
    plt.ylim(0, 14)  # Adjust y-axis limit to 14
    plt.tight_layout(rect=[0, 0, 1.4, 1.2])

    # Save the figure
    plt.savefig(save_path, dpi=500)

    return None

def ranked_univariate(uni_mae, save_path, zoom = True):
    """
    Takes a dictionary of MAE values and performs the following:
    1. Computes the average MAE for each feature across all models.
    2. Sorts the dictionary by the average MAE in descending order.
    3. Plots the sorted values as a horizontal bar chart.
    4. Saves the plot to the specified path.
    5. Returns the sorted list of parameter names (features).
    
    Parameters:
    uni_mae (dict): Dictionary where keys are feature names and values are their MAE for different models.
    save_path (str): Path where the plot should be saved.
    zoom (boolean): Turn it off to see the whole bar (like in the paper). Turn it off to zoom in close to 12.31 hours.
    
    Returns:
    list: A list of parameter names sorted by their average MAE values in descending order.
    """
    
    # Compute the average MAE for each feature across models
    average_mae = {feature: sum(models.values()) / len(models) for feature, models in uni_mae.items()}
    sorted_average_mae = {}

    if zoom == True:
        for i in average_mae.keys():
            average_mae[i] = (12.31 - average_mae[i]) * 60.0

        sorted_average_mae = dict(sorted(average_mae.items(), key=lambda item: item[1], reverse=True))

    else:
        # Sorting the dictionary by the average MAE in ascending order
        sorted_average_mae = dict(sorted(average_mae.items(), key=lambda item: item[1], reverse=False))

    old_keys = list(sorted_average_mae)
    new_keys = [r"$\Delta B_{SW}$", r"$\Delta P_{SW}$", "CME speed" , "CME half_width", r"$\Delta n_{SW}$" , "latitude", r"$\Delta v_{SW}$", "longitude", r"$\Delta T_{SW}$"]

    sorted_average_mae = {new_key: value for new_key, value in zip(new_keys, sorted_average_mae.values())}

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.barh(list(sorted_average_mae.keys()), list(sorted_average_mae.values()), color='blue', height=0.5)
    plt.xlabel('Average MAE')
    plt.title('Average MAE for Each Feature Based on Univariate ML Models')
    plt.grid(True)

    # Save the plot to the specified path
    plt.savefig(save_path, dpi=500)

    # Close the plot to free memory
    plt.close()

    # change the keys back for the rest of the workflow
    sorted_average_mae = {new_key: value for new_key, value in zip(old_keys, sorted_average_mae.values())}

    # Return the sorted list of parameter names
    return list(sorted_average_mae.keys())

def ml_plots(uni_mae, multi_mae, save_path, cme_data, selected_features):
    """
    This function makes the plots for the machine learning analysis, including a dotted line for the ENLIL MAE.

    Parameters:
    uni_mae : dictionary
        The nested dictionary containing the mean absolute error for each feature and each ML technique for univariate
        analysis.

    multi_mae : dictionary
        The dictionary containing the mean absolute error for each machine learning model, for multivariate analysis.

    save_path : str
        The path where the plots will be saved.

    cme_data : dictionary
        Dictionary containing CME data, used to calculate the ENLIL MAE.

    selected_features : list
        List of features to be included in the plot.

    Returns:
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    enlil_mae = np.mean(np.abs(cme_data["PE"]))
    
    plot_data = {'KNN': [], 'LR': [], 'SVM': []}
    for feature in selected_features:
        for method in plot_data.keys():
            method_lower = method.lower()  # Convert method to lowercase to match keys in the dictionary
            plot_data[method].append(uni_mae[feature][method_lower])

    # Add multivariate results to the plot data
    for method in plot_data.keys():
        method_lower = method.lower()
        plot_data[method].append(multi_mae[method_lower])

    # Define plot settings
    colors = {'KNN': 'red', 'LR': 'green', 'SVM': 'blue'}
    bar_width = 0.25
    n_features = len(selected_features) + 1  # Plus one for the multivariate data
    index = np.arange(n_features)

    plt.figure(figsize=(20, 10))

    # Plot features and data
    for i, ml_technique in enumerate(plot_data.keys()):
        plt.bar(index + i * bar_width, plot_data[ml_technique], color=colors[ml_technique], width=bar_width, label=ml_technique)

    feature_labels = selected_features + ['Multivariate']
    plt.xticks(index + bar_width / 2 * len(plot_data), feature_labels, rotation=45, fontsize=16)

    plt.xlabel('Feature', fontsize=26)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=26)
    plt.title('MAE vs Selected Features', fontsize=30)
    plt.axhline(y=enlil_mae, color='black', linestyle='--', linewidth=6, label='ENLIL MAE')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=28, markerscale=1.5, labelspacing=1.2)
    plt.grid(axis='y')
    plt.ylim(0, 14)  # Adjust y-axis limit to 14
    plt.tight_layout(rect=[0, 0, 0.75, 1])

    # Save the figure
    plt.savefig(save_path, dpi=500)
    return None
    
def save_predictions(predictions_df):
    """
    This function updates an existing Excel file with new machine learning predictions by appending the predictions
    as new columns. It reads the existing data, appends the new predictions, and saves the updated DataFrame back to the
    Excel file.

    Parameters:
    predictions_df : DataFrame
        The DataFrame containing the machine learning predictions for one machine learning run (all models included) 

    Returns:
    None

    #######################################################################################################################
    AN ISSUE: There is a problem with this function. The file it creates, ml_PE.xlsx, has too manny columns. It keeps making
    duplicate columns for the same data. This is a problem that needs to be fixed.
    #######################################################################################################################
    """
    # Attempt to read the existing Excel file
    try:
        ml_PE = pd.read_excel('ml_PE.xlsx')
    except FileNotFoundError:
        print("File 'ml_PE.xlsx' not found. A new file will be created.")
        ml_PE = pd.DataFrame()

    # Append the prediction_df to the ml_transit DataFrame as new columns
    ml_PE = pd.concat([ml_PE, predictions_df], axis=1)

    # Write the updated DataFrame back into the Excel file, overwriting the old file
    ml_PE.to_excel('ml_PE.xlsx', index=False)
    print("Predictions have been saved/updated in 'ml_PE.xlsx'.")

    return None

# And with that, Machine Learning protocols module is finished
############################################################################################################
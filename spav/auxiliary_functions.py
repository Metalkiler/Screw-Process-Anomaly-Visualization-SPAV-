import h2o
import pickle
import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def data_preprocessing(train_df: pd.DataFrame, feature_cols):
    """
    Create pipeline for training on the selected feature columns of a pandas DataFrame. Use the training subset ONLY.
    Steps: 1) imputation of missing values with KNN; 2) Scaling of features using a MinMaxScaler with range [0, 1]
    Can be subsequently applied on the training and test subsets of a dataset.
    """
    # Pre-processing
    steps = []
    # Impute missing values
    steps.append(("imputer", KNNImputer()))
    # Scale features to have zero mean and unit variance.
    # We want our input data's feature range to match the tanh activation function's expected input feature range.
    steps.append(("scaler", MinMaxScaler(feature_range=(0, 1))))
    pipe = Pipeline(steps)
    pipe.fit(train_df[feature_cols])
    return pipe


def get_reconstruction_df(model, df: pd.DataFrame) -> pd.DataFrame:
    df_h2o = h2o.H2OFrame(df.copy())  # need h2o frame for use with h2o model
    reconstruction = model.predict(df_h2o)
    reconstruction_df = reconstruction.as_data_frame(use_multi_thread=True).set_index(df.index)

    return reconstruction_df


def get_reconstruction_df_ae(model, df: pd.DataFrame) -> pd.DataFrame:
         # need h2o frame for use with h2o model
    reconstruction = model.predict(df)
    reconstruction_df = pd.DataFrame(reconstruction, columns=df.columns, index=df.index)

    return reconstruction_df



def find_optimal_threshold(fpr, tpr, thresholds):
    distances = np.sqrt(fpr ** 2 + (tpr - 1) ** 2)
    return thresholds[np.argmin(distances)]


def get_score_df(model, X_test: pd.DataFrame, y_true: pd.Series, cycle_id: int = None):
    """
    Get dataframe with AE scores, actual value and process id per point across all processes
    (as index if multi-index or cycle_id column if local score).
    """
    if cycle_id:
        y_true = y_true.loc[cycle_id]
        X_test = X_test.loc[cycle_id]

    if isinstance(model, h2o.estimators.deeplearning.H2OAutoEncoderEstimator):
        X_test_reset = X_test.copy().reset_index()
        X_test_h2o = h2o.H2OFrame(X_test_reset)
        X_pred = model.predict(X_test_h2o)
    else:
        X_pred = model.predict(X_test)

    if isinstance(model, h2o.estimators.deeplearning.H2OAutoEncoderEstimator):
        X_pred = X_pred.as_data_frame(use_multi_thread=True).set_index(X_test.index)

    score_df = pd.DataFrame(index=X_test.index)
    score_df["ae"] = np.mean(np.abs(X_pred.values - X_test.values), axis=1)
    score_df["actual"] = y_true
    if cycle_id:
        score_df["cycle_id"] = cycle_id

    return score_df


def find_nearest_angle(process_angle_column: pd.Series, angle_value: Union[float, int]):
    abs_angle_diffs = np.abs(process_angle_column - angle_value)
    nearest_index = int(abs_angle_diffs.argsort()[0])
    nearest_value = float(process_angle_column.iloc[nearest_index])
    print(f"Nearest value to {angle_value} is {nearest_value} at index {nearest_index}.")
    return nearest_index


def find_nearest_threshold(y_true: pd.Series, y_pred: pd.Series, threshold: float = None, force: bool = False):
    # Find nearest roc curve threshold either by calculating the optimal value or user-defined threshold
    # from the true and predicted anomaly scores from several processes.
    fpr, tpr, all_thresholds = roc_curve(y_true, y_pred)
    print(auc(fpr, tpr))

    if force:
        return threshold
    else:
        if threshold is None:
            print("Calculating optimal threshold...")
            nearest_threshold_value = find_optimal_threshold(fpr, tpr, all_thresholds)
            print(f"Optimal threshold:", nearest_threshold_value)
            return nearest_threshold_value

        threshold_differences = np.abs(all_thresholds - threshold)
        nearest_threshold_index = threshold_differences.argsort()[0]
        nearest_threshold_value = all_thresholds[nearest_threshold_index]
        return nearest_threshold_value


def absolute_error(y_true: pd.Series, y_pred: pd.Series):
    # TODO input can also be np array, pd.df
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()

    error = np.abs(y_pred - y_true)

    return error


def classify_anomaly_by_threshold(dataset: pd.DataFrame, threshold_value: float):


    dataset["label"]= dataset['ae'] >= threshold_value
    detected_anomaly_idx = dataset.loc[dataset['label'] == True].index.values
    return detected_anomaly_idx



def scoreModel(model, tDataset, th, y_test, cycle_id):
    #y_test = tDataset.pop('cyclegof')
    tDataset = tDataset.astype('float32')

    # X_test = create_sequences(tDataset.values,40)
    X_pred = model.predict(tDataset, verbose=0)
    # X_pred = X_pred.reshape(X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2])

    X_pred = pd.DataFrame(X_pred, columns=tDataset.columns)
    # X_pred.index = x_test.index

    scored = pd.DataFrame(index=tDataset.index)
    # Xtest = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])
    Xtest = tDataset
    scored['ae'] = np.mean(np.mean(np.abs(X_pred.values - tDataset.values), axis=1))
    #scored['Loss_mae1'] = 1 - np.mean(np.mean(np.abs(X_pred.values - tDataset.values), axis=1))
    #scored['Threshold'] = th
    #scored['Anomaly'] = scored['Loss_mae'] >= scored['Threshold']
    scored['Actual'] = y_test.values.tolist()
    scored['cycle_id'] = cycle_id
    return scored


def get_anomaly_scores(model, df, modelType="DFFN", yData=None):
    cycle_ids = df.index.get_level_values(0)  # retrieve unique cycle ids
    unique_ids = np.unique(cycle_ids)

    if modelType == "H2o":
        df_h2o = h2o.H2OFrame(df)  # need h2o frame for use with h2o model
        # Returns an H2OFrame object containing the reconstruction MSE or the per-feature squared error
        anomalies = model.anomaly(df_h2o)
        # Replicate the multi-index structure of the screwing data
        anomalies_df = pd.concat([anomalies.as_data_frame(use_multi_thread=True), pd.DataFrame({"cycle_id": cycle_ids})], axis=1)
        anomalies_df = anomalies_df.set_index(['cycle_id', anomalies_df.groupby('cycle_id').cumcount()])
    else:
        DFScores=[]
        for uid in unique_ids:
            grupo = df.loc[uid]
            y_dataFiltered = yData.loc[uid]
            cycleid=[uid for i in range(0, grupo.shape[0])]
            df_testes = scoreModel(model, grupo, 0, y_dataFiltered, cycleid)
            DFScores.append(df_testes)
        anomalies_df = pd.DataFrame()
        anomalies_df = pd.concat([pd.DataFrame(i) for i in DFScores], ignore_index=True)
    return anomalies_df

def get_anomaly_scores_single_ae(model, df, yData=None, cycleid=None):

    cycleid=[cycleid for i in range(0, df.shape[0])]
    df_testes = scoreModel_single(model, df, 0, yData, cycleid)


    return df_testes

def scoreModel_single(model, tDataset, th, y_test, cycle_id):
    #y_test = tDataset.pop('cyclegof')
    tDataset = tDataset.astype('float32')

    # X_test = create_sequences(tDataset.values,40)
    X_pred = model.predict(tDataset, verbose=0)
    # X_pred = X_pred.reshape(X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2])

    X_pred = pd.DataFrame(X_pred, columns=tDataset.columns)
    # X_pred.index = x_test.index

    scored = pd.DataFrame(index=tDataset.index)
    # Xtest = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])
    Xtest = tDataset
    scored['ae'] = np.mean(np.abs(X_pred.values - tDataset.values), axis=1)
    #scored['Loss_mae1'] = 1 - np.mean(np.mean(np.abs(X_pred.values - tDataset.values), axis=1))
    #scored['Threshold'] = th
    #scored['Anomaly'] = scored['Loss_mae'] >= scored['Threshold']
    scored['Actual'] = y_test.values.tolist()
    scored['cycle_id'] = cycle_id
    return scored

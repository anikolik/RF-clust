# to handle paths
import sys
from pathlib import Path

# to handle datasets
import pandas as pd
import numpy as np

# for calculating paiwise similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

# for creating estimator
from sklearn.base import BaseEstimator


class RF_clus(BaseEstimator):
    def __init__(self, base_estimator: object, metric: str, similarity_threshold: float, method: str, weights: pd.DataFrame):
        """
        Initialization of class.
        
        base_estimator: fitted ML model instance.
        metric: similarity metric options: ["cosine", "euclidean"].
        similarity_threshold: Needs to be set accordingly to chosen metric.
        method: aggregation method for the calibration of the prediction ["mean", "median", "weighted"].
        weights: weights to included in the calculation of the similarity.
        """
        self.base_estimator = base_estimator
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        self.method = method
        self.weights = weights
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Function to get train data.
        
        X_train: pandas df with f_id and i_id as index and features as columns.
        y_train: pandas df with f_id and i_id as index and target as column.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        return self
    
        
    def predict_rf_clust(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        Function to make predictions with RF+clust.
        
        X_test: pandas df with f_id and i_id as index and features as columns.
        y_test: pandas df with f_id and i_id as index and target as column.
        
        """
        print("Predicting with RF+Clust: ")
        # create results placeholders
        rfclus_predictions = pd.DataFrame()
        mask = pd.DataFrame()
        similar_instances = pd.DataFrame()
        similarity = pd.DataFrame()
        
        for index in X_test.index:
            
            # select the test instance
            test_instance = pd.DataFrame(X_test.loc[index, :]).transpose()
            print(f"Test instance: {index}")
            print(test_instance)

            # calculate the similarity of the test instance with the training instances
            similarity_temp = self.calculate_similarity(X_train=self.X_train, test_instance=test_instance)
            print("similarity: ")
            print(similarity_temp)
        
            # find the most similar instances from train
            similar_instances_temp = self.find_similar_instances_with_threshold(similarity=similarity_temp)
            print("similar_instances: ")
            print(similar_instances_temp)
            
            # make prediction with the base model
            rf_prediction = self.base_estimator.predict(test_instance)
            
            # calibrate prediction
            rfclus_prediction = pd.DataFrame(y_test.loc[index]).transpose().rename(columns={"log_precision": "true"}).copy()    
            
            rfclus_prediction["predicted"] = self.calibrate_prediction(test_prediction=rf_prediction
                                                  , y_train_similar=self.y_train.loc[similar_instances_temp, :]
                                                  , similarity=similarity_temp)
            print("rfclus_prediction: ")
            print(rfclus_prediction)
            
            # create a mask to flag instances for which predictions can/cannot be made
            mask_temp = pd.DataFrame(self.mask(similar_instances_temp), index=pd.MultiIndex.from_tuples([index], names=["f_id", "i_id"]), columns = ["predicted"])        
            print("mask: ")
            print(mask_temp)
            
            # save
            similarity = pd.concat([similarity, similarity_temp.rename_axis(["f_id", "i_id"])], axis=0)

            similar_instances_temp_df = pd.DataFrame(index=[0], columns = ["similar_instances"])       
            similar_instances_temp_df.loc[0, "similar_instances"] = similar_instances_temp
            similar_instances_temp_df.index = pd.MultiIndex.from_tuples([index], names=["f_id", "i_id"])
            similar_instances = pd.concat([similar_instances, similar_instances_temp_df], axis=0)

            rfclus_predictions = pd.concat([rfclus_predictions, rfclus_prediction.rename_axis(["f_id", "i_id"])], axis=0)
            mask = pd.concat([mask, mask_temp], axis=0)

                
        return rfclus_predictions, mask, similarity, similar_instances
    
    
    def calibrate_prediction(self, test_prediction: float, y_train_similar: pd.DataFrame, similarity: pd.DataFrame):
        """
        Function to calibrate the prediction with the algorithm performance on the most similar instances.
        
        y_train: pandas df with f_id and i_id as index and target as column.
        y_train: pandas df with f_id and i_id as index and similarity with other insatnces in columns.
        """        
        # if there are similar instances over a certain threshold calibrate the prediction for the test instance 
        # with the true values of similar instances from train
        if len(y_train_similar) > 0: 

            if self.method == "mean": 
                # take prediction and calibrate with mean value of similar instances
                rfclus_prediction = (test_prediction + y_train_similar["log_precision"].mean())/2
            
            elif self.method == "median":
                # take prediction and calibrate with median value of similar instances
                rfclus_prediction = (test_prediction + y_train_similar["log_precision"].median())/2
            
            elif self.method == "weighted": 
                
                # weighted calibration according to similarity
                numerator = 0
                denominator = 0

                for ms in y_train_similar.index:
                    print(f"ms: {ms}")

                    # get prediction for simialr instance
                    ms_value = y_train_similar.loc[ms]["log_precision"]
                    print(f"ms value: {ms_value}")

                    # get similarity with test instance as weight
                    ms_weight = similarity[ms].values[0]
                    print(f"ms weight: {ms_weight}")

                    # weight prediction
                    if self.metric != "cosine":
                        ms_weight = 1/ms_weight
                        
                    numerator = numerator + (ms_weight * ms_value)

                    # normalize with sum of the weights
                    denominator = denominator + ms_weight

                rfclus_prediction = (test_prediction + (numerator / denominator)) / 2
            
            return rfclus_prediction

        else:  # if there are no similar instances prediction cannot be made with rf+clust

            return test_prediction
        
        
    def mask(self, similar_instances):
        """
        Function to create flag if predition can be calibrated or not.
        """        
        # if there are similar instances over a certain threshold
        if len(similar_instances) > 0: 
            return False
        else: # if there are no similar instances
            return True

        
    def find_similar_instances_with_threshold(self, similarity: pd.DataFrame) -> list:
        """
        Function to find similar instances according to a threshold.
        """ 
        # find instances with similarity greater/less than a threshold
        if self.metric == "cosine":
            similar_instances = similarity.ge(self.similarity_threshold)
        
        elif self.metric == "euclidean":
            similar_instances = similarity.le(self.similarity_threshold)
      
        return similar_instances.apply(lambda x: x.index[x].tolist(), axis=1).values[0]
    
    
    def calculate_similarity(self, X_train: pd.DataFrame, test_instance: pd.DataFrame) -> pd.DataFrame:
        """
        Function for calculating similarity of a test instance with train instances.
        """        
        # concatenate the train dataset and test instance
        data = pd.concat([X_train, test_instance], axis=0)       

        # calculate paiwise similarity
        if self.weights is not None:
            distances = pdist(data, metric=self.metric, w=self.weights)
        else:
            distances = pdist(data, metric=self.metric)

        # convert the result to a square matrix
        distances = squareform(distances)
        distances = pd.DataFrame(distances, index=data.index, columns=data.index)

        # if metric == "cosine" transform to similarity
        if self.metric == "cosine":
            distances = 1 - distances
            
        # get similarities of the test instance
        similarity = pd.DataFrame(distances.loc[test_instance.index[0]]).transpose()

        # drop similarity with itself
        similarity = similarity.drop(test_instance.index[0], axis=1)


        return similarity
    
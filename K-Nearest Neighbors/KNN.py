import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import pickle

class KNN_DTW_Classifier:
    """
    This class trains a K-Nearest Neighbors model with Dynamic Time Wrapping for classification tasks using group k-fold cross-validation.

    Attributes
    ----------
    features : numpy array
        Preprocessed feature set with shape (samples, time_steps, channels).
    target : numpy array
        Target variable for classification. Same length as the number of samples.
    groups : numpy array
        Indicating the chunk index for each sample.
    n_splits : int
        Number of folds for cross-validation.
    n_neighbors : int
        Number of neighbors for K-Nearest Neighbors.
    """
    def __init__(self, features, target, groups, n_splits=5, n_neighbors=15, scale=False):
        self.features = features
        self.target = target
        self.groups = groups
        self.n_splits = n_splits
        self.n_neighbors = n_neighbors
        self.scale = scale

    def train(self):
        """
        Train the K-Nearest Neighbors model with Dynamic Time Wrapping distance using group k-fold cross-validation and evaluate using various metrics.

        Returns
        -------
        None
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        results_test = []
        results_train = []

        y_true_list = []
        y_prob_list = []
        y_train_true_list = []
        y_train_prob_list = []

        start_time = time.time()

        # Create and train K-Nearest Neighbors classifier
        for train_index, test_index in gkf.split(self.features, self.target, self.groups):
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.target[train_index], self.target[test_index]


            if self.scale:
                # Scale the features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
                X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


            # Create the K-Nearest Neighbors model with DTW as distance metric
            knn = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors, metric="dtw")

            # Train the K-Nearest Neighbors model
            knn.fit(X_train, y_train)

            # Make predictions
            y_pred = knn.predict(X_test)
            y_prob = knn.predict_proba(X_test)[:, 1]  # Probability of class 1, ranges from 0 to 1

            # Lists with actual and predicted targets for the test data
            y_true_list.extend(y_test)
            y_prob_list.extend(y_prob)

            # Evaluate metrics
            kappa = cohen_kappa_score(y_test, y_pred)
            mean_f1 = f1_score(y_test, y_pred, average='weighted')
            mean_sensitivity = recall_score(y_test, y_pred, average='macro')
            mean_precision = precision_score(y_test, y_pred, average='macro')
            mean_accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            # Store the results
            results_test.append({
                "Kappa": round(kappa, 3),
                "F1": round(mean_f1, 3),
                "Sensitivity": round(mean_sensitivity, 3),
                "Precision": round(mean_precision, 3),
                "Accuracy": round(mean_accuracy, 3),
                "AUC": round(auc, 3)
            })

            # Make predictions on the training set
            y_train_pred = knn.predict(X_train)
            y_train_prob = knn.predict_proba(X_train)[:, 1]

            # Aggregate true labels and predicted probabilities for the training set
            y_train_true_list.extend(y_train)
            y_train_prob_list.extend(y_train_prob)

            # Evaluating metrics for training set
            kappa_train = cohen_kappa_score(y_train, y_train_pred)
            mean_f1_train = f1_score(y_train, y_train_pred, average='weighted')
            mean_sensitivity_train = recall_score(y_train, y_train_pred, average='macro')
            mean_precision_train = precision_score(y_train, y_train_pred, average='macro')
            mean_accuracy_train = accuracy_score(y_train, y_train_pred)
            auc_train = roc_auc_score(y_train, y_train_prob)

            # Store the results for the training set
            results_train.append({
                "Kappa_train": round(kappa_train, 3),
                "F1_train": round(mean_f1_train, 3),
                "Sensitivity_train": round(mean_sensitivity_train, 3),
                "Precision_train": round(mean_precision_train, 3),
                "Accuracy_train": round(mean_accuracy_train, 3),
                "AUC_train": round(auc_train, 3)
            })

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time}")

        # Results as a DataFrame (model evaluation on test data)
        results_test_per_fold = pd.DataFrame(results_test)
        print("\nPer Fold Cross-Validation Results for test set:\n", results_test_per_fold)

        # Average results for the test set
        average_results = round(results_test_per_fold.mean(), 3)
        print("\nAverage Cross-Validation Results for test set:\n", average_results)

        # Results for train data
        results_train_per_fold = pd.DataFrame(results_train)
        print("\nPer Fold Cross-Validation Results for train set:\n", results_train_per_fold)

        # Average results for the training set
        average_train_results = round(results_train_per_fold.mean(), 3)
        print("\nAverage Cross-Validation Results for train set:\n", average_train_results)

        # ROC-AUC for test data
        fpr_test, tpr_test, _ = roc_curve(y_true_list, y_prob_list)
        auc_test = average_results.loc['AUC']
        plt.figure()
        plt.plot(fpr_test, tpr_test, color='orange', label=f'ROC curve (AUC = {auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve for model evaluation (K-Nearest Neighbors)')
        plt.legend(loc='lower right')
        plt.show()

        # ROC-AUC for train data
        fpr_train, tpr_train, _ = roc_curve(y_train_true_list, y_train_prob_list)
        auc_train = average_train_results.loc['AUC_train']
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='blue', label=f'ROC curve (AUC = {auc_train:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve for training set (K-Nearest Neighbors)')
        plt.legend(loc='lower right')
        plt.show()

        # Plotting both ROC curves together
        plt.figure()
        plt.plot(fpr_test, tpr_test, color='orange', label=f'Test ROC (AUC = {auc_test:.2f})')
        plt.plot(fpr_train, tpr_train, color='blue', label=f'Train ROC (AUC = {auc_train:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curves for Training and Test Sets')
        plt.legend(loc='lower right')
        plt.show()


        # Train the final model on the entire dataset
        final_knn = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors, metric="dtw")

        if self.scale:
            # Scale the features
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features.reshape(-1, self.features.shape[-1])).reshape(
                self.features.shape)

        # Train the K-Nearest Neighbors model
        final_knn.fit(self.features, self.target)

        # Save the model
        with open('model_A_knn_dtw.pkl', 'wb') as file:
            pickle.dump(final_knn, file)

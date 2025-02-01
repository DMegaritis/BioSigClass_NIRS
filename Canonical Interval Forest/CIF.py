import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score,
                             roc_curve)
from sktime.classification.interval_based import CanonicalIntervalForest
import pickle


class CIF_Classifier:
    """
    This class trains a Canonical Interval Forest (Canonical Interval Forest) classifier using group k-fold cross-validation.

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
    n_estimators : int
        Number of trees in the forest.
    """

    def __init__(self, features, target, groups, n_splits=5, n_estimators=50):
        self.features = features
        self.target = target
        self.groups = groups
        self.n_splits = n_splits
        self.n_estimators = n_estimators

    def train(self):
        """
        Train the Canonical Interval Forest model using group k-fold cross-validation and evaluate using various metrics.

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

        for train_index, test_index in gkf.split(self.features, self.target, self.groups):
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.target[train_index], self.target[test_index]

            cif = CanonicalIntervalForest(n_estimators=self.n_estimators)  # Updated to CIFClassifier
            cif.fit(X_train, y_train)

            y_pred = cif.predict(X_test)
            y_prob = cif.predict_proba(X_test)[:, 1].astype(float)

            y_true_list.extend(y_test)
            y_prob_list.extend(y_prob)

            kappa = cohen_kappa_score(y_test, y_pred)
            mean_f1 = f1_score(y_test, y_pred, average='weighted')
            mean_sensitivity = recall_score(y_test, y_pred, average='macro')
            mean_precision = precision_score(y_test, y_pred, average='macro')
            mean_accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            results_test.append({
                "Kappa": round(kappa, 3),
                "F1": round(mean_f1, 3),
                "Sensitivity": round(mean_sensitivity, 3),
                "Precision": round(mean_precision, 3),
                "Accuracy": round(mean_accuracy, 3),
                "AUC": round(auc, 3)
            })

            y_train_pred = cif.predict(X_train)
            y_train_prob = cif.predict_proba(X_train)[:, 1].astype(float)
            y_train_true_list.extend(y_train)
            y_train_prob_list.extend(y_train_prob)

            kappa_train = cohen_kappa_score(y_train, y_train_pred)
            mean_f1_train = f1_score(y_train, y_train_pred, average='weighted')
            mean_sensitivity_train = recall_score(y_train, y_train_pred, average='macro')
            mean_precision_train = precision_score(y_train, y_train_pred, average='macro')
            mean_accuracy_train = accuracy_score(y_train, y_train_pred)
            auc_train = roc_auc_score(y_train, y_train_prob)

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
        print(f"Training time: {training_time:.2f} seconds")

        results_test_per_fold = pd.DataFrame(results_test)
        print("\nPer Fold Cross-Validation Results for test set:\n", results_test_per_fold)
        average_results = round(results_test_per_fold.mean(), 3)
        print("\nAverage Cross-Validation Results for test set:\n", average_results)

        results_train_per_fold = pd.DataFrame(results_train)
        print("\nPer Fold Cross-Validation Results for train set:\n", results_train_per_fold)
        average_train_results = round(results_train_per_fold.mean(), 3)
        print("\nAverage Cross-Validation Results for train set:\n", average_train_results)

        # ROC Curve for test set
        fpr_test, tpr_test, _ = roc_curve(y_true_list, y_prob_list)
        auc_test = average_results.loc['AUC']
        plt.figure()
        plt.plot(fpr_test, tpr_test, color='orange', label=f'ROC curve (AUC = {auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve for Test Set (Canonical Interval Forest)')
        plt.legend(loc='lower right')
        plt.show()

        # ROC Curve for training set
        fpr_train, tpr_train, _ = roc_curve(y_train_true_list, y_train_prob_list)
        auc_train = average_train_results.loc['AUC_train']
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='blue', label=f'ROC curve (AUC = {auc_train:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve for Training Set (Canonical Interval Forest)')
        plt.legend(loc='lower right')
        plt.show()

        # Combined ROC Curve for both test and train sets
        plt.figure()
        plt.plot(fpr_test, tpr_test, color='orange', label=f'Test ROC (AUC = {auc_test:.2f})')
        plt.plot(fpr_train, tpr_train, color='blue', label=f'Train ROC (AUC = {auc_train:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curves for Training and Test Sets')
        plt.legend(loc='lower right')
        plt.show()

        # Train final model on the entire dataset
        final_cif = CanonicalIntervalForest(n_estimators=self.n_estimators)
        final_cif.fit(self.features, self.target)

        with open('pre_trained_cif.pkl', 'wb') as file:
            pickle.dump(final_cif, file)



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score

import matplotlib.pyplot as plt

class HealthDataAnalysis:
    def __init__(self):
        url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
        self.df = pd.read_csv(url)
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.male = self.female = None
        self.oldPatients = self.youngPatients = None
        self.genderMetrics = None
        self.ageMetrics = None
        self.overallAccuracy = None
        

    def trainModel(self):
        # Drop the 'target' column from the data set.
        X = self.df.drop('target', axis=1)
        # Keep the 'target' column separately as the thing to predict on
        y = self.df['target']
        # Split the training and testing data sets from the X and y arrays above. Use 0.25 as the test size ratio
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # We need to standardize the samples by removing the mean and scaling to fit the variance.
        scaler = StandardScaler()
        # Fit the training and test data sets
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.fit_transform(self.X_test)

        # Train the logistic regression model and fit it on the scaled training data sets and their labels
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train)

    def overallAccuracyMetrics(self, y_pred):
        # Use the model to predict the value for the scaled test samples and calculate the overall accuracy and classification report for the predictions

        return {
            "Overall Accuracy": accuracy_score(self.y_test, y_pred),
            "Classification Report": classification_report(self.y_test, y_pred)
        }
        
    def biasMetrics(self, data1, data2):
        # Calculate the accuracy and recall for both data sets and calculate the error rates for both data sets using the error_rates function
        return {
            "Accuracy1": accuracy_score(data1['true'], data1['pred']),
            "Recall1": recall_score(data1['true'], data1['pred']),
            "Accuracy2": accuracy_score(data2['true'], data2['pred']),
            "Recall2": recall_score(data2['true'], data2['pred']),
            "Error Rates1": self.error_rates(data1),
            "Error Rates2": self.error_rates(data2),
        }

    def evaluateBias(self):
        # Use the model to predict the value for the scaled test samples and calculate the overall accuracy and classification report for the predictions
        y_pred = self.model.predict(self.X_test_scaled)
        self.overallAccuracy = self.overallAccuracyMetrics(y_pred)

        test_results = self.X_test.copy()
        test_results['true'] = self.y_test.values
        test_results['pred'] = y_pred
        test_results['sex'] = self.X_test['sex'].values
        self.male = test_results[test_results['sex'] == 1]
        self.female = test_results[test_results['sex'] == 0]
        self.genderMetrics = self.biasMetrics(self.male, self.female)

        threshold = self.df['age'].median()
        self.oldPatients = test_results[test_results['age'] >= threshold]
        self.youngPatients = test_results[test_results['age'] < threshold]
        self.ageMetrics = self.biasMetrics(self.youngPatients, self.oldPatients)


    def error_rates(self, df):
        # Calculate the confusion matrix and extract true negatives, false negatives, true positives, and false positives
        cm = confusion_matrix(df['true'], df['pred'], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return {
            "False Positive Rate": fp / (fp + tn),
            "False Negative Rate": fn / (fn + tp)
        }
    
    def plotErrorRates(self, categoryString, data1String, data2String, error_rates_data1, error_rates_data2):
        import matplotlib.pyplot as plt
        groups = [data1String, data2String]
        fnr = [ error_rates_data1['False Negative Rate'], error_rates_data2['False Negative Rate'] ]
        fpr = [ error_rates_data1['False Positive Rate'], error_rates_data2['False Positive Rate'] ]
        x= np.arange(len(groups))
        width = 0.4
        plt.bar(x - width/2, fpr, width, label='False Positive Rate')
        plt.bar(x + width/2, fnr, width, label='False Negative Rate')
        plt.ylabel("Error Rates")
        plt.title(f"Error Rates by {categoryString}")
        plt.xticks(x, groups)
        plt.legend()
        plt.savefig(f'error_rates_{categoryString}.png', dpi=300)
        plt.show()
        plt.close()
        labels = ['False Positive Rate', 'False Negative Rate']
        
    def printAndPlot(self):
        print("Overall Accuracy Metrics:")
        print(f"Overall accuracy: {self.overallAccuracy['Overall Accuracy']}")
        print("Classification Report:")
        print(self.overallAccuracy["Classification Report"])

        print("\nGender Bias Metrics:")
        print(f"Male: Accuracy Metric: {self.genderMetrics['Accuracy1']:.2f}, Recall Metric: {self.genderMetrics['Recall1']:.2f}, False positive: {self.genderMetrics['Error Rates1']['False Positive Rate']:.2f}, False negative: {self.genderMetrics['Error Rates1']['False Negative Rate']:.2f}")
        print(f"Female: Accuracy Metric: {self.genderMetrics['Accuracy2']:.2f}, Recall Metric: {self.genderMetrics['Recall2']:.2f}, False positive: {self.genderMetrics['Error Rates2']['False Positive Rate']:.2f}, False negative: {self.genderMetrics['Error Rates2']['False Negative Rate']:.2f}")
        self.plotErrorRates("Gender", "Male", "Female", self.genderMetrics['Error Rates1'], self.genderMetrics['Error Rates2'])

        print(f"\nAge Bias Metrics: (Young vs Old), median age is {self.df['age'].median()}")
        print(f"Young: Accuracy Metric: {self.ageMetrics['Accuracy1']:.2f}, Recall Metric: {self.ageMetrics['Recall1']:.2f}, False positive: {self.ageMetrics['Error Rates1']['False Positive Rate']:.2f}, False negative: {self.ageMetrics['Error Rates1']['False Negative Rate']:.2f}")
        print(f"Old: Accuracy Metric: {self.ageMetrics['Accuracy2']:.2f}, Recall Metric: {self.ageMetrics['Recall2']:.2f}, False positive: {self.ageMetrics['Error Rates2']['False Positive Rate']:.2f}, False negative: {self.ageMetrics['Error Rates2']['False Negative Rate']:.2f}")
        self.plotErrorRates("Age", "Young", "Old", self.ageMetrics['Error Rates1'], self.ageMetrics['Error Rates2'])



def main():
    ha = HealthDataAnalysis()
    ha.trainModel()
    ha.evaluateBias()
    ha.printAndPlot()



if __name__ == "__main__":
    main()

# Importing necessary libraries
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize 

# Load datasets
train_df = pd.read_csv('kdd_train.csv')
test_df = pd.read_csv('kdd_test.csv')

# Identify and encode categorical columns (example: 'protocol_type', 'service', 'flag') 
categorical_columns = ['protocol_type', 'service', 'flag'] 

# Apply label encoding or one-hot encoding 
for col in categorical_columns: 
    if train_df[col].dtype == 'object': 
        # Label encode or use pd.get_dummies if categories are non-ordinal 
        label_encoder = LabelEncoder() 
        train_df[col] = label_encoder.fit_transform(train_df[col]) 
        test_df[col] = label_encoder.transform(test_df[col]) 

# Fill missing values, if any 
train_df.fillna(0, inplace=True) 
test_df.fillna(0, inplace=True)

# Separate features and target
X_train = train_df.drop('labels', axis=1)
y_train = train_df['labels']
X_test = test_df.drop('labels', axis=1)
y_test = test_df['labels']

#ensure all data is numerical by converting to float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Preprocessing: Label encode target and scale features
y_train = label_encoder.fit_transform(y_train)

try :
    y_test = label_encoder.transform(y_test)
except ValueError as e:
    unseen_labels = set(y_test)-set(label_encoder.classes_)
    print("Unseen labels in test set: {}".format(unseen_labels))

    label_encoder.classes_ = np.concatenate([label_encoder.classes_,list(unseen_labels)])
    y_test = label_encoder.transform(y_test)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define categories for the target variable
categories = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'U2R', 4: 'R2L'}

# Function to train and predict using classifiers in a multilevel fashion
def multilevel_classification(X_train, y_train, X_test, classifiers):
    prediction_vector = []
    
    for clf_name, clf in classifiers.items():
        # Train the classifier
        clf.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = clf.predict(X_test)
        
        prediction_vector.append((clf_name, y_pred))
    
    return prediction_vector

# Define classifiers
classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(),
    'XGBoost': xgb.XGBClassifier(),
}

# Adding ANN
def build_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train ANN
ann_model = build_ann_model(X_train.shape[1])
ann_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Predict using classifiers
predictions = multilevel_classification(X_train, y_train, X_test, classifiers)

# Evaluate models
for clf_name, y_pred in predictions:
    print("Results for {}: ".format(clf_name))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Predict with ANN and evaluate
y_pred_ann = ann_model.predict(X_test)
y_pred_ann = (y_pred_ann > 0.5).astype(int)
print("Results for ANN: ")
print(confusion_matrix(y_test, y_pred_ann))
print(classification_report(y_test, y_pred_ann))

# Plot ROC curves
def plot_roc_curve(y_true, y_pred, clf_name):
    n_classes = len(set(y_train))
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = plt.cm.get_cmap('tab10', n_classes)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Multiclass - {clf_name}')
    plt.legend(loc="lower right")
    
    plt.savefig('roc_curve_{}.png'.format(clf_name))
    plt.show()

# Plot ROC for each classifier
for clf_name, y_pred in predictions:
    # Since classifiers might not have `predict_proba` method, we need to use OneVsRestClassifier
    model = OneVsRestClassifier(classifiers[clf_name])
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    plot_roc_curve(y_test, y_pred_proba, clf_name)

# Plot ROC for ANN
y_pred_proba_ann = ann_model.predict(X_test)
plot_roc_curve(y_test, y_pred_proba_ann, 'ANN')
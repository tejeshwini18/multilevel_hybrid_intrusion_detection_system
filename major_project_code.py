# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv('kdd_train.csv')
test_df = pd.read_csv('kdd_test.csv')

# Separate features and target
X_train = train_df.drop('Label', axis=1)
y_train = train_df['Label']
X_test = test_df.drop('Label', axis=1)
y_test = test_df['Label']

# Preprocessing: Label encode target and scale features
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
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
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(clf_name, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for {}'.format(clf_name))
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC for each classifier
for clf_name, y_pred in predictions:
    plot_roc_curve(y_test, y_pred, clf_name)

# Plot ROC for ANN
plot_roc_curve(y_test, y_pred_ann, 'ANN')

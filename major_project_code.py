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
import tensorflow as tf

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

# Filter out classes with very few samples (less than 10)
class_counts = pd.Series(y_train).value_counts()
valid_classes = class_counts[class_counts >= 10].index
mask_train = np.isin(y_train, valid_classes)
mask_test = np.isin(y_test, valid_classes)

X_train_filtered = X_train[mask_train]
y_train_filtered = y_train[mask_train]
X_test_filtered = X_test[mask_test]
y_test_filtered = y_test[mask_test]

# Create a mapping from old labels to new consecutive labels
label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(valid_classes))}

# Remap the labels
y_train_filtered = np.array([label_mapping[label] for label in y_train_filtered])
y_test_filtered = np.array([label_mapping[label] for label in y_test_filtered])

# Update n_classes after filtering
n_classes = len(valid_classes)

print(f"Number of classes after filtering: {n_classes}")
print("Class mapping:")
for old_label, new_label in label_mapping.items():
    print(f"Original label {old_label} -> New label {new_label}")

classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'XGBoost': xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    ),
}

# Adding ANN
def build_ann_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# Train ANN with early stopping
ann_model = build_ann_model(X_train_filtered.shape[1])
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

ann_model.fit(
    X_train_filtered, y_train_filtered,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Predict using classifiers
predictions = multilevel_classification(X_train_filtered, y_train_filtered, X_test_filtered, classifiers)

# Evaluate models with zero_division=0 to handle classes with no samples
for clf_name, y_pred in predictions:
    print(f"\nResults for {clf_name}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_filtered, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test_filtered, y_pred, zero_division=0))

# Predict with ANN and evaluate
y_pred_ann = ann_model.predict(X_test_filtered)
y_pred_ann = np.argmax(y_pred_ann, axis=1)
print("\nResults for ANN:")
print("Confusion Matrix:")
print(confusion_matrix(y_test_filtered, y_pred_ann))
print("\nClassification Report:")
print(classification_report(y_test_filtered, y_pred_ann, zero_division=0))

# Plot ROC curves
def plot_roc_curve(y_true, y_pred_proba, clf_name):
    n_classes = len(np.unique(y_true))
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = plt.colormaps['tab10'](np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        # Handle NaN values in predictions
        mask = ~np.isnan(y_pred_proba[:, i])
        if np.any(mask):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[mask, i], y_pred_proba[mask, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, 
                    label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Multiclass - {clf_name}')
    plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0))
    
    plt.savefig(f'roc_curve_{clf_name}.png', bbox_inches='tight')
    plt.close()

# Plot ROC for each classifier
for clf_name, _ in predictions:
    try:
        model = OneVsRestClassifier(classifiers[clf_name])
        model.fit(X_train_filtered, y_train_filtered)
        y_pred_proba = model.predict_proba(X_test_filtered)
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0)
        plot_roc_curve(y_test_filtered, y_pred_proba, clf_name)
    except Exception as e:
        print(f"Could not plot ROC curve for {clf_name}: {str(e)}")

# Plot ROC for ANN
try:
    y_pred_proba_ann = ann_model.predict(X_test_filtered)
    y_pred_proba_ann = np.nan_to_num(y_pred_proba_ann, nan=0.0)
    plot_roc_curve(y_test_filtered, y_pred_proba_ann, 'ANN')
except Exception as e:
    print(f"Could not plot ROC curve for ANN: {str(e)}")
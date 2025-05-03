#  Credit Card Fraud Detection System

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Machine Learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, make_scorer, matthews_corrcoef
)
from sklearn.feature_selection import mutual_info_classif

# SMOTE for handling imbalanced data
from imblearn.over_sampling import SMOTE

# SHAP for model explanation
import shap

# Optional: for progress tracking in long operations
from tqdm import tqdm

class CreditCardFraudDetector:
    """
    A comprehensive system for credit card fraud detection using various ML techniques.
    
    This class implements a full pipeline including:
    - Data exploration and visualization
    - Feature engineering
    - Model training with multiple algorithms
    - Advanced evaluation techniques
    - Model explainability
    - Anomaly detection
    - Concept drift monitoring
    - Real-time detection simulation
    """
    
    def __init__(self, data_path):
        """Initialize the fraud detector with data path"""
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_res = None
        self.y_res = None
        
    def load_data(self):
        """Load the credit card dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded with shape: {self.df.shape}")
        return self
        
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n=== Exploratory Data Analysis ===")
        
        # Basic statistics
        print("\nBasic Dataset Information:")
        print(self.df.info())
        
        print("\nSummary Statistics:")
        print(self.df.describe())
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print("\nMissing Values:")
        print(missing_values if missing_values.sum() > 0 else "No missing values")
        
        # Class distribution
        class_dist = self.df['Class'].value_counts()
        print("\nClass Distribution:")
        print(class_dist)
        print(f"Fraud ratio: {class_dist[1] / class_dist.sum():.6f}")
        
        # Visualizations
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Class', data=self.df)
        plt.title('Class Distribution (0: Normal, 1: Fraud)')
        plt.show()
        
        # Amount distribution by class
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df[self.df['Class'] == 0]['Amount'], kde=True)
        plt.title('Amount Distribution - Normal Transactions')
        plt.xlim([0, 2500])
        
        plt.subplot(1, 2, 2)
        sns.histplot(self.df[self.df['Class'] == 1]['Amount'], kde=True, color='red')
        plt.title('Amount Distribution - Fraudulent Transactions')
        plt.xlim([0, 2500])
        plt.tight_layout()
        plt.show()
        
        # Time distribution by class
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df[self.df['Class'] == 0]['Time'], bins=48, kde=True)
        sns.histplot(self.df[self.df['Class'] == 1]['Time'], bins=48, kde=True, color='red')
        plt.title('Time Distribution by Class')
        plt.legend(['Normal', 'Fraud'])
        plt.show()
        
        # Correlation matrix for anonymized features
        plt.figure(figsize=(20, 16))
        corr_matrix = self.df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.show()
        
        # Correlation with target variable
        plt.figure(figsize=(12, 8))
        feature_correlations = corr_matrix['Class'].sort_values(ascending=False)
        sns.barplot(x=feature_correlations.values, y=feature_correlations.index)
        plt.title('Feature Correlation with Target Variable')
        plt.show()
        
        return self
    
    def engineer_features(self):
        """Create additional features to improve model performance"""
        print("\n=== Feature Engineering ===")
        
        # Make a copy of the original dataframe to preserve it
        df_engineered = self.df.copy()
        
        # 1. Transform 'Time' into more meaningful features
        # Convert seconds to datetime assuming a reference date
        reference_date = '2023-01-01'  # Any date can be used as reference
        df_engineered['TransactionDT'] = pd.to_datetime(reference_date) + pd.to_timedelta(df_engineered['Time'], unit='s')
        
        # Extract time-based features
        df_engineered['Hour'] = df_engineered['TransactionDT'].dt.hour
        df_engineered['Day'] = df_engineered['TransactionDT'].dt.day
        df_engineered['DayOfWeek'] = df_engineered['TransactionDT'].dt.dayofweek
        df_engineered['Weekend'] = df_engineered['DayOfWeek'].isin([5, 6]).astype(int)
        
        # 2. Amount-based features
        df_engineered['LogAmount'] = np.log1p(df_engineered['Amount'])
        df_engineered['AmountBin'] = pd.qcut(df_engineered['Amount'], q=10, labels=False, duplicates='drop')
        
        # 3. Transaction velocity (frequency within time windows)
        # This would be more valuable with actual timestamps and card IDs
        # For demonstration purposes, we'll use simulated windows
        window_size = 3600  # seconds (1 hour)
        df_engineered['TransactionVelocity'] = df_engineered['Time'].rolling(window=window_size).count()
        df_engineered['TransactionVelocity'].fillna(0, inplace=True)
        
        # 4. Interactions between features
        # V1-V28 are PCA components, so we can create interactions between important ones
        # Based on correlation with target, let's use top correlated features
        corr_matrix = df_engineered.corr()
        top_features = corr_matrix['Class'].abs().sort_values(ascending=False).head(5).index.tolist()
        top_features = [f for f in top_features if f != 'Class' and f != 'Amount' and f != 'Time']
        
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                df_engineered[f'{feat1}_{feat2}_interaction'] = df_engineered[feat1] * df_engineered[feat2]
        
        # Add feature for amount deviation from mean
        fraud_mean = df_engineered[df_engineered['Class']==1]['Amount'].mean()
        normal_mean = df_engineered[df_engineered['Class']==0]['Amount'].mean()
        
        df_engineered['AmountDeviation_Fraud'] = (df_engineered['Amount'] - fraud_mean).abs()
        df_engineered['AmountDeviation_Normal'] = (df_engineered['Amount'] - normal_mean).abs()
        
        # Remove the datetime column as it's not useful for modeling
        df_engineered.drop('TransactionDT', axis=1, inplace=True)
        
        print(f"Original features: {len(self.df.columns)}")
        print(f"After feature engineering: {len(df_engineered.columns)}")
        
        # Update the dataframe
        self.df_engineered = df_engineered
        
        return self
    
    def prepare_data(self, test_size=0.3, random_state=42):
        """Split data and handle class imbalance"""
        print("\n=== Data Preparation ===")
        
        # Use engineered data if available, otherwise use original
        df_to_use = getattr(self, 'df_engineered', self.df)
        
        # Split features and target
        X = df_to_use.drop('Class', axis=1)
        y = df_to_use['Class']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        
        # Handle class imbalance with SMOTE
        print("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE - Training set shape: {X_res.shape}")
        print(f"Class distribution after SMOTE: {pd.Series(y_res).value_counts()}")
        
        # Store the datasets
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_res = X_res
        self.y_res = y_res
        
        return self
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n=== Model Training ===")
        
        # Define the models to train
        models = {
            'LogisticRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'RandomForest': Pipeline([
                ('model', RandomForestClassifier(n_jobs=-1, random_state=42))
            ]),
            'SGDClassifier': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SGDClassifier(max_iter=1000, tol=1e-3, random_state=42))
            ]),
            'KNN': Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(n_neighbors=5))
            ])
        }
        
        # Define parameter grids for GridSearchCV
        param_grids = {
            'LogisticRegression': {
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga'],
                'model__class_weight': [None, 'balanced']
            },
            'RandomForest': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5],
                'model__class_weight': [None, 'balanced']
            },
            'SGDClassifier': {
                'model__loss': ['log', 'hinge'],
                'model__penalty': ['l1', 'l2'],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__class_weight': [None, 'balanced']
            },
            'KNN': {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2]  # Manhattan or Euclidean
            }
        }
        
        # Define scoring metric - Matthews Correlation Coefficient is good for imbalanced data
        mcc_scorer = make_scorer(matthews_corrcoef)
        
        # Train models with grid search
        for name, pipeline in models.items():
            print(f"\nTraining {name}...")
            grid = GridSearchCV(
                estimator=pipeline, 
                param_grid=param_grids[name],
                scoring=mcc_scorer,
                cv=5,
                n_jobs=-1,
                verbose=1
            )
            
            # Use balanced data for training
            grid.fit(self.X_res, self.y_res)
            
            # Store the best model
            self.models[name] = grid
            
            # Print results
            print(f"Best parameters: {grid.best_params_}")
            print(f"Best cross-validation score: {grid.best_score_:.4f}")
            
            # Evaluate on test set
            y_pred = grid.predict(self.X_test)
            print(f"Test set results:")
            print(f"  Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
            print(f"  Precision: {precision_score(self.y_test, y_pred):.4f}")
            print(f"  Recall: {recall_score(self.y_test, y_pred):.4f}")
            print(f"  F1 Score: {f1_score(self.y_test, y_pred):.4f}")
            print(f"  AUC-ROC: {roc_auc_score(self.y_test, y_pred):.4f}")
            print(f"  MCC: {matthews_corrcoef(self.y_test, y_pred):.4f}")
        
        # Find the best overall model based on test set performance
        model_performances = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            model_performances[name] = f1_score(self.y_test, y_pred)
        
        best_model_name = max(model_performances, key=model_performances.get)
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest overall model: {best_model_name} with F1 Score: {model_performances[best_model_name]:.4f}")
        
        return self
    
    def evaluate_models(self):
        """Perform advanced evaluation of trained models"""
        print("\n=== Advanced Model Evaluation ===")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}:")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Classification report
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {name}')
            plt.show()
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(self.y_test, y_prob):.4f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.show()
            
            # Precision-Recall Curve (better for imbalanced data)
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'F1 = {f1_score(self.y_test, y_pred):.4f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {name}')
            plt.legend()
            plt.show()
            
            # Cost analysis - assuming different costs for false positives and false negatives
            tn, fp, fn, tp = cm.ravel()
            # Assume a false negative (missing fraud) costs 10x more than false positive
            cost_fp = 1  # Cost of false positive (legitimate transaction flagged as fraud)
            cost_fn = 10  # Cost of false negative (missed fraud)
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            
            print(f"Cost Analysis (FP={cost_fp}, FN={cost_fn}):")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  Total Cost: {total_cost}")
            
        return self
    
    def perform_anomaly_detection(self):
        """Complement classification with anomaly detection techniques"""
        print("\n=== Anomaly Detection ===")
        
        # Initialize anomaly detection models
        anomaly_models = {
            'IsolationForest': IsolationForest(
                contamination=0.01,  # Expected proportion of anomalies
                random_state=42
            ),
            'LocalOutlierFactor': LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.01,
                novelty=True
            )
        }
        
        # Train anomaly detection models on legitimate transactions only
        X_train_normal = self.X_train[self.y_train == 0]
        
        # Scale the data
        scaler = StandardScaler()
        X_train_normal_scaled = scaler.fit_transform(X_train_normal)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Create a DataFrame to store anomaly scores
        anomaly_scores = pd.DataFrame({'True_Class': self.y_test.values})
        
        # Train models and get anomaly scores
        for name, model in anomaly_models.items():
            print(f"Training {name}...")
            
            if name == 'LocalOutlierFactor':
                model.fit(X_train_normal_scaled)
                # Negative scores: lower = more anomalous
                anomaly_scores[f'{name}_score'] = -model.decision_function(X_test_scaled)
            else:
                model.fit(X_train_normal_scaled)
                # Negative scores: lower = more anomalous
                anomaly_scores[f'{name}_score'] = -model.decision_function(X_test_scaled)
        
        # Add best classifier's probability
        best_model_name = max(
            {name: f1_score(self.y_test, model.predict(self.X_test)) 
             for name, model in self.models.items()},
            key=lambda k: {name: f1_score(self.y_test, model.predict(self.X_test)) 
                          for name, model in self.models.items()}[k]
        )
        best_model = self.models[best_model_name]
        anomaly_scores['Classifier_prob'] = best_model.predict_proba(self.X_test)[:, 1]
        
        # Visualize the anomaly scores
        plt.figure(figsize=(14, 8))
        
        # Plot histograms of anomaly scores by class
        for i, col in enumerate([c for c in anomaly_scores.columns if c.endswith('_score')]):
            plt.subplot(1, 2, i+1)
            sns.histplot(
                data=anomaly_scores, 
                x=col, 
                hue='True_Class',
                bins=50,
                alpha=0.7
            )
            plt.title(f'{col} Distribution by Class')
            plt.axvline(anomaly_scores[col].quantile(0.95), color='r', linestyle='--', 
                       label='95% Threshold')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Evaluate hybrid approach (combining classifier with anomaly scores)
        print("\nEvaluating hybrid approach...")
        
        # Create a simple ensemble by averaging normalized scores
        # First normalize all scores to 0-1 range
        for col in [c for c in anomaly_scores.columns if c.endswith('_score') or c.endswith('_prob')]:
            min_val = anomaly_scores[col].min()
            max_val = anomaly_scores[col].max()
            anomaly_scores[f'{col}_norm'] = (anomaly_scores[col] - min_val) / (max_val - min_val)
        
        # Create ensemble score
        norm_cols = [c for c in anomaly_scores.columns if c.endswith('_norm')]
        anomaly_scores['ensemble_score'] = anomaly_scores[norm_cols].mean(axis=1)
        
        # Evaluate at different thresholds
        thresholds = np.linspace(0.1, 0.9, 9)
        results = []
        
        for thresh in thresholds:
            preds = (anomaly_scores['ensemble_score'] > thresh).astype(int)
            results.append({
                'threshold': thresh,
                'precision': precision_score(anomaly_scores['True_Class'], preds),
                'recall': recall_score(anomaly_scores['True_Class'], preds),
                'f1': f1_score(anomaly_scores['True_Class'], preds),
                'accuracy': accuracy_score(anomaly_scores['True_Class'], preds)
            })
        
        results_df = pd.DataFrame(results)
        
        # Plot metrics vs threshold
        plt.figure(figsize=(10, 6))
        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            plt.plot(results_df['threshold'], results_df[metric], marker='o', label=metric)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs. Threshold for Hybrid Approach')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Find optimal threshold based on F1 score
        best_threshold = results_df.loc[results_df['f1'].idxmax(), 'threshold']
        print(f"Optimal threshold based on F1 score: {best_threshold:.2f}")
        
        # Results at optimal threshold
        best_idx = results_df['f1'].idxmax()
        print(f"Best hybrid model results:")
        print(f"  Precision: {results_df.loc[best_idx, 'precision']:.4f}")
        print(f"  Recall: {results_df.loc[best_idx, 'recall']:.4f}")
        print(f"  F1 Score: {results_df.loc[best_idx, 'f1']:.4f}")
        print(f"  Accuracy: {results_df.loc[best_idx, 'accuracy']:.4f}")
        
        # Store the hybrid model information
        self.hybrid_model = {
            'anomaly_models': anomaly_models,
            'classifier': best_model,
            'scaler': scaler,
            'optimal_threshold': best_threshold
        }
        
        return self
    
    def explain_model(self):
        """Explain model predictions using SHAP"""
        print("\n=== Model Explainability with SHAP ===")
        
        # Get the best model
        best_model_name = max(
            {name: f1_score(self.y_test, model.predict(self.X_test)) 
             for name, model in self.models.items()},
            key=lambda k: {name: f1_score(self.y_test, model.predict(self.X_test)) 
                          for name, model in self.models.items()}[k]
        )
        model = self.models[best_model_name]
        
        # For pipeline models, extract the actual model
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            actual_model = model.named_steps['model']
            # For pipeline models, we need to transform the data first
            if 'scaler' in model.named_steps:
                X_test_transformed = model.named_steps['scaler'].transform(self.X_test)
            else:
                X_test_transformed = self.X_test
        else:
            actual_model = model
            X_test_transformed = self.X_test
        
        # Create a sampling of test data to explain (SHAP can be computationally intensive)
        # Include more fraud cases in the sample since they're rare
        fraud_indices = np.where(self.y_test == 1)[0]
        non_fraud_indices = np.where(self.y_test == 0)[0]
        
        # Select all fraud cases and an equal number of non-fraud cases
        n_fraud = len(fraud_indices)
        selected_non_fraud = np.random.choice(non_fraud_indices, min(n_fraud, 100), replace=False)
        selected_indices = np.concatenate([fraud_indices, selected_non_fraud])
        
        X_explain = X_test_transformed[selected_indices]
        if isinstance(X_explain, pd.DataFrame):
            feature_names = X_explain.columns
        else:
            feature_names = self.X_test.columns
            X_explain = pd.DataFrame(X_explain, columns=feature_names)
        
        print(f"Explaining {len(X_explain)} transactions using SHAP...")
        
        try:
            # Create explainer
            if best_model_name == 'RandomForest':
                explainer = shap.TreeExplainer(actual_model)
            else:
                # KernelExplainer works with any model but is slower
                # Create a background dataset for the explainer using training data
                X_train_sample = shap.sample(self.X_train, 100)
                if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                    X_train_sample = model.named_steps['scaler'].transform(X_train_sample)
                explainer = shap.KernelExplainer(actual_model.predict_proba, X_train_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_explain)
            
            # For models that output multiple classes, use the values for class 1 (fraud)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="bar")
            plt.title(f"SHAP Feature Importance - {best_model_name}")
            plt.tight_layout()
            plt.show()
            
            # Detailed summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names)
            plt.title(f"SHAP Summary Plot - {best_model_name}")
            plt.tight_layout()
            plt.show()
            
            # Individual explanation for a fraud case
            fraud_idx = np.where(self.y_test.iloc[selected_indices] == 1)[0][0]
            plt.figure(figsize=(14, 8))
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                shap_values[fraud_idx],
                X_explain.iloc[fraud_idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title("SHAP Force Plot - Fraud Transaction Example")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            print("Continuing with remaining analysis...")
        
        return self
    
    def detect_concept_drift(self, window_size=100):
        """Detect potential concept drift over time"""
        print("\n=== Concept Drift Detection ===")
        
        # Use the best model for drift detection
        best_model = self.best_model
        
        # Sort the test set by time to simulate streaming data
        if 'Time' in self.X_test.columns:
            test_data = pd.concat([self.X_test, self.y_test], axis=1)
            test_data = test_data.sort_values(by='Time')
            
            X_stream = test_data.drop('Class', axis=1)
            y_stream = test_data['Class']
            
            # Track model performance over time
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            window_indices = []
            
            # Process data in windows
            for i in range(0, len(X_stream), window_size):
                # Skip windows that are too small
                if i + window_size > len(X_stream):
                    continue
                    
                window_indices.append(i)
                X_window = X_stream.iloc[i:i+window_size]
                y_window = y_stream.iloc[i:i+window_size]
                
                # Calculate performance metrics on this window
                y_pred = best_model.predict(X_window)
                
                accuracies.append(accuracy_score(y_window, y_pred))
                
                # Handle cases where a class might be missing in the window
                if len(np.unique(y_window)) == 1:
                    precisions.append(np.nan)
                    f1_scores.append(np.nan)
                else:
                    precisions.append(precision_score(y_window, y_pred))
                    recalls.append(recall_score(y_window, y_pred))
                    f1_scores.append(f1_score(y_window, y_pred))
            
            # Plot performance over time
            plt.figure(figsize=(14, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(window_indices, accuracies, 'o-')
            plt.title('Accuracy Over Time')
            plt.xlabel('Window Start Index')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(window_indices, precisions, 'o-')
            plt.title('Precision Over Time')
            plt.xlabel('Window Start Index')
            plt.ylabel('Precision')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.plot(window_indices, recalls, 'o-')
            plt.title('Recall Over Time')
            plt.xlabel('Window Start Index')
            plt.ylabel('Recall')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            plt.plot(window_indices, f1_scores, 'o-')
            plt.title('F1 Score Over Time')
            plt.xlabel('Window Start Index')
            plt.ylabel('F1 Score')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Statistical test for drift detection
            # Calculate mean and standard deviation for the first half
            mid_point = len(accuracies) // 2
            first_half_mean = np.mean(accuracies[:mid_point])
            first_half_std = np.std(accuracies[:mid_point])
            
            # Check if any of the second half points deviate significantly
            drift_detected = False
            for i, acc in enumerate(accuracies[mid_point:]):
                if abs(acc - first_half_mean) > 2 * first_half_std:
                    drift_detected = True
                    print(f"Potential concept drift detected at window {mid_point + i} "
                          f"(index {window_indices[mid_point + i]})")
            
            if not drift_detected:
                print("No significant concept drift detected using the 2-sigma rule.")
            
            # Feature distribution drift
            # Select a few important features
            important_features = self.X_test.columns[:5]  # Placeholder - would use feature importance
            
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(important_features):
                plt.subplot(len(important_features), 1, i+1)
                
                # First half distribution
                first_half_data = X_stream[feature].iloc[:len(X_stream)//2]
                second_half_data = X_stream[feature].iloc[len(X_stream)//2:]
                
                sns.kdeplot(first_half_data, label='First Half')
                sns.kdeplot(second_half_data, label='Second Half')
                
                plt.title(f'Distribution Drift - {feature}')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print("Time feature not available for concept drift analysis.")
        
        return self
    
    def simulate_realtime_detection(self, n_transactions=100):
        """Simulate real-time fraud detection with alerts"""
        print("\n=== Real-time Detection Simulation ===")
        
        # Use the best model or hybrid approach
        model = self.hybrid_model if hasattr(self, 'hybrid_model') else self.best_model
        
        # Create a simulated stream of transactions from test data
        # Mix mostly legitimate with some fraudulent transactions
        fraud_indices = np.where(self.y_test == 1)[0]
        non_fraud_indices = np.where(self.y_test == 0)[0]
        
        # Select a mix of transactions - 5% fraud rate (higher than real-world for illustration)
        n_fraud = int(0.05 * n_transactions)
        n_non_fraud = n_transactions - n_fraud
        
        selected_fraud = np.random.choice(fraud_indices, min(n_fraud, len(fraud_indices)), replace=False)
        selected_non_fraud = np.random.choice(non_fraud_indices, min(n_non_fraud, len(non_fraud_indices)), replace=False)
        
        selected_indices = np.concatenate([selected_fraud, selected_non_fraud])
        np.random.shuffle(selected_indices)  # Shuffle to simulate random ordering
        
        X_stream = self.X_test.iloc[selected_indices]
        y_stream = self.y_test.iloc[selected_indices]
        
        # Create a dataframe to track results
        results = pd.DataFrame({
            'Transaction_ID': range(1, n_transactions+1),
            'True_Label': y_stream.values
        })
        
        # Process transactions one by one
        print("Processing transactions...")
        start_time = time.time()
        
        alerts = []
        scores = []
        
        for i, (idx, transaction) in enumerate(X_stream.iterrows()):
            # Format transaction as a dataframe row
            tx = transaction.values.reshape(1, -1)
            
            # Make prediction
            if hasattr(self, 'hybrid_model'):
                # Use hybrid approach
                scaler = model['scaler']
                tx_scaled = scaler.transform(tx)
                
                # Get classifier probability
                clf_prob = model['classifier'].predict_proba(tx)[0, 1]
                
                # Get anomaly scores
                anomaly_scores = []
                for name, anomaly_model in model['anomaly_models'].items():
                    if name == 'LocalOutlierFactor':
                        score = -anomaly_model.decision_function(tx_scaled)[0]
                    else:
                        score = -anomaly_model.decision_function(tx_scaled)[0]
                    
                    # Normalize score
                    min_val, max_val = 0, 1  # Would use actual min/max from training
                    score_norm = (score - min_val) / (max_val - min_val)
                    anomaly_scores.append(score_norm)
                
                # Calculate ensemble score
                ensemble_score = (np.mean(anomaly_scores) + clf_prob) / 2
                scores.append(ensemble_score)
                
                # Alert if score exceeds threshold
                is_fraud = ensemble_score > model['optimal_threshold']
            else:
                # Use just the classifier
                is_fraud = model.predict(tx)[0] == 1
                if hasattr(model, 'predict_proba'):
                    scores.append(model.predict_proba(tx)[0, 1])
                else:
                    scores.append(int(is_fraud))
            
            # Generate alert for fraud
            if is_fraud:
                alerts.append({
                    'Transaction_ID': i + 1,
                    'Score': scores[-1],
                    'Is_True_Fraud': y_stream.iloc[i] == 1
                })
        
        end_time = time.time()
        
        # Add scores to results
        results['Fraud_Score'] = scores
        
        # Calculate metrics
        true_positives = sum((results['True_Label'] == 1) & (results['Fraud_Score'] > 0.5))
        false_positives = sum((results['True_Label'] == 0) & (results['Fraud_Score'] > 0.5))
        true_negatives = sum((results['True_Label'] == 0) & (results['Fraud_Score'] <= 0.5))
        false_negatives = sum((results['True_Label'] == 1) & (results['Fraud_Score'] <= 0.5))
        
        print(f"Simulation complete. Processed {n_transactions} transactions in {end_time - start_time:.2f} seconds.")
        print(f"Average processing time: {(end_time - start_time) / n_transactions * 1000:.2f} ms per transaction")
        
        print(f"\nSimulation Results:")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  True Negatives: {true_negatives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  Accuracy: {(true_positives + true_negatives) / n_transactions:.4f}")
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            print(f"  Precision: {precision:.4f}")
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            print(f"  Recall: {recall:.4f}")
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"  F1 Score: {f1:.4f}")
        
        # Plot ROC curve for the simulation
        fpr, tpr, thresholds = roc_curve(results['True_Label'], results['Fraud_Score'])
        roc_auc = roc_auc_score(results['True_Label'], results['Fraud_Score'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Real-time Simulation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Plot alerts
        if alerts:
            alert_df = pd.DataFrame(alerts)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(
                alert_df['Transaction_ID'],
                alert_df['Score'],
                c=alert_df['Is_True_Fraud'].map({True: 'red', False: 'blue'}),
                alpha=0.7,
                s=100
            )
            plt.xlabel('Transaction ID')
            plt.ylabel('Fraud Score')
            plt.title('Fraud Alerts')
            plt.colorbar(
                plt.cm.ScalarMappable(
                    cmap=plt.cm.coolwarm,
                    norm=plt.Normalize(0, 1)
                ),
                label='Is True Fraud'
            )
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return self
    
    def deploy_model(self, model_path='fraud_detection_model.pkl'):
        """Save the model for deployment"""
        import pickle
        
        # Use the best model
        model = self.hybrid_model if hasattr(self, 'hybrid_model') else self.best_model
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\nModel saved to {model_path}")
        print("Ready for deployment!")
        
        return self
    
    def run_full_pipeline(self):
        """Run the complete fraud detection pipeline"""
        return (self.load_data()
                .explore_data()
                .engineer_features()
                .prepare_data()
                .train_models()
                .evaluate_models()
                .perform_anomaly_detection()
                .explain_model()
                .detect_concept_drift()
                .simulate_realtime_detection()
                .deploy_model())


# Example usage
if __name__ == "__main__":
    # Set path to the credit card fraud dataset
    data_path = "creditcard.csv"
    
    # Initialize and run the fraud detection system
    fraud_detector = CreditCardFraudDetector(data_path)
    fraud_detector.run_full_pipeline()
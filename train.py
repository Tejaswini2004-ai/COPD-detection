# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load and preprocess data
# def load_and_preprocess(file_path):
#     data = pd.read_csv(r"C:\Users\SRIVIDYA\Desktop\miniproject\finalalldata.csv")
    
#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True)
    
#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]
    
#     # Handle missing values
#     imputer = SimpleImputer(strategy='median')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
#     return X, y

# # Feature engineering
# def engineer_features(X):
#     # Polynomial features
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     poly_features = poly.fit_transform(X)
#     poly_feature_names = poly.get_feature_names_out(X.columns)
#     X_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
    
#     # Interaction terms
#     X["age_bmi"] = X["age"] * X["bmi"]
#     X["age_smoke"] = X["age"] * X["smoke"]
#     X["bmi_smoke"] = X["bmi"] * X["smoke"]
    
#     # Combine original and polynomial features
#     X_combined = pd.concat([X, X_poly], axis=1)
    
#     return X_combined

# # Feature selection
# # def select_features(X, y):
# #     # First pass with SelectKBest
# #     selector = SelectKBest(f_classif, k=min(20, X.shape[1]))  # Adjusting k to avoid index mismatch
# #     X_selected = selector.fit_transform(X, y)
# #     selected_features = X.columns[selector.get_support()]
    
# #     # Second pass with RFE
# #     model = XGBClassifier(random_state=42)
# #     rfe = RFE(model, n_features_to_select=min(10, len(selected_features)))  # Ensuring match with SelectKBest
# #     X_rfe = rfe.fit_transform(X[selected_features], y)
# #     final_features = selected_features[rfe.support_]
    
# #     return X[final_features]
# # Feature selection
# # import numpy as np
# import numpy as np
# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.feature_selection import SelectKBest, f_classif, RFE

# def select_features(X, y):
#     # Ensure we don't select more features than available
#     k = min(30, X.shape[1])  
#     selector = SelectKBest(f_classif, k=k)
#     X_selected = selector.fit_transform(X, y)

#     # Get names of selected features
#     selected_features = X.columns[selector.get_support()]  
#     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#     # Apply RFE only if selected features are enough
#     n_features = min(20, len(selected_features))  
#     model = XGBClassifier(random_state=42)
#     rfe = RFE(model, n_features_to_select=n_features)
#     X_rfe = rfe.fit_transform(X_selected_df, y)

#     # Get final selected features
#     final_features = X_selected_df.columns[rfe.support_]

#     print(f"Final Selected Features: {list(final_features)}")  # Debugging output

#     return X[final_features]

# # Model training and evaluation
# def train_and_evaluate(X, y):
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Define base models
#     xgb = XGBClassifier(random_state=42)
#     lgbm = LGBMClassifier(random_state=42)
#     rf = RandomForestClassifier(random_state=42)
    
#     # Hyperparameter tuning for XGBoost
#     param_grid = {
#         'max_depth': [5, 7, 9],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [500, 1000],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }
    
#     grid_search = GridSearchCV(
#         estimator=xgb,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=StratifiedKFold(3),  # Reduced folds for efficiency
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train_scaled, y_train)
#     best_xgb = grid_search.best_estimator_
    
#     # Create ensemble model
#     estimators = [
#         ('xgb', best_xgb),
#         ('lgbm', lgbm),
#         ('rf', rf)
#     ]
    
#     stack = StackingClassifier(
#         estimators=estimators,
#         final_estimator=XGBClassifier(random_state=42),
#         cv=3
#     )
    
#     stack.fit(X_train_scaled, y_train)
    
#     # Evaluate
#     y_pred = stack.predict(X_test_scaled)
#     y_proba = stack.predict_proba(X_test_scaled)[:, 1]
    
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_proba)
    
#     print("\nModel Performance:")
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     print(f"Precision: {precision * 100:.2f}%")
#     print(f"Recall: {recall * 100:.2f}%")
#     print(f"F1-Score: {f1 * 100:.2f}%")
#     print(f"AUC-ROC: {roc_auc * 100:.2f}%")
    
#     # Plot confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=['No COPD', 'COPD'], 
#                 yticklabels=['No COPD', 'COPD'])
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()
    
#     return stack, scaler

# # Main execution
# def main():
#     dataset_path = "/mnt/data/finalalldata.csv"  # Ensure dataset is in the correct path
#     print("Loading and preprocessing data...")
#     X, y = load_and_preprocess(dataset_path)
    
#     print("\nEngineering features...")
#     X_engineered = engineer_features(X)
    
#     print("\nSelecting best features...")
#     X_selected = select_features(X_engineered, y)
    
#     print("\nTraining and evaluating model...")
#     model, scaler = train_and_evaluate(X_selected, y)
    
#     return model, scaler

# if __name__ == "__main__":
#     model, scaler = main()
    

#second code

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE


# # Load and preprocess data
# def load_and_preprocess(file_path):
#     data = pd.read_csv(r"C:\Users\SRIVIDYA\Desktop\miniproject\finalalldata.csv")
    
#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]
    
#     # Handle missing values
#     imputer = SimpleImputer(strategy='median')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
#     return X, y

# # Feature Engineering
# def engineer_features(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X)
    
#     pca = PCA(n_components=15)  # Reduce dimensions to avoid overfitting
#     X_pca = pca.fit_transform(X_poly)
    
#     return pd.DataFrame(X_pca)

# # Apply SMOTE to balance dataset
# def balance_data(X, y):
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# # Feature Selection
# # def select_features(X, y):
# #     k = min(30, X.shape[1])  
# #     selector = SelectKBest(f_classif, k=k)
# #     X_selected = selector.fit_transform(X, y)

# #     selected_features = X.columns[selector.get_support()]  
# #     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# #     model = XGBClassifier(random_state=42)
# #     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
# #     X_rfe = rfe.fit_transform(X_selected_df, y)
    
# #     final_features = X_selected_df.columns[rfe.support_]

# #     return X[final_features]
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from xgboost import XGBClassifier
# import pandas as pd

# def select_features(X, y):
#     # Ensure we don't select more features than available
#     k = min(30, X.shape[1])  
#     selector = SelectKBest(f_classif, k=k)
#     X_selected = selector.fit_transform(X, y)

#     # Get names of selected features
#     selected_features = X.columns[selector.get_support()]  
#     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#     # Check if enough features exist for RFE
#     if len(selected_features) < 20:
#         print("⚠️ Not enough features for RFE. Returning SelectKBest features.")
#         return X[selected_features]

#     # Apply RFE
#     model = XGBClassifier(random_state=42)
#     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
#     X_rfe = rfe.fit_transform(X_selected_df, y)

#     # Get final selected features
#     final_features = X_selected_df.columns[rfe.support_]

#     print(f"✅ Final Selected Features: {list(final_features)}")  # Debugging output

#     return X[final_features]


# # Model Training and Evaluation
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     xgb = XGBClassifier(random_state=42)
#     lgbm = LGBMClassifier(random_state=42)
#     rf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'max_depth': [7, 9, 11],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [1000, 1500, 2000],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }

#     grid_search = GridSearchCV(
#         estimator=xgb,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=StratifiedKFold(5),
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train_scaled, y_train)
#     best_xgb = grid_search.best_estimator_

#     estimators = [
#         ('xgb', best_xgb),
#         ('lgbm', lgbm),
#         ('rf', rf)
#     ]

#     stack = StackingClassifier(
#         estimators=estimators,
#         final_estimator=XGBClassifier(random_state=42),
#         cv=5
#     )

#     stack.fit(X_train_scaled, y_train)

#     y_pred = stack.predict(X_test_scaled)
#     y_proba = stack.predict_proba(X_test_scaled)[:, 1]

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_proba)

#     print("\n🔥 Model Performance:")
#     print(f"✅ Accuracy: {accuracy * 100:.2f}%")
#     print(f"✅ Precision: {precision * 100:.2f}%")
#     print(f"✅ Recall: {recall * 100:.2f}%")
#     print(f"✅ F1-Score: {f1 * 100:.2f}%")
#     print(f"✅ AUC-ROC: {roc_auc * 100:.2f}%")

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No COPD', 'COPD'],
#                 yticklabels=['No COPD', 'COPD'])
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()

#     return stack, scaler

# # Main Execution
# def main():
#     dataset_path = "/mnt/data/finalalldata.csv"  
#     print("📌 Loading and preprocessing data...")
#     X, y = load_and_preprocess(dataset_path)

#     print("📌 Applying feature engineering...")
#     X_engineered = engineer_features(X)

#     print("📌 Balancing data using SMOTE...")
#     X_balanced, y_balanced = balance_data(X_engineered, y)

#     print("📌 Selecting best features...")
#     X_selected = select_features(X_balanced, y_balanced)

#     print("📌 Training and evaluating model...")
#     model, scaler = train_and_evaluate(X_selected, y_balanced)

#     return model, scaler

# if __name__ == "__main__":
#     model, scaler = main()
   

# third code 


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.impute import KNNImputer
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# from sklearn.decomposition import PCA
# from imblearn.combine import SMOTETomek
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import StackingClassifier, RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

# # ✅ Load and Preprocess Data
# def load_and_preprocess():
#     data = pd.read_csv(r"C:\Users\SRIVIDYA\Desktop\miniproject\finalalldata.csv")
#     print(f"✅ Data Loaded: {data.shape}")

#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')

#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]

#     # Impute missing values with KNN
#     imputer = KNNImputer(n_neighbors=5)
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#     # Feature Scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y

# # ✅ Feature Engineering
# def feature_engineering(X):
#     poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
#     X_poly = poly.fit_transform(X)

#     # Apply PCA for dimensionality reduction
#     pca = PCA(n_components=15)  # Adjust based on dataset
#     X_pca = pca.fit_transform(X_poly)

#     print(f"✅ Feature Engineering Completed: {X_pca.shape}")
#     return X_pca

# # ✅ Balance Data using SMOTE + Tomek
# def balance_data(X, y):
#     smt = SMOTETomek(random_state=42)
#     X_resampled, y_resampled = smt.fit_resample(X, y)
#     print(f"✅ Balanced Data Shape: {X_resampled.shape}")
#     return X_resampled, y_resampled

# # ✅ Model Training with Stacking
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

#     # Define Base Models
#     xgb = XGBClassifier(n_estimators=1500, max_depth=7, learning_rate=0.01, eval_metric='logloss')
#     lgbm = LGBMClassifier(n_estimators=1500, max_depth=7, learning_rate=0.01)
#     cat = CatBoostClassifier(iterations=1500, depth=7, learning_rate=0.01, verbose=0)
#     rf = RandomForestClassifier(n_estimators=1000, max_depth=10)

#     # Stacking Model
#     stack = StackingClassifier(
#         estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
#         final_estimator=rf,
#         cv=10
#     )

#     # Train Model
#     stack.fit(X_train, y_train)
#     y_pred = stack.predict(X_test)

#     # Evaluate Performance
#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, stack.predict_proba(X_test)[:, 1])
    
#     print("\n✅ Final Model Evaluation ===")
#     print(f"✅ Accuracy: {acc:.4f}")
#     print(f"✅ AUC-ROC: {auc:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))

#     # Confusion Matrix
#     cm = sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()

# # ✅ Run Full Model Pipeline
# def main():
#     print("\n🚀 Starting COPD Prediction Model")
#     X, y = load_and_preprocess()
#     X = feature_engineering(X)
#     X, y = balance_data(X, y)
#     train_and_evaluate(X, y)

# if __name__ == "__main__":
#     main()


#  fourth code 
 
# import pandas as pd
# import numpy as np
# import optuna
# import time
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler, PolynomialFeatures
# from sklearn.impute import KNNImputer
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import StackingClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings

# warnings.filterwarnings('ignore')

# # ✅ Load and Preprocess Data
# def load_and_preprocess():
#     data = pd.read_csv(r"C:\Users\SRIVIDYA\Desktop\miniproject\finalalldata.csv")
#     print(f"✅ Data Loaded: {data.shape}")

#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')

#     X = data.drop(columns=["label"])
#     y = data["label"]

#     imputer = KNNImputer(n_neighbors=3, weights='distance')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#     scaler = RobustScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y

# # ✅ Feature Engineering
# def feature_engineering(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X)

#     pca = PCA(n_components=0.98)
#     X_pca = pca.fit_transform(X_poly)

#     print(f"✅ Feature Engineering Completed: {X_pca.shape}")
#     return X_pca

# # ✅ Data Balancing with SMOTE
# def balance_data(X, y):
#     smote = SMOTE(sampling_strategy='auto', random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     print(f"✅ Balanced Data Shape: {X_resampled.shape}")
#     return X_resampled, y_resampled

# # ✅ Hyperparameter Optimization using Optuna (Reduced Trials)
# def tune_hyperparameters(X, y):
#     def objective(trial):
#         params = {
#             'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 300, 500),
#             'xgb_max_depth': trial.suggest_int('xgb_max_depth', 4, 8),
#             'xgb_lr': trial.suggest_float('xgb_lr', 0.01, 0.03),
#             'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 300, 500),
#             'lgb_max_depth': trial.suggest_int('lgb_max_depth', 4, 8),
#             'lgb_lr': trial.suggest_float('lgb_lr', 0.01, 0.03),
#             'cat_iterations': trial.suggest_int('cat_iterations', 300, 500),
#             'cat_depth': trial.suggest_int('cat_depth', 4, 8),
#             'cat_lr': trial.suggest_float('cat_lr', 0.01, 0.03)
#         }

#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

#         xgb = XGBClassifier(
#             n_estimators=params['xgb_n_estimators'],
#             max_depth=params['xgb_max_depth'],
#             learning_rate=params['xgb_lr'],
#             eval_metric='logloss',
#             use_label_encoder=False,
#             random_state=42
#         )

#         lgbm = LGBMClassifier(
#             n_estimators=params['lgb_n_estimators'],
#             max_depth=params['lgb_max_depth'],
#             learning_rate=params['lgb_lr'],
#             random_state=42
#         )

#         cat = CatBoostClassifier(
#             iterations=params['cat_iterations'],
#             depth=params['cat_depth'],
#             learning_rate=params['cat_lr'],
#             verbose=0,
#             random_seed=42
#         )

#         stack = StackingClassifier(
#             estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
#             final_estimator=LGBMClassifier(n_estimators=300),
#             cv=3,
#             n_jobs=-1
#         )

#         stack.fit(X_train, y_train)
#         y_pred = stack.predict(X_val)
#         acc = accuracy_score(y_val, y_pred)

#         return acc

#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=3)  # ✅ Reduced trials to 3 for faster execution
#     print("\n✅ Best Hyperparameters Found:", study.best_params)
#     return study.best_params

# # ✅ Train and Evaluate Model
# def train_and_evaluate(X, y, best_params):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

#     xgb_params = {k.replace("xgb_", ""): v for k, v in best_params.items() if "xgb_" in k}
#     lgb_params = {k.replace("lgb_", ""): v for k, v in best_params.items() if "lgb_" in k}
#     cat_params = {k.replace("cat_", ""): v for k, v in best_params.items() if "cat_" in k}

#     xgb = XGBClassifier(**xgb_params, eval_metric='logloss', random_state=42)
#     lgbm = LGBMClassifier(**lgb_params, random_state=42)
#     cat = CatBoostClassifier(**cat_params, verbose=0, random_seed=42)

#     stack = StackingClassifier(
#         estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
#         final_estimator=LGBMClassifier(n_estimators=300),
#         cv=3,
#         n_jobs=-1
#     )

#     stack.fit(X_train, y_train)
#     y_pred = stack.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, stack.predict_proba(X_test)[:, 1])

#     print("\n✅ Model Evaluation ===")
#     print(f"✅ Accuracy: {acc:.4f}")
#     print(f"✅ AUC-ROC: {auc:.4f}")
#     print(classification_report(y_test, y_pred))

#     plt.figure(figsize=(6,6))
#     sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.show()

# # ✅ Main Function (Runs Entire Pipeline)
# def main():
#     start_time = time.time()
#     print("\n🚀 Running Optimized COPD Prediction Model")
    
#     X, y = load_and_preprocess()
#     X = feature_engineering(X)
#     X, y = balance_data(X, y)
    
#     best_params = tune_hyperparameters(X, y)
#     train_and_evaluate(X, y, best_params)

#     print(f"\n⏳ Execution Time: {time.time() - start_time:.2f} seconds")

# # ✅ Run Script
# if __name__ == "__main__":
#     main()


# fivth code 
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.impute import KNNImputer
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import warnings

# # Import TensorFlow Model Optimization properly
# try:
#     from tensorflow_model_optimization.quantization.keras import quantize_model
# except ImportError:
#     print("\n⚠️ Warning: TensorFlow Model Optimization is not installed. Run: pip install tensorflow-model-optimization\n")
#     quantize_model = None

# warnings.filterwarnings('ignore')

# # 1. Data Loading and Preprocessing
# def load_and_preprocess():
#     data = pd.read_csv(r"C:\\Users\\SRIVIDYA\\Desktop\\miniproject\\finalalldata.csv")
#     print(f"✅ Data Loaded: {data.shape}")
#     print("Class Distribution:\n", data['label'].value_counts())
    
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
#     # Remove near-zero variance features
#     variances = data.drop(columns=['label']).var()
#     low_var_cols = variances[variances < 0.005].index.tolist()
#     data.drop(columns=low_var_cols, inplace=True)
#     print(f"Dropped {len(low_var_cols)} low-variance features")

#     X = data.drop(columns=["label"])
#     y = data["label"]

#     # Advanced imputation
#     imputer = KNNImputer(n_neighbors=7, weights='distance')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#     # Non-linear transformation
#     scaler = QuantileTransformer(output_distribution='normal')
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y

# # 2. Feature Engineering
# def feature_engineering(X):
#     stats = np.column_stack([
#         X.mean(axis=1),
#         X.std(axis=1),
#         X.max(axis=1) - X.min(axis=1),
#         np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)
#     ])
    
#     X_enhanced = np.column_stack([X, stats])
    
#     # Dimensionality reduction
#     pca = PCA(n_components=0.99)
#     X_pca = pca.fit_transform(X_enhanced)
#     print(f"✅ Feature Engineering Completed: {X_pca.shape}")
#     return X_pca

# # 3. Neural Network Model
# def build_model(input_shape, quantize=False):
#     inputs = Input(shape=(input_shape,))
#     x = Dense(128, activation='swish', kernel_regularizer='l2')(inputs)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation='swish', kernel_regularizer='l2')(x)
#     x = Dropout(0.2)(x)
#     outputs = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=inputs, outputs=outputs)
    
#     if quantize and quantize_model:
#         model = quantize_model(model)
    
#     model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # 4. Training and Evaluation
# def train_and_evaluate(X, y):
#     X, y = SMOTE(sampling_strategy=0.95, random_state=42).fit_resample(X, y)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
    
#     model = build_model(X_train.shape[1], quantize=False)
#     early_stop = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
    
#     print("\n🚀 Training Neural Network...")
#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=32, callbacks=[early_stop], verbose=1)
    
#     # Find optimal threshold
#     y_proba = model.predict(X_test).flatten()
#     thresholds = np.linspace(0.3, 0.7, 100)
#     accuracies = [accuracy_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
#     optimal_threshold = thresholds[np.argmax(accuracies)]
#     y_pred = (y_proba >= optimal_threshold).astype(int)
    
#     print(f"\n🔥 Final Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Optimal Threshold: {optimal_threshold:.3f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
    
#     # Plot training history
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.show()

# # 5. Main Function
# def main():
#     print("\n🚀 Starting Neural Network for COPD Prediction")
#     X, y = load_and_preprocess()
#     X = feature_engineering(X)
#     train_and_evaluate(X, y)

# if __name__ == "__main__":
#     main()



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.impute import KNNImputer
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# from sklearn.decomposition import PCA
# from imblearn.combine import SMOTETomek
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import StackingClassifier, RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

# # ✅ Load and Preprocess Data
# def load_and_preprocess():
#     data = pd.read_csv(r"C:\Users\SRIVIDYA\Desktop\miniproject\finalalldata.csv")
#     print(f"✅ Data Loaded: {data.shape}")

#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')

#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]

#     # Impute missing values with KNN
#     imputer = KNNImputer(n_neighbors=5)
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#     # Feature Scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y

# # ✅ Feature Engineering
# def feature_engineering(X):
#     poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
#     X_poly = poly.fit_transform(X)

#     # Apply PCA for dimensionality reduction
#     pca = PCA(n_components=15)  # Adjust based on dataset
#     X_pca = pca.fit_transform(X_poly)

#     print(f"✅ Feature Engineering Completed: {X_pca.shape}")
#     return X_pca

# # ✅ Balance Data using SMOTE + Tomek
# def balance_data(X, y):
#     smt = SMOTETomek(random_state=42)
#     X_resampled, y_resampled = smt.fit_resample(X, y)
#     print(f"✅ Balanced Data Shape: {X_resampled.shape}")
#     return X_resampled, y_resampled

# # ✅ Model Training with Stacking
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

#     # Define Base Models
#     xgb = XGBClassifier(n_estimators=1500, max_depth=7, learning_rate=0.01, eval_metric='logloss')
#     lgbm = LGBMClassifier(n_estimators=1500, max_depth=7, learning_rate=0.01)
#     cat = CatBoostClassifier(iterations=1500, depth=7, learning_rate=0.01, verbose=0)
#     rf = RandomForestClassifier(n_estimators=1000, max_depth=10)

#     # Stacking Model
#     stack = StackingClassifier(
#         estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
#         final_estimator=rf,
#         cv=10
#     )

#     # Train Model
#     stack.fit(X_train, y_train)
#     y_pred = stack.predict(X_test)

#     # Evaluate Performance
#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, stack.predict_proba(X_test)[:, 1])
    
#     print("\n✅ Final Model Evaluation ===")
#     print(f"✅ Accuracy: {acc:.4f}")
#     print(f"✅ AUC-ROC: {auc:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))

#     # Confusion Matrix
#     cm = sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()

# # ✅ Run Full Model Pipeline
# def main():
#     print("\n🚀 Starting COPD Prediction Model")
#     X, y = load_and_preprocess()
#     X = feature_engineering(X)
#     X, y = balance_data(X, y)
#     train_and_evaluate(X, y)

# if __name__ == "__main__":  
#     main()




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE


# # Load and preprocess data
# def load_and_preprocess(file_path):
#     data = pd.read_csv(r"C:\Users\SRIVIDYA\Desktop\miniproject\finalalldata.csv")
    
#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]
    
#     # Handle missing values
#     imputer = SimpleImputer(strategy='median')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
#     return X, y

# # Feature Engineering
# def engineer_features(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X)
    
#     pca = PCA(n_components=15)  # Reduce dimensions to avoid overfitting
#     X_pca = pca.fit_transform(X_poly)
    
#     return pd.DataFrame(X_pca)

# # Apply SMOTE to balance dataset
# def balance_data(X, y):
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# # Feature Selection
# # def select_features(X, y):
# #     k = min(30, X.shape[1])  
# #     selector = SelectKBest(f_classif, k=k)
# #     X_selected = selector.fit_transform(X, y)

# #     selected_features = X.columns[selector.get_support()]  
# #     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# #     model = XGBClassifier(random_state=42)
# #     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
# #     X_rfe = rfe.fit_transform(X_selected_df, y)
    
# #     final_features = X_selected_df.columns[rfe.support_]

# #     return X[final_features]
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from xgboost import XGBClassifier
# import pandas as pd

# def select_features(X, y):
#     # Ensure we don't select more features than available
#     k = min(30, X.shape[1])  
#     selector = SelectKBest(f_classif, k=k)
#     X_selected = selector.fit_transform(X, y)

#     # Get names of selected features
#     selected_features = X.columns[selector.get_support()]  
#     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#     # Check if enough features exist for RFE
#     if len(selected_features) < 20:
#         print("⚠ Not enough features for RFE. Returning SelectKBest features.")
#         return X[selected_features]

#     # Apply RFE
#     model = XGBClassifier(random_state=42)
#     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
#     X_rfe = rfe.fit_transform(X_selected_df, y)

#     # Get final selected features
#     final_features = X_selected_df.columns[rfe.support_]

#     print(f"✅ Final Selected Features: {list(final_features)}")  # Debugging output

#     return X[final_features]


# # Model Training and Evaluation
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     xgb = XGBClassifier(random_state=42)
#     lgbm = LGBMClassifier(random_state=42)
#     rf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'max_depth': [7, 9, 11],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [1000, 1500, 2000],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }

#     grid_search = GridSearchCV(
#         estimator=xgb,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=StratifiedKFold(5),
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train_scaled, y_train)
#     best_xgb = grid_search.best_estimator_

#     estimators = [
#         ('xgb', best_xgb),
#         ('lgbm', lgbm),
#         ('rf', rf)
#     ]

#     stack = StackingClassifier(
#         estimators=estimators,
#         final_estimator=XGBClassifier(random_state=42),
#         cv=5
#     )

#     stack.fit(X_train_scaled, y_train)

#     y_pred = stack.predict(X_test_scaled)
#     y_proba = stack.predict_proba(X_test_scaled)[:, 1]

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_proba)

#     print("\n🔥 Model Performance:")
#     print(f"✅ Accuracy: {accuracy * 100:.2f}%")
#     print(f"✅ Precision: {precision * 100:.2f}%")
#     print(f"✅ Recall: {recall * 100:.2f}%")
#     print(f"✅ F1-Score: {f1 * 100:.2f}%")
#     print(f"✅ AUC-ROC: {roc_auc * 100:.2f}%")

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No COPD', 'COPD'],
#                 yticklabels=['No COPD', 'COPD'])
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()

#     return stack, scaler

# # Main Execution
# def main():
#     dataset_path = "/mnt/data/finalalldata.csv"  
#     print("📌 Loading and preprocessing data...")
#     X, y = load_and_preprocess(dataset_path)

#     print("📌 Applying feature engineering...")
#     X_engineered = engineer_features(X)

#     print("📌 Balancing data using SMOTE...")
#     X_balanced, y_balanced = balance_data(X_engineered, y)

#     print("📌 Selecting best features...")
#     X_selected = select_features(X_balanced, y_balanced)

#     print("📌 Training and evaluating model...")
#     model, scaler = train_and_evaluate(X_selected, y_balanced)

#     return model, scaler

# if __name__ == "__main__":
#     model, scaler = main()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
# import joblib  # for saving model and scaler

# # Load and preprocess data
# def load_and_preprocess(file_path):
#     data = pd.read_csv(r"C:\Users\SRIVIDYA\Desktop\miniproject\finalalldata.csv")
    
#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]
    
#     # Handle missing values
#     imputer = SimpleImputer(strategy='median')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
#     return X, y

# # Feature Engineering
# def engineer_features(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X)
    
#     pca = PCA(n_components=15)  # Reduce dimensions to avoid overfitting
#     X_pca = pca.fit_transform(X_poly)
    
#     return pd.DataFrame(X_pca)

# # Apply SMOTE to balance dataset
# def balance_data(X, y):
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# # Feature Selection
# def select_features(X, y):
#     k = min(30, X.shape[1])  
#     selector = SelectKBest(f_classif, k=k)
#     X_selected = selector.fit_transform(X, y)

#     # Get names of selected features
#     selected_features = X.columns[selector.get_support()]  
#     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#     # Check if enough features exist for RFE
#     if len(selected_features) < 20:
#         print("⚠ Not enough features for RFE. Returning SelectKBest features.")
#         return X[selected_features]

#     # Apply RFE
#     model = XGBClassifier(random_state=42)
#     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
#     X_rfe = rfe.fit_transform(X_selected_df, y)

#     # Get final selected features
#     final_features = X_selected_df.columns[rfe.support_]

#     print(f"✅ Final Selected Features: {list(final_features)}")  # Debugging output

#     return X[final_features]

# # Model Training and Evaluation
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     xgb = XGBClassifier(random_state=42)
#     lgbm = LGBMClassifier(random_state=42)
#     rf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'max_depth': [7, 9, 11],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [1000, 1500, 2000],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }

#     grid_search = GridSearchCV(
#         estimator=xgb,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=StratifiedKFold(5),
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train_scaled, y_train)
#     best_xgb = grid_search.best_estimator_

#     estimators = [
#         ('xgb', best_xgb),
#         ('lgbm', lgbm),
#         ('rf', rf)
#     ]

#     stack = StackingClassifier(
#         estimators=estimators,
#         final_estimator=XGBClassifier(random_state=42),
#         cv=5
#     )

#     stack.fit(X_train_scaled, y_train)

#     y_pred = stack.predict(X_test_scaled)
#     y_proba = stack.predict_proba(X_test_scaled)[:, 1]

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_proba)

#     print("\n🔥 Model Performance:")
#     print(f"✅ Accuracy: {accuracy * 100:.2f}%")
#     print(f"✅ Precision: {precision * 100:.2f}%")
#     print(f"✅ Recall: {recall * 100:.2f}%")
#     print(f"✅ F1-Score: {f1 * 100:.2f}%")
#     print(f"✅ AUC-ROC: {roc_auc * 100:.2f}%")

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No COPD', 'COPD'],
#                 yticklabels=['No COPD', 'COPD'])
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()

#     # Save the model and scaler
#     joblib.dump(stack, 'copd_model.pkl')  # Save the model
#     joblib.dump(scaler, 'scaler.pkl')     # Save the scaler

#     print("✅ Model and scaler saved successfully!")

#     return stack, scaler

# # Main Execution
# def main():
#     dataset_path = "/mnt/data/finalalldata.csv"  
#     print("📌 Loading and preprocessing data...")
#     X, y = load_and_preprocess(dataset_path)

#     print("📌 Applying feature engineering...")
#     X_engineered = engineer_features(X)

#     print("📌 Balancing data using SMOTE...")
#     X_balanced, y_balanced = balance_data(X_engineered, y)

#     print("📌 Selecting best features...")
#     X_selected = select_features(X_balanced, y_balanced)

#     print("📌 Training and evaluating model...")
#     model, scaler = train_and_evaluate(X_selected, y_balanced)

#     return model, scaler

# if __name__ == "__main__":
#     model, scaler = main()





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
# import joblib  # for saving model and scaler

# # Load and preprocess data
# def load_and_preprocess(file_path):
#     data = pd.read_csv(file_path)
    
#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]
    
#     # Handle missing values
#     imputer = SimpleImputer(strategy='median')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
#     return X, y

# # Feature Engineering
# def engineer_features(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X)
    
#     pca = PCA(n_components=15)  # Reduce dimensions to avoid overfitting
#     X_pca = pca.fit_transform(X_poly)
    
#     return pd.DataFrame(X_pca)

# # Apply SMOTE to balance dataset
# def balance_data(X, y):
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# # Feature Selection
# def select_features(X, y):
#     k = min(30, X.shape[1])  
#     selector = SelectKBest(f_classif, k=k)
#     X_selected = selector.fit_transform(X, y)

#     # Get names of selected features
#     selected_features = X.columns[selector.get_support()]  
#     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#     # Check if enough features exist for RFE
#     if len(selected_features) < 20:
#         print("⚠ Not enough features for RFE. Returning SelectKBest features.")
#         return X[selected_features]

#     # Apply RFE
#     model = XGBClassifier(random_state=42)
#     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
#     X_rfe = rfe.fit_transform(X_selected_df, y)

#     # Get final selected features
#     final_features = X_selected_df.columns[rfe.support_]

#     print(f"✅ Final Selected Features: {list(final_features)}")  # Debugging output

#     return X[final_features]

# # Model Training and Evaluation
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     xgb = XGBClassifier(random_state=42)
#     lgbm = LGBMClassifier(random_state=42)
#     rf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'max_depth': [7, 9, 11],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [1000, 1500, 2000],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }

#     grid_search = GridSearchCV(
#         estimator=xgb,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=StratifiedKFold(5),
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train_scaled, y_train)
#     best_xgb = grid_search.best_estimator_

#     estimators = [
#         ('xgb', best_xgb),
#         ('lgbm', lgbm),
#         ('rf', rf)
#     ]

#     stack = StackingClassifier(
#         estimators=estimators,
#         final_estimator=XGBClassifier(random_state=42),
#         cv=5
#     )

#     stack.fit(X_train_scaled, y_train)

#     # Save the model and scaler
#     joblib.dump(stack, 'copd_model.pkl')  # Save the model
#     joblib.dump(scaler, 'scaler.pkl')     # Save the scaler

#     print("✅ Model and scaler saved successfully!")

#     return stack, scaler

# # Predict function for a single patient input
# def predict_copd(model, scaler, patient_data):
#     # Assuming patient_data is a dictionary with input details
#     patient_df = pd.DataFrame([patient_data])
    
#     # Preprocess the data (apply same scaler)
#     patient_scaled = scaler.transform(patient_df)
    
#     # Get prediction
#     prediction = model.predict(patient_scaled)
#     prediction_prob = model.predict_proba(patient_scaled)[:, 1]  # Probability for COPD (class 1)
    
#     return prediction[0], prediction_prob[0]

# # Main Execution
# def main():
#     dataset_path = "/mnt/data/finalalldata.csv"  
#     print("📌 Loading and preprocessing data...")
#     X, y = load_and_preprocess(dataset_path)

#     print("📌 Applying feature engineering...")
#     X_engineered = engineer_features(X)

#     print("📌 Balancing data using SMOTE...")
#     X_balanced, y_balanced = balance_data(X_engineered, y)

#     print("📌 Selecting best features...")
#     X_selected = select_features(X_balanced, y_balanced)

#     print("📌 Training and evaluating model...")
#     model, scaler = train_and_evaluate(X_selected, y_balanced)

#     return model, scaler

# if __name__ == "__main__":
#     model, scaler = main()

#     # Example: Patient data input
#     patient_data = {
#         'sex': 1,         # Female
#         'age': 50,        # 50 years old
#         'bmi': 24,        # BMI value
#         'smoke': 1,       # Smokes
#         'rs10007052': 0.5,
#         'rs8192288': 0.6,
#         'rs20541': 0.7,
#         'rs12922394': 0.8,
#         'rs2910164': 0.5,
#         'rs161976': 0.6,
#         'rs473892': 0.9,
#         'rs159497': 0.7,
#         'rs9296092': 0.8
#     }

#     prediction, prob = predict_copd(model, scaler, patient_data)
#     print(f"Prediction: {'COPD' if prediction == 1 else 'No COPD'}, Probability: {prob:.2f}")



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE  # <-- This line
# import joblib  # for saving model and scaler

# # Load and preprocess data
# def load_and_preprocess(file_path):
#     data = pd.read_csv(r'C:\Users\SRIVIDYA\Desktop\miniproject\data\finalalldata.csv')
    
#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]
    
#     # Handle missing values
#     imputer = SimpleImputer(strategy='median')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
#     return X, y

# # Feature Engineering
# def engineer_features(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X)
    
#     pca = PCA(n_components=15)  # Reduce dimensions to avoid overfitting
#     X_pca = pca.fit_transform(X_poly)
    
#     return pd.DataFrame(X_pca)

# # Apply SMOTE to balance dataset
# def balance_data(X, y):
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# # Feature Selection
# def select_features(X, y):
#     k = min(30, X.shape[1])  
#     selector = SelectKBest(f_classif, k=k)
#     X_selected = selector.fit_transform(X, y)

#     # Get names of selected features
#     selected_features = X.columns[selector.get_support()]  
#     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#     # Check if enough features exist for RFE
#     if len(selected_features) < 20:
#         print("⚠ Not enough features for RFE. Returning SelectKBest features.")
#         return X[selected_features]

#     # Apply RFE
#     model = XGBClassifier(random_state=42)
#     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
#     X_rfe = rfe.fit_transform(X_selected_df, y)

#     # Get final selected features
#     final_features = X_selected_df.columns[rfe.support_]

#     print(f"✅ Final Selected Features: {list(final_features)}")  # Debugging output

#     return X[final_features]

# # Model Training and Evaluation
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     xgb = XGBClassifier(random_state=42)
#     lgbm = LGBMClassifier(random_state=42)
#     rf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'max_depth': [7, 9, 11],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [1000, 1500, 2000],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }

#     grid_search = GridSearchCV(
#         estimator=xgb,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=StratifiedKFold(5),
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train_scaled, y_train)
#     best_xgb = grid_search.best_estimator_

#     estimators = [
#         ('xgb', best_xgb),
#         ('lgbm', lgbm),
#         ('rf', rf)
#     ]

#     stack = StackingClassifier(
#         estimators=estimators,
#         final_estimator=XGBClassifier(random_state=42),
#         cv=5
#     )

#     stack.fit(X_train_scaled, y_train)

#     # Save the model and scaler
#     joblib.dump(stack, 'copd_model.pkl')  # Save the model
#     joblib.dump(scaler, 'scaler.pkl')     # Save the scaler

#     print("✅ Model and scaler saved successfully!")

#     return stack, scaler

# # Main Execution
# def main():
#     dataset_path = "/mnt/data/finalalldata.csv"  
#     print("📌 Loading and preprocessing data...")
#     X, y = load_and_preprocess(dataset_path)

#     print("📌 Applying feature engineering...")
#     X_engineered = engineer_features(X)

#     print("📌 Balancing data using SMOTE...")
#     X_balanced, y_balanced = balance_data(X_engineered, y)

#     print("📌 Selecting best features...")
#     X_selected = select_features(X_balanced, y_balanced)

#     print("📌 Training and evaluating model...")
#     model, scaler = train_and_evaluate(X_selected, y_balanced)

#     return model, scaler

# if __name__ == "__main__":
#     model, scaler = main()





#train themodel
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif, RFE
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE  # <-- This line
# import joblib  # for saving model and scaler

# # Load and preprocess data
# def load_and_preprocess(file_path):
#     data = pd.read_csv(r'C:\Users\SRIVIDYA\Desktop\miniproject\data\finalalldata.csv')
    
#     # Drop unnecessary columns
#     data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
#     # Separate features and target
#     X = data.drop(columns=["label"])
#     y = data["label"]
    
#     # Handle missing values
#     imputer = SimpleImputer(strategy='median')
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
#     return X, y

# # Feature Engineering
# def engineer_features(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X)
    
#     pca = PCA(n_components=15)  # Reduce dimensions to avoid overfitting
#     X_pca = pca.fit_transform(X_poly)
    
#     return pd.DataFrame(X_pca)

# # Apply SMOTE to balance dataset
# def balance_data(X, y):
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# # Feature Selection
# def select_features(X, y):
#     k = min(30, X.shape[1])  
#     selector = SelectKBest(f_classif, k=k)
#     X_selected = selector.fit_transform(X, y)

#     # Get names of selected features
#     selected_features = X.columns[selector.get_support()]  
#     X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

#     # Check if enough features exist for RFE
#     if len(selected_features) < 20:
#         print("⚠ Not enough features for RFE. Returning SelectKBest features.")
#         return X[selected_features]

#     # Apply RFE
#     model = XGBClassifier(random_state=42)
#     rfe = RFE(model, n_features_to_select=min(20, len(selected_features)))
#     X_rfe = rfe.fit_transform(X_selected_df, y)

#     # Get final selected features
#     final_features = X_selected_df.columns[rfe.support_]

#     print(f"✅ Final Selected Features: {list(final_features)}")  # Debugging output

#     return X[final_features]

# # Model Training and Evaluation
# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     xgb = XGBClassifier(random_state=42)
#     lgbm = LGBMClassifier(random_state=42)
#     rf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'max_depth': [7, 9, 11],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [1000, 1500, 2000],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }

#     grid_search = GridSearchCV(
#         estimator=xgb,
#         param_grid=param_grid,
#         scoring='accuracy',
#         cv=StratifiedKFold(5),
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train_scaled, y_train)
#     best_xgb = grid_search.best_estimator_

#     estimators = [
#         ('xgb', best_xgb),
#         ('lgbm', lgbm),
#         ('rf', rf)
#     ]

#     stack = StackingClassifier(
#         estimators=estimators,
#         final_estimator=XGBClassifier(random_state=42),
#         cv=5
#     )

#     stack.fit(X_train_scaled, y_train)

#     # Predicting on the test set
#     y_pred = stack.predict(X_test_scaled)
#     y_pred_proba = stack.predict_proba(X_test_scaled)[:, 1]  # Probability for COPD (class 1)

#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred_proba)
#     conf_matrix = confusion_matrix(y_test, y_pred)

#     # Print the metrics
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"ROC AUC Score: {roc_auc:.4f}")
#     print(f"Confusion Matrix:\n{conf_matrix}")

#     # Save the model and scaler
#     joblib.dump(stack, 'copd_model.pkl')  # Save the model
#     joblib.dump(scaler, 'scaler.pkl')     # Save the scaler

#     print("✅ Model and scaler saved successfully!")

#     return stack, scaler

# # Main Execution
# def main():
#     dataset_path = "/mnt/data/finalalldata.csv"  
#     print("📌 Loading and preprocessing data...")
#     X, y = load_and_preprocess(dataset_path)

#     print("📌 Applying feature engineering...")
#     X_engineered = engineer_features(X)

#     print("📌 Balancing data using SMOTE...")
#     X_balanced, y_balanced = balance_data(X_engineered, y)

#     print("📌 Selecting best features...")
#     X_selected = select_features(X_balanced, y_balanced)

#     print("📌 Training and evaluating model...")
#     model, scaler = train_and_evaluate(X_selected, y_balanced)

#     return model, scaler

# if __name__ == "__main__":
#     model, scaler = main()   # Run the script and print all metrics

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import joblib

# Load and preprocess data
def load_and_preprocess(file_path):
    data = pd.read_csv(r'C:\Users\SRIVIDYA\Desktop\miniproject\data\finalalldata.csv')
    
    # Drop unnecessary columns
    data.drop(columns=["uid", "class"], inplace=True, errors='ignore')
    
    # Separate features and target
    X = data.drop(columns=["label"])
    y = data["label"]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, y

# Feature Engineering
def engineer_features(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    pca = PCA(n_components=15)  # Reduce dimensions to avoid overfitting
    X_pca = pca.fit_transform(X_poly)
    
    return X_pca, pca  # Return both the transformed data and the fitted PCA

# Train Model and Save Everything
def train_and_save(X, y, pca):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb = XGBClassifier(random_state=42)
    lgbm = LGBMClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)

    estimators = [
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('rf', rf)
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(random_state=42),
        cv=5
    )

    stack.fit(X_train_scaled, y_train)

    # Save the model, scaler, and PCA
    joblib.dump(stack, 'copd_model.pkl')  # Save the trained model
    joblib.dump(scaler, 'scaler.pkl')     # Save the scaler
    joblib.dump(pca, 'pca.pkl')           # Save the PCA

    print("Model, scaler, and PCA saved successfully!")

    # Evaluate the model
    y_pred = stack.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Main Execution
def main():
    #dataset_path = "C:\Users\SRIVIDYA\Desktop\miniproject\data\finalalldata.csv"  # Update with the actual path
    dataset_path = "C:\\Users\\SRIVIDYA\\Desktop\\miniproject\\data\\finalalldata.csv"

    X, y = load_and_preprocess(dataset_path)
    X_pca, pca = engineer_features(X)
    train_and_save(X_pca, y, pca)  # Pass the pca object here

if __name__ == "__main__":
    main()

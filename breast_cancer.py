import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

file = "FILEPATH/data.csv"
df = pd.read_csv(file).sample(frac=1).reset_index(drop=True)
print(df.info())
print(df.isnull().sum())
df.drop(columns = ['id', 'Unnamed: 32'], inplace=True)
df['diagnosis'] = df['diagnosis'].replace({'B': 0, 'M': 1})

def dist_plots(data):
    num_cols = len(data.columns)
    num_rows = (num_cols + 4) // 5  # Adjusting for the number of rows needed

    fig, axes = plt.subplots(nrows=num_rows, ncols=8, figsize=(15, num_rows*1.75))
    axes = axes.flatten()

    for i, col in enumerate(data.columns):
        ax = axes[i]
        if data[col].nunique() < 3:  # Assuming categorical if fewer than 3 unique values
            sns.countplot(data=data, x=col, ax=ax)
        else:
            sns.histplot(data=data, x=col, bins=25, kde=True, ax=ax)

    # Turn off extra subplots
    for i in range(num_cols, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

dist_plots(df)

# We can see that most of the variables follow normal disribution, there are no NA values, and no extreme outliers.
# area_se seems to be dominated by smaller values.  However, there are no 0 values.

# Creating a heatmap for linear correlations
sns.heatmap(df.iloc[:,1:].corr(), cmap='coolwarm')
plt.yticks(size = 3)
plt.xticks(size = 3)
plt.show()

# It looks like the the radius texture and perimeter are storngly correlated with area
# Smoothness and concavity are correlated with symmetry

def lin_plots(data):
    correlations = data.iloc[:, 1:].corr()
    num_pairs = sum(1 for i in range(len(correlations.columns)) for j in range(i + 1, len(correlations.columns)) if abs(correlations.iloc[i, j]) > 0.7)
    num_rows = (num_pairs + 7) // 8  # Adjusting for the number of rows needed
    fig, axes = plt.subplots(nrows=num_rows, ncols=8, figsize=(15, num_rows*0.8))
    axes = axes.flatten()

    plot_count = 0
    for i, col in enumerate(correlations.columns):
        for j in range(i + 1, len(correlations.columns)):
            if abs(correlations.iloc[i, j]) > 0.7 and correlations.iloc[i,j] < 1:
                ax = axes[plot_count]
                sns.scatterplot(data=correlations, x=correlations.columns[i], y=correlations.columns[j], ax=ax)
                plot_count += 1
                if plot_count >= num_pairs:
                    break  # Exit the loop if all required plots are done
        else:
            continue
        break

    # Turn off extra subplots
    for i in range(plot_count, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Lineplots EDA')
    plt.tight_layout()
    plt.show()

lin_plots(df)

# Looks like most of the measurements are highly correlated.  With all or most having a significant correlation (greater than 0.7)

# I will move on to feature reduction and model selection in a pipeline format
X = df.drop(columns=['diagnosis'])
y = df.diagnosis

def model_pipeline(X_data, y_data, model, p_grid):
    steps = [
        ('StandardScaler', StandardScaler()),
        ('PCA', PCA()),
        ('model', model)
    ]
    
    pipe = Pipeline(steps)
    
    grid_search = GridSearchCV(pipe, param_grid=p_grid, cv=5, scoring='accuracy', n_jobs=4)

    grid_search.fit(X_data, y_data)
    
    return grid_search

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

svc_param_grid = {'PCA__n_components': [10, 20, 30],
              'model__C': [0.1, 0.2, 1, 10, 20],
              'model__gamma': [0.001, 0.005, 0.01, 1],
              'model__kernel': ['linear', 'rbf']}

svc = SVC(random_state = 42)
svc_model = model_pipeline(X_train_resampled, y_train_resampled, svc, svc_param_grid)
print(svc_model.best_score_)

lr_param_grid = {
    'PCA__n_components': [10, 20, 25, 30],
    'model__C': [0.01, 0.1, 0.2, 10, 20],
    'model__max_iter': [500, 700, 1000, 2000]
}

lr = LogisticRegression(random_state=42)
lr_model = model_pipeline(X_train_resampled, y_train_resampled, lr, lr_param_grid)
print(lr_model.best_score_)

xgb_param_grid = {
    'PCA__n_components': [10, 15, 20],
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [None, 3, 4, 5],
    'model__min_child_weight': [1, 3, 5],
    'model__subsample': [0.8, 0.9, 1.0],
    'model__objective': ['binary:logistic'],
    'model__eval_metric': ['logloss']
}

xgb_class = xgb.XGBClassifier(random_state=42)
xgb_model = model_pipeline(X_train_resampled, y_train_resampled, xgb_class, xgb_param_grid)
print("Best Score: ", xgb_model.best_score_, "\nBest Params: ", xgb_model.best_params_, sep="")

# XGBoost performs better than SVC and Logistic Regression, I will proceed with that model.

def fit_model(X_data, y_data, model, params):
    steps = [
        ('StandardScaler', StandardScaler()),
        ('PCA', PCA()),
        ('model', model)
    ]

    pipe = Pipeline(steps)

    pipe.set_params(**params)

    pipe.fit(X_data, y_data)

    return pipe

final_params = {
    'PCA__n_components': 15,
    'model__eval_metric': 'logloss',
    'model__learning_rate': 0.1,
    'model__max_depth': 3,
    'model__min_child_weight': 1,
    'model__n_estimators': 300,
    'model__objective': 'binary:logistic',
    'model__subsample': 0.8,
    'model__random_state': 42
}

clf = fit_model(X_train_resampled, y_train_resampled, xgb_class, final_params)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plotting feature importance against PCs
pc_importance = clf['model'].feature_importances_
plt.bar(range(len(pc_importance)), pc_importance)
plt.xticks(ticks=range(0,15),labels=range(1,15+1))
plt.xlabel('Principal Component')
plt.ylabel('Feature Importance')
plt.title('Feature Importance of Principal Components')
plt.show()

pca = clf['PCA']
original_feature_names = X.columns  # List of original feature names

# Access the loadings of the PCA components
component_loadings = pca.components_

# Identify the top contributing original features for each principal component
top_features_per_component = {}
for i, component_loading in enumerate(component_loadings):
    top_feature_indices = np.argsort(np.abs(component_loading))[::-1][:2]  # Top 5 features with highest absolute loadings
    top_features_per_component[f'PC{i}'] = [original_feature_names[idx] for idx in top_feature_indices]

# Print the top contributing original features for each principal component
for pc, features in top_features_per_component.items():
    print(f'Principal Component {pc}:')
    print(', '.join(features))
    print()

# So within PCA, we can see the most important component is PC1.  
# Within PC1, we can see that concave_points and concavity_mean are the top variables.
# Within other PCs we can also see that texture, smoothness, and fractal dimensions are also top predictors.

X_test_scaled = StandardScaler().fit_transform(X_test)
X_test_pca = PCA(n_components=15).fit_transform(X_test_scaled)

y_pred_proba = clf['model'].predict_proba(X_test_pca)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nfor Breast Cancer Model')
plt.legend(loc="lower right")
plt.show()

# Final metrics are 97% accuracy, f1 94%, precision 93%, 96% recall.
# Model is precise and we are catching most false negatives, which is important, better to have a false positive than a false negative.
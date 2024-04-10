import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

df = pd.read_csv("FILE_PATH/diabetes-2.csv")
df.columns = ["pregnancy", "glucose", "bp", "skin_thick", "insulin", "BMI", "dia_ped_fun", "age", "outcome"]

# # Checking structure and looking for NA values in the dataset
print(df.info())
print(df.describe().T)
print(df.head())

# # Exploring the distribution
sns.pairplot(df, hue = 'outcome',diag_kind="kde", height = 0.8, corner = True)
plt.suptitle("Pairplot of Diabetes Dataset")
plt.show()

def histograms(data, column, hue):
    """
    Function will create histograms for variables of interest: Skin Thickness, Insulin, and Age
    """
    column_mapping = {
        'skin_thick': 'Skin Thickness',
        'insulin': 'Insulin',
        'age': 'Age'
    }
    g = sns.FacetGrid(data, col = hue, hue = hue, height = 5)
    g.map(sns.histplot, data = data,x = column, bins = 25, kde = True)
    g.set_titles("")
    g.figure.suptitle("Distribution of {} by Outcome".format(column_mapping.get(column, column.capitalize())))
    g.set_xlabels(column_mapping.get(column, column.capitalize()))
    g.add_legend()
    g.legend.get_texts()[0].set_text('Non-Diabetic')
    g.legend.get_texts()[1].set_text('Diabetic')
    plt.show()

histograms(df, "skin_thick", 'outcome')
histograms(df, "insulin", 'outcome')
histograms(df, 'age', 'outcome')

sns.heatmap(data = df.corr(), annot = True, cmap="vlag",
             xticklabels=['Pregnancy', 'Glucose', 'Blood\nPressure', 'Skin\nThickness', 'Insulin', 'BMI', 
                           'Diabetes\nPedigree\nFunction', 'Age', 'Outcome'],
                            yticklabels=['Pregnancy', 'Glucose', 'Blood\nPressure', 'Skin\nThickness', 'Insulin', 'BMI', 
                           'Diabetes\nPedigree\nFunction', 'Age', 'Outcome'])
plt.xticks(size = 5, rotation = 45)
plt.yticks(size = 5)
plt.title("Correlations Heatmap for Diabetes Dataset")
plt.show()

df1 = df[df['outcome'] == 0]
df2 = df[df['outcome'] == 1]

def replace_median(x):
    """
    Function will replace the medain values for the 0 values in the Data Frame
    """
    # Replace 0 values with NaN
    x.replace({'glucose': 0, 'bp': 0, 'BMI': 0, 'insulin': 0, 'skin_thick': 0}, np.nan, inplace=True)
    
    # Replace NaN values with median for each column
    x['glucose'].fillna(x['glucose'].median(), inplace=True)
    x['bp'].fillna(x['bp'].median(), inplace=True)
    x['BMI'].fillna(x['BMI'].median(), inplace=True)
    x['insulin'].fillna(x['insulin'].median(), inplace=True)
    x['skin_thick'].fillna(x['skin_thick'].median(), inplace=True)
    
    return x


replace_median(df1)
replace_median(df2)

list_df = [df1, df2]
df_combined = pd.concat(list_df)

# Now that all data is free of 0s, I will move on to balancing the class imbalance of the model.
# Moving on to final model, models to explore are logisitic, SVC, RF, XGBoost

# Will use GridSearchCV to find out best parameters for Logistic, RF, SVC, XGBoost
RFC = RandomForestClassifier(random_state=42)
support = SVC(random_state=42)
xgb_class = xgb.XGBClassifier(random_state=42)
lr = LogisticRegression(random_state=42)
scaler = StandardScaler()

X_final = df_combined.drop(columns=['outcome'])
y_final = df_combined['outcome']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Countplots for before and after SMOTE is applied
sns.countplot(data = pd.DataFrame(y_train), hue = 'outcome', x = 'outcome')
plt.xticks([0,1], ['Non-Diabetic', 'Diabetic'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title("Count Plot for Training Data Outcome Before SMOTE")
plt.show()

# Apply SMOTE only on the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

sns.countplot(data = pd.DataFrame(y_train_resampled), hue = 'outcome', x = 'outcome')
plt.xticks([0,1], ['Non-Diabetic', 'Diabetic'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Count Plot for Training Data Outcome After SMOTE')
plt.show()

# Scale features using the scaler fitted on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

RF_param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced'],
    'probability': [True, False]
}

xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss', 'auc', 'error']
}

lr_param_grid = {
    'penalty': ['l1', 'l2', None],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 300, 400],
    'fit_intercept': [True, False],
    'C': [0.6, 0.8, 1]
}

def grid_search(model, grid, fold):
    """
    Function will take grid parameters and do a grid search to find the best parameters
    """
    search = GridSearchCV(model, grid, cv = fold, scoring='accuracy', verbose = 1)
    search.fit(X_train_scaled, y_train_resampled)
    print("Best {} Parameters:".format(model), search.best_params_)
    print("Best {} Score:".format(model), search.best_score_)

grid_search(RFC, RF_param_grid, 5)
grid_search(lr, lr_param_grid, 5)
grid_search(support, svc_param_grid, 5)
grid_search(xgb_class, xgb_param_grid, 5)

# XGBoost has the best accuracy score at 0.82, so I will go with that model

# Creating XGboost with hyperparameters

xgb_class_tuned = xgb.XGBClassifier(eval_metric='logloss',
                                     learning_rate=0.2,
                                     max_depth=5,
                                     reg_alpha = 17,
                                     reg_lambda=30,
                                     min_child_weight=1,
                                     n_estimators=200,
                                     objective='binary:logistic',
                                     subsample=0.9,
                                     random_state=42)
xgb_model = xgb_class_tuned.fit(X_train_scaled, y_train_resampled)

# Prediction and Evaluation of Model
def predictions(data, model, true):
    """
    Function will take a trained model, predict, and then print out the relevant metrics
    """
    y_pred = model.predict(data)
    print("Accuracy:", accuracy_score(y_true=true, y_pred=y_pred))
    print("F1:", f1_score(y_true=true, y_pred=y_pred, average = 'macro'))
    print("Precision:", precision_score(y_true=true, y_pred=y_pred, average = 'macro'))
    print("Recall:", recall_score(y_true=true, y_pred=y_pred, average = 'macro'))
    print("ROC AUC:", roc_auc_score(y_true=true, y_score=y_pred))

print("Testing Data:")
predictions(X_test_scaled, xgb_model, y_test)
print("\nTraining Data:")
predictions(X_train_scaled, xgb_model, y_train_resampled)


feature_mapping = {
    'f4': 'Insulin',
    'f7': 'Age',
    'f5':'BMI',
    'f6':'Diabetes\nPedigree\nFunction',
    'f2':'Blood\nPressure',
    'f0':'Pregnancy',
    'f3':'Skin\nThickness',
    'f1':'Glucose'
}

color_map = {
    'Insulin':"red",
    'BMI':"red",
    'Glucose':"red",
    'Skin\nThickness':'red',
    'Diabetes\nPedigree\nFunction':'red'
}

ax = xgb.plot_importance(xgb_class_tuned, grid = False, title = "Diabetes Model Feature Importance")
ax.set_yticklabels([feature_mapping[label.get_text()] for label in ax.get_yticklabels()], size=5.5)
for bar, feature_name in zip(ax.patches, [label.get_text() for label in ax.get_yticklabels()]):
    bar.set_color(color_map.get(feature_name, 'gray'))
plt.show()

ax = xgb.plot_importance(xgb_class_tuned, grid = False, title = 'Diabetes Model Feature Gain Importance',
                          importance_type="gain", values_format="{v:.2f}", xlabel="Gain Score")
ax.set_yticklabels([feature_mapping[label.get_text()] for label in ax.get_yticklabels()], size=5.5)
plt.xlim(0,9)
plt.show()

y_pred_proba = xgb_class_tuned.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nfor Diabetes Model')
plt.legend(loc="lower right")
plt.show()

train_sizes, train_scores, test_scores = learning_curve(xgb_class_tuned, X_train_scaled, y_train_resampled, cv=5)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve for Diabetes Model')
plt.legend(loc='lower right')
plt.show()

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
sns.heatmap(cm, cmap = "Reds", 
            xticklabels=['Non-Diabetic', 'Diabetic'], 
            yticklabels=['Non-Diabetic', 'Diabetic'], 
            annot = True)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix of Testing Dataset")
plt.show()
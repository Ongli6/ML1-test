
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#=========================DATA PRP=========================

# Load Data and clean
df = pd.read_excel(r"C:\Users\oyoung\OneDrive - Activate Learning\Documents\Data\Student List Raw data.xlsx")
df.drop(['AcademicYear', 'ApplicationDate', 'OpenAction', 'StudentID', 'Forename', 'Surname', 'ProvInstance', 'Year', 'DoB', 'AgeGroup', 'School'], axis=1, inplace=True)
print (df.head())

# split data into numeric and text     
numeric_features = ['AgeLSC']
# impute missing values
df[numeric_features] = df[numeric_features].replace('', np.nan)
imputer = SimpleImputer(strategy='mean')
numeric_data = imputer.fit_transform(df[numeric_features])

# Fill text columns with empty strings
text_columns = df.select_dtypes(include=['object']).columns
# avoid TF-IDF errors by ensuring no NaNS
df[text_columns] = df[text_columns].fillna('')
numeric_columns = df.select_dtypes(include=['number']).columns

# Keep a copy of the target before dropping
target = df['OpenActionCategory']

# Drop the target column from the dataframe before text combining
df_features = df.drop(columns=['OpenActionCategory'])

# Now fill NA and combine text on df_features
df_features = df_features.fillna('')

# Combine all columns in df_features into one text string
df_features['combined_text'] = df_features.astype(str).agg(' '.join, axis=1)

# --- TEXT VECTORIZATION ---
vector = df_features['combined_text']
print (vector.head())
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = vectorizer.fit_transform(vector)


# --- NUMERIC PREPROCESSING ---
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_data)  # use the imputed numpy array, not original df 

# --- COMBINE TEXT + NUMERIC ---
numeric_sparse = csr_matrix(scaled_numeric)  # Convert to sparse format
combined_matrix = hstack([tfidf_matrix, numeric_sparse])  # Final matrix (n_samples x total_features)

feature_names = vectorizer.get_feature_names_out()


#=========================MODEL DEFINE=========================

#define x and Y
X = combined_matrix
y = (target == '810 - Withdrawn').astype(int)

#Define model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    verbose=1,
    random_state=42,
    n_jobs=-1
)


#=========================OUTPUT VALIDATION=========================

# STRATIFIED K-FOLD CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get predicted probabilities from cross-validation
y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
threshold = 0.51696
y_pred = (y_proba >= threshold).astype(int)

# Define scoring metrics
scoring = {
    'f1': make_scorer(f1_score),
    'roc_auc': 'roc_auc'
}

#sanity check should expect low auc
y_shuffled = np.random.permutation(y)
shuffled_scores = cross_val_score(model, X, y_shuffled, cv=5, scoring='roc_auc')
print(f"Shuffled ROC AUC: {np.mean(shuffled_scores):.4f}")

#RUn the model
model.fit(X,y)

importances = model.feature_importances_
# Combine feature names
feature_names = list(vectorizer.get_feature_names_out()) + list(numeric_features)

# Get feature importances
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by absolute importance
top_features = importance_df.sort_values(by='Importance', ascending=False).head(20)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
plt.title('Top 20 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

#prints results
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

print(f"ROC-AUC scores per fold: {scores}")
print(f"Mean ROC-AUC: {np.mean(scores):.4f}")
print(f"Log Loss:       {log_loss(y, y_proba):.4f}")
print(f"ROC AUC:        {roc_auc_score(y, y_proba):.4f}")
print(f"Precision:      {precision_score(y, y_pred):.4f}")
print(f"Recall:         {recall_score(y, y_pred):.4f}")
print(f"F1 Score:       {f1_score(y, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

precisions, recalls, thresholds = precision_recall_curve(y, y_proba)



#plot to visually find the best trade-off ratio
#plt.plot(thresholds, precisions[:-1], label='Precision')
#plt.plot(thresholds, recalls[:-1], label='Recall')
#plt.xlabel('Threshold')
#plt.legend()
#plt.grid()
#plt.show()


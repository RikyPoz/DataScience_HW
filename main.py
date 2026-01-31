import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import re
import html

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

from nltk.stem import SnowballStemmer

SEED = 42   
TRAIN_FILE = 'winter_project_2026/development.csv'   
EVAL_FILE = 'winter_project_2026/evaluation.csv'     
OUTPUT_FILE = 'winter_project_2026/sample_submission.csv'

def preprocess_text(df, remove_duplicates=False):
    if remove_duplicates:
        initial_len = len(df)
        
        if 'label' in df.columns:
            df['count_per_label'] = df.groupby(['title', 'article', 'label'])['title'].transform('count')
            df['max_count_for_text'] = df.groupby(['title', 'article'])['count_per_label'].transform('max')
            df_majority = df[df['count_per_label'] == df['max_count_for_text']].copy()
            
            # 4. Step Tie-Breaker: Gestione dei Pareggi (es. 1 Sport vs 1 Business)
            inconsistent_mask = df_majority.duplicated(subset=['title', 'article'], keep=False)
            ties = df_majority[inconsistent_mask].groupby(['title', 'article'])['label'].nunique()
            ties_indices = ties[ties > 1].index
            
            df_clean = df_majority.set_index(['title', 'article']).drop(index=ties_indices, errors='ignore').reset_index()
            df = df_clean.drop_duplicates(subset=['title', 'article'], keep='first').copy()
            
            drop_cols = ['count_per_label', 'max_count_for_text']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
            
        else:
            df = df.drop_duplicates(subset=['title', 'article'], keep='first').copy()  
    else:
        df = df.copy()
        
    # Feature weighting    
    source_feature = (df['source'].fillna('') + " ") * 3
    title_weighted  = (df['title'].fillna('') + " ") * 2
    df['text_combined'] = source_feature + title_weighted + " " + df['article'].fillna('')
    
    # Stemming
    stemmer = SnowballStemmer("english")
    
    def clean(text):
        text = str(text).lower()
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z0-9\s$€%£]', ' ', text) # useful symbols for class=business (€,%,$)                  
        text = re.sub(r'\s+', ' ', text).strip()
        return " ".join([stemmer.stem(word) for word in text.split()]) # stemming

    df['clean_text'] = df['text_combined'].apply(clean)
    return df


def extract_time_features(df):
    df = df.copy()
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['has_date'] = df['timestamp_dt'].notna().astype(int)
    df['hour'] = df['timestamp_dt'].dt.hour.fillna(-1)
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(-1)
    df = df.drop(columns=['timestamp_dt'])
    
    return df

def main():    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(42)
    
    dev_df = pd.read_csv(TRAIN_FILE)
    eval_df = pd.read_csv(EVAL_FILE)
    
    dev_df = extract_time_features(dev_df, remove_duplicates=True)
    eval_df = extract_time_features(eval_df, remove_duplicates=False)
    
    feature_cols = ['clean_text', 'page_rank', 'hour', 'has_date']
    
    X = dev_df[feature_cols]
    y = dev_df['label']
    
    text_features = 'clean_text'
    num_features = ['page_rank', 'hour', 'has_date']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('txt', TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1,2)), text_features),
            ('num', MinMaxScaler(), num_features)
        ],
        remainder='drop' 
    )
    
    # Pipeline Completa
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LinearSVC(class_weight='balanced', random_state=SEED))
    ])
    
    # 4. Training
    model_pipeline.fit(X_train, y_train)
    
    # 5. Valutazione
    print("Valutazione...")
    y_pred = model_pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))
    print(f"F1 Macro: {f1_score(y_val, y_pred, average='macro'):.4f}")

    pd.DataFrame({
        'Id': eval_ids,
        'Predicted': y_pred_eval
    }).to_csv(OUTPUT_FILE, index=False)
    
if __name__ == "__main__":
    main()
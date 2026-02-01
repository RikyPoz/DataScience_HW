import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import re
import html

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

from nltk.stem import SnowballStemmer

SEED = 42   
TRAIN_FILE = 'development.csv'   
EVAL_FILE = 'evaluation.csv'     
OUTPUT_FILE = 'sample_submission.csv'

def preprocess_text(df, remove_duplicates=False):
    if remove_duplicates:
        initial_len = len(df)
        
        if 'label' in df.columns:
            df['count_per_label'] = df.groupby(['title', 'article', 'label'])['title'].transform('count')
            df['max_count_for_text'] = df.groupby(['title', 'article'])['count_per_label'].transform('max')

            df_majority = df[df['count_per_label'] == df['max_count_for_text']].copy()
            
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
        
    # Feature weighting: boost importance of Source and Title     
    source_feature = (df['source'].fillna('') + " ") * 3
    title_weighted  = (df['title'].fillna('') + " ") * 3
    df['text_combined'] = source_feature + title_weighted + " " + df['article'].fillna('')
    
    stemmer = SnowballStemmer("english")
    
    def clean(text):
        text = str(text).lower()
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text) 
        text = re.sub(r'[^a-z0-9\s$€%£]', ' ', text)       
        text = re.sub(r'\s+', ' ', text).strip()
        return " ".join([stemmer.stem(word) for word in text.split()])

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
    
    eval_ids = eval_df['Id']
    
    dev_df['article'] = dev_df['article'].replace('\\N', '')
    eval_df['article'] = eval_df['article'].replace('\\N', '')

    dev_df = extract_time_features(dev_df)
    eval_df = extract_time_features(eval_df) 

    dev_df = preprocess_text(dev_df, remove_duplicates=True)
    eval_df = preprocess_text(eval_df, remove_duplicates=False) 

    dev_df = dev_df[dev_df['clean_text'].str.len() > 20].copy()
    
    feature_cols = ['clean_text', 'source', 'page_rank', 'hour', 'day_of_week', 'has_date']
    
    X = dev_df[feature_cols] 
    y = dev_df['label']
    
    final_text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english', 
            min_df=3, 
            sublinear_tf=True,
            ngram_range=(1,2)
        )),
        ('selector', SelectKBest(chi2, k=40000))
    ])

    final_pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('tfidf', final_text_pipeline, 'clean_text'),
                ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=5), ['source','day_of_week']), 
                ('num', MinMaxScaler(), ['page_rank', 'hour', 'has_date'])
            ]
        )),
        ('clf', LinearSVC(
            class_weight='balanced', 
            random_state=42, 
            dual='auto',
            C=0.2
        ))
    ])
    
    final_pipeline.fit(X, y)
    y_pred_eval = final_pipeline.predict(eval_df)

    pd.DataFrame({
        'Id': eval_ids,
        'Predicted': y_pred_eval
    }).to_csv(OUTPUT_FILE, index=False)
    
if __name__ == "__main__":
    main()
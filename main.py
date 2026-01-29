import pandas as pd
import numpy as np
import html
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

TRAIN_FILE = 'winter_project_2026/development.csv'
EVAL_FILE = 'winter_project_2026/evaluation.csv'
SUBMISSION_FILE = 'winter_project_2026/sample_submission.csv'

BEST_PARAMS = {
    'C': 0.2,
    'ngram_range': (1, 2),
    'min_df': 3,
    'max_features': None 
}

def preprocess_data(df, is_train=True):
    """
    Fast preprocessing function tailored for time constraints.
    Avoids heavy loops like lemmatization/stemming.
    """
    source_feat = (df['source'].fillna('') + " ") * 3
    
    # Fast vectorized string concatenation
    df['text_combined'] = (
        source_feat + 
        df['title'].fillna('') + " " + 
        df['article'].fillna('')
    )
    
    return df

def main():

    dev_df = pd.read_csv(TRAIN_FILE)
    eval_df = pd.read_csv(EVAL_FILE)

    dev_df = preprocess_data(dev_df, is_train=True)
    eval_df = preprocess_data(eval_df, is_train=False)

    X_train = dev_df[['text_combined', 'page_rank']]
    y_train = dev_df['label']
    X_eval = eval_df[['text_combined', 'page_rank']]

    
    # Text Pipeline
    text_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            sublinear_tf=True,
            strip_accents='unicode', # Fast internal cleaning
            ngram_range=BEST_PARAMS['ngram_range'],
            min_df=BEST_PARAMS['min_df'],
            max_features=BEST_PARAMS['max_features']
        ))
    ])

    # Full Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipe, 'text_combined'),
            ('num', StandardScaler(), ['page_rank'])
        ]
    )

    # Final Estimator
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LinearSVC(
            C=BEST_PARAMS['C'],
            class_weight='balanced', 
            dual='auto',
            random_state=42
        ))
    ])

    # 4. Train
    model.fit(X_train, y_train)

    # 5. Predict
    y_pred = model.predict(X_eval)

    # 6. Save Submission
    submission = pd.DataFrame({
        'Id': eval_df['Id'],
        'Predicted': y_pred
    }).to_csv(SUBMISSION_FILE, index=False)



if __name__ == "__main__":
    main()
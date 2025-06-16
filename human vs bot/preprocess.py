import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Selecting required features
    df = df[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Bot Label']]

    # Convert 'Verified' boolean to integer
    df['Verified'] = df['Verified'].astype(int)

    # Splitting data
    X = df.drop(columns=['Bot Label'])
    y = df['Bot Label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

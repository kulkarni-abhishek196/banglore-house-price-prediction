import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    #1 made standardized for size column converted '2 bedroom' to '2 BHK'
    df['size_filled'] = df['size'].str.replace(r'(\d+)\s*Bedroom', r'\1 BHK', case=False, regex=True)

    #2. dropping society as it has more than 5k rows missing
    df = df.drop(columns=['society'])

    #3. dropped rows with missing values for size and size_filled
    df = df.dropna(subset=['size', 'size_filled'])

    #4. filling 'bath' NaN values using median values w.r.to size_filled
    grouped_bath_medians = df.groupby('size_filled')['bath'].transform('median')
    df['bath'] = df['bath'].fillna(grouped_bath_medians)

    #5.filling 'balcony' NaN values using median values w.r.to size_filled
    grouped_balcony_median = df.groupby('size_filled')['balcony'].transform('median')
    df['balcony'] = df['balcony'].fillna(grouped_balcony_median)

    #6. removed sizes beyond 5BHK
    allowed_bhk = ["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"]
    df = df[df['size_filled'].isin(allowed_bhk)]

    #7. replaced random dates from availabilty column with 'Not ready to move' so it will have just 2 categories
    df.loc[df['availability'] != 'Ready To Move', 'availability'] = 'Not Ready To Move'

    #8.Total _sqft had ranges as well plus Mt. and yards units as well.. so removed that total 234
    non_numeric_idx = df[~df['total_sqft'].str.replace('.','',regex=False).str.isnumeric()].index
    df = df.drop(index=non_numeric_idx)

    #9. Changed total_sqft to type float
    df['total_sqft'] = df['total_sqft'].astype(float)

    #10. Some wrong data entries removed with validation (checked minimum total_sqft required is > 300) so removed entries below than this
    df = df[df['total_sqft'] >= 300]

    #11. new price per sqft feature
    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

    #12. removed area_type
    df = df.drop(columns=['area_type'])

    #13. dropped missing locations
    missing_location = df[df['location'].isna()]
    df = df.drop(index=missing_location.index)

    #14. removed remaining duplicates
    df = df.drop_duplicates()

    #15. encoded rare locations to 'others' category
    counts = df['location'].value_counts()
    rare_locations = counts[counts < 10].index
    df['location'] = df['location'].replace(rare_locations, 'other_rare_location')

    #16. logging transformation on total_sqft
    df['total_sqft_log'] = np.log1p(df['total_sqft'])

    #17. engineered one more feature sqft per bhk
    all_bhk = df['size_filled'].values
    bhkNumbers = df['size_filled'].str.replace('BHK', '', regex=False).str.strip().astype(int)
    df['sqft_perBHK'] = df['total_sqft'] / bhkNumbers

    #Splitting features and target column
    X = df.drop(columns=['price', 'price_per_sqft', 'total_sqft', 'size'])
    y = df['price']

    #columns
    numerical_cols = ["total_sqft_log", "bath", "balcony", "sqft_perBHK"]
    category_cols = ["size_filled", "availability"]
    location_col = ['location']

    preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numerical_cols),
        ('categorical', OneHotEncoder(), category_cols),
        ('location', TargetEncoder(categories='auto', cv=5, smooth='auto', random_state=42), location_col)
    ],
    remainder='drop'
    )

    return X, y, preprocessor
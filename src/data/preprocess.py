from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(numeric_features=None, categorical_features=None):
    nf = numeric_features or NUMERIC_FEATURES
    cf = categorical_features or CATEGORICAL_FEATURES
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    ct = ColumnTransformer(transformers=[
        ("num", num, nf),
        ("cat", cat, cf)
    ])
    return ct

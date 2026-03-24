# preprocessing.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess(df):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    binary_columns = ["PaperlessBilling","PhoneService","Dependents","Partner"]
    for i in binary_columns:
        if i in df.columns:
            df[i] = df[i].map({"Yes":1,"No":0})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female":0,"Male":1})

    cat_col = list(df.select_dtypes("object").columns)
    num_col = ["MonthlyCharges","TotalCharges","tenure"]

    preprocessing = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_col),
        ("num", StandardScaler(), num_col)
    ], remainder="passthrough")

    encoded = preprocessing.fit_transform(df)
    new_cols = preprocessing.get_feature_names_out()
    final_df = pd.DataFrame(encoded, columns=new_cols)
    final_df.columns = [c.split("_")[-1] for c in final_df.columns]

    if "id" in final_df.columns:
        final_df.drop(columns=["id"], inplace=True)

    return final_df
import pandas as pd
import re

def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    text_columns = df.select_dtypes(include=["str"]).columns
 
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
 
    return df

MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}
def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
 
    return df

def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_columns = [
    "condition",
    "fuel_type",
    "transmission",
    "drive_unit",
    "segment"
]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )
    return df

def _remove_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.dropna(subset=["priceUSD"])
 
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
 
    df_clean = (
        df
        .pipe(_strip_string_values)
        .pipe(_replace_missing_like_values)
        .pipe(_clean_categorical_values)
        .pipe(_remove_rows_with_missing_target)
        .reset_index(drop=True)
    )
 
    return df_clean

def main() -> None:
    """Load raw data, clean it, and save the cleaned dataset."""
    print("Loading raw dataset...")
 
    df_raw = pd.read_csv("cars.csv")
 
    print("Cleaning dataset...")
 
    df_cleaned = clean(df_raw)
 
    print("Saving cleaned dataset...")
 
    df_cleaned.to_csv("cars_clean.csv", index=False)
 
    print("Cleaned dataset saved to.")

if __name__ == "__main__":
    main()
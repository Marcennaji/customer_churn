import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder
from .encoder_base import EncoderBase


class OneHotEncoderWrapper(EncoderBase):
    """Encodes categorical variables using One-Hot Encoding."""

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            encoder = OneHotEncoder(sparse=False, drop="first")
            encoded_data = encoder.fit_transform(df[[column]])
            encoded_df = pd.DataFrame(
                encoded_data, columns=encoder.get_feature_names_out([column])
            )

            df = df.drop(columns=[column]).reset_index(drop=True)
            df = pd.concat([df, encoded_df], axis=1)
            logging.info(f"Applied One-Hot Encoding to '{column}'.")
        return df

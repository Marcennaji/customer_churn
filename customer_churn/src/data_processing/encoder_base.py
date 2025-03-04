from abc import ABC, abstractmethod
import pandas as pd


class EncoderBase(ABC):
    """Abstract Base Class for categorical encoders."""

    @abstractmethod
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Encodes a specific column in the dataset."""
        pass

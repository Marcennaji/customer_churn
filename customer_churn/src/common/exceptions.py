# Custom exception for missing columns

class ColumnNotFoundError(Exception):
    """Raised when a specified column is not found in the DataFrame."""
    pass
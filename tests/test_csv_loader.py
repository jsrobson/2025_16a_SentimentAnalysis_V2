import pandas as pd
import pytest

from utils import CSVLoader

def test_load_valid_csv(tmp_path):
    # Arrange
    csv_content = "text,sentiment\nhello,positive\nbye,negative"
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(csv_content)

    loader = CSVLoader(str(csv_file))

    # Act
    df = loader.load()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["text", "sentiment"]
    assert df.iloc[0]["text"] == "hello"

def test_load_missing_file_raises_fnf(tmp_path):
    missing_file = tmp_path / "missing.csv"
    loader = CSVLoader(str(missing_file))

    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        loader.load()

def test_load_invalid_extension_raises_value_error(tmp_path):
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("just some text")

    loader = CSVLoader(str(txt_file))

    with pytest.raises(ValueError, match="Invalid file type"):
        loader.load()
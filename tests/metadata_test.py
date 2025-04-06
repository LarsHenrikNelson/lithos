import pytest
import os
import glob
from lithos.utils import metadata_utils


@pytest.fixture(scope="session")
def clean_directory(directory):
    for file in glob.glob(directory, recursive=True):
        os.remove(file)


def test_home_dir():
    home_dir = metadata_utils.home_dir()
    assert home_dir.exists()
    assert home_dir.is_dir()


def test_metadata_dir():
    metadata_dir = metadata_utils.metadata_dir()
    assert metadata_dir.exists()
    assert metadata_dir.is_dir()

    # Clean up directory
    metadata_dir.rmdir()


def test_save_metadata():
    metadata = {"key": "value"}  # Example metadata dictionary

    # Save metadata to a file
    metadata_utils.save_metadata(metadata, "test_metadata")  # Save to default location

    # Check if the file exists in the default metadata directory
    assert metadata_utils.metadata_dir().joinpath("test_metadata.txt").exists()

    metadata = metadata_utils.load_metadata(
        "test_metadata"
    )  # Load metadata from the file
    assert "key" in metadata  # Check if the key exists in the loaded metadata
    assert metadata["key"] == "value"  # Check if the value is correct

    # Clean up the test file after the test
    metadata_utils.metadata_dir().joinpath("test_metadata.txt").unlink()


def test_set_metadata_dir():
    # Set a custom metadata directory for testing
    custom_dir = metadata_utils.home_dir().joinpath("custom_metadata_dir")
    metadata_utils.set_metadata_dir(custom_dir)

    # Check if the custom directory is set correctly
    assert metadata_utils.metadata_dir() == custom_dir

    # Clean up the custom directory after the test
    custom_dir.rmdir()  # Remove the directory if it's empty

    # Reset the metadata directory to the default location
    metadata_utils.set_metadata_dir(metadata_utils.home_dir().joinpath("metadata"))

    # Check if the metadata directory is reset to the default location
    assert metadata_utils.metadata_dir() == metadata_utils.home_dir().joinpath(
        "metadata"
    )

    # Clean up the default metadata directory after the test (if it's empty)
    metadata_utils.metadata_dir().rmdir()  # Remove the directory if it's empty
    # Clean up the test file after the test
    metadata_utils.home_dir().joinpath("metadata_dir.txt").unlink()


def test_metadata_to_string():
    # Set a custom metadata directory for testing
    metadata = {"key": "value", "key2": ["item1", "item2"]}
    output = metadata_utils.metadata_to_string(metadata)
    test = [
        "{\n",
        "'key': \"value\",\n",
        "'key2':\n",
        "  [\n",
        "  'item1',",
        "'item2',\n",
        "  ],\n",
        "}\n",
    ]
    assert output == test

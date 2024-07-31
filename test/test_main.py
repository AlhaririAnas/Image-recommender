import pytest
import torch
import pickle
from unittest.mock import patch, MagicMock, mock_open
from collections import defaultdict
import sys
from resources.main import run, parser


# Mock ImageGenerator, get_metadata, save_metadata_in_database, get_last_entry, get_similarities
@patch("resources.generator.ImageGenerator")
@patch("resources.metadata_reader.get_metadata")
@patch("resources.metadata_reader.save_metadata_in_database")
@patch("resources.metadata_reader.get_last_entry")
@patch("resources.similarity.get_similarities")
@patch("resources.metadata_reader.create_database")
def test_resume_after_crash(
    mock_create_database,
    mock_get_similarities,
    mock_get_last_entry,
    mock_save_metadata_in_database,
    mock_get_metadata,
    MockImageGenerator,
):
    # Set up mocks
    mock_image_gen = MagicMock()
    images = [MagicMock()] * 10
    mock_image_gen.image_generator.return_value = images
    MockImageGenerator.return_value = mock_image_gen

    mock_get_last_entry.return_value = 0
    mock_get_similarities.return_value = {"similarity": 1}

    args = parser.parse_args(
        ["-m", "-s", "-p", "fake_path", "-d", "cpu", "--pkl_file", "fake_similarities.pkl", "--checkpoint", "2"]
    )

    similarities = defaultdict()

    # Simulate initial run with crash after processing 2 images
    with patch("builtins.open", mock_open()) as mock_open_:
        with patch("pickle.load", return_value=similarities):
            with patch("pickle.dump") as mock_pickle_dump:

                def side_effect(*args, **kwargs):
                    if mock_pickle_dump.call_count == 1:
                        # Update similarities to simulate progress before crash
                        similarities[1] = {"similarity": 1}
                        similarities[2] = {"similarity": 1}
                        pickle.dump(similarities, *args, **kwargs)
                        raise RuntimeError("Simulated crash")
                    else:
                        pickle.dump(similarities, *args, **kwargs)

                mock_pickle_dump.side_effect = side_effect

                with pytest.raises(RuntimeError, match="Simulated crash"):
                    run(args)

    assert len(similarities) == 2  # Checkpoint after processing 2 images

    # Simulate restart and continuation
    with patch("builtins.open", mock_open()) as mock_open_:
        with patch("pickle.load", return_value=similarities):
            with patch("pickle.dump") as mock_pickle_dump:
                run(args)

                # Assertions
                mock_create_database.assert_called_once()
                assert mock_get_last_entry.call_count == 1
                assert mock_get_metadata.call_count == 10
                assert mock_save_metadata_in_database.call_count == 10
                assert mock_get_similarities.call_count == 10
                assert mock_pickle_dump.call_count >= 6  # Continuing from where it left off


@pytest.fixture(autouse=True)
def mock_sys_argv():
    original_argv = sys.argv
    sys.argv = ["prog_name"]
    yield
    sys.argv = original_argv

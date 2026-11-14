"""Native waveform compatibility; no detector scores or labels are involved."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from ontoresponse.data.audio import read_audio, read_audio_details


def flac_fixture(tmp_path, samplerate=16000):
    path = tmp_path / "recording.flac"
    x = np.sin(np.arange(800) * 0.07).astype("float32") * 0.4
    sf.write(path, np.stack([x, -x * 0.3], axis=1), samplerate, subtype="PCM_16")
    return path


def test_working_soundfile_reader_keeps_samples_and_does_not_run_ffmpeg(tmp_path):
    path = flac_fixture(tmp_path)
    expected, _ = sf.read(path, dtype="float32", always_2d=True)
    with patch("ontoresponse.data.audio.subprocess.run", side_effect=AssertionError):
        actual, meta = read_audio_details(path)
    np.testing.assert_array_equal(actual, expected.mean(1))
    assert meta == dict(reader="soundfile", samplerate=16000, channels=2, frames=800)
    np.testing.assert_array_equal(read_audio(path), actual)


def test_strict_flac_fallback_matches_native_pcm_without_resampling(tmp_path):
    path = flac_fixture(tmp_path)
    expected, _ = sf.read(path, dtype="float32", always_2d=True)
    with patch("ontoresponse.data.audio.sf.read", side_effect=sf.LibsndfileError(161)):
        actual, meta = read_audio_details(path)
    np.testing.assert_array_equal(actual, expected.mean(1))
    assert meta["reader"] == "ffmpeg_strict" and meta["frames"] == 800
    assert meta["channels"] == 2 and meta["samplerate"] == 16000
    assert "soundfile_error" in meta


def test_flac_fallback_does_not_resample_unexpected_rate(tmp_path):
    path = flac_fixture(tmp_path, 48000)
    with (
        patch("ontoresponse.data.audio.sf.read", side_effect=sf.LibsndfileError(161)),
        patch("ontoresponse.data.audio.subprocess.run", side_effect=AssertionError),
        pytest.raises(ValueError, match="Expected 16000Hz"),
    ):
        read_audio(path)


@pytest.mark.parametrize("failed_decoder", [False, True])
def test_flac_fallback_rejects_short_output_or_decoder_error(tmp_path, failed_decoder):
    path = flac_fixture(tmp_path)
    output = SimpleNamespace(
        returncode=1 if failed_decoder else 0,
        stdout=bytes(800 * 2 * 4 if failed_decoder else 100),
        stderr=b"corrupt frame" if failed_decoder else b"",
    )
    with (
        patch("ontoresponse.data.audio.sf.read", side_effect=sf.LibsndfileError(161)),
        patch("ontoresponse.data.audio.subprocess.run", return_value=output),
        pytest.raises(ValueError, match="Strict FLAC|frame count"),
    ):
        read_audio(path)

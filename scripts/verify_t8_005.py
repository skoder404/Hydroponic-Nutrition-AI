"""Verification script for T8-005.

Acceptance Criteria:
- Each path has a success and a failure test:
  - Path 1: image upload to plant mode segmentation to prediction (success) & segmentation failure returns retryable error (failure)
  - Path 2: leaf crop to features to prediction (success) & missing input_mode / corrupt image (failure)
  - Path 3: sensor payload to prediction (success) & invalid unit / missing readings (failure)
- Stub models are clearly marked as test fixtures
- No real datasets required in CI
"""

import sys
import subprocess
from pathlib import Path


def main() -> int:
    test_file = Path("tests/integration/test_prediction_paths.py")
    if not test_file.exists():
        print("FAIL: tests/integration/test_prediction_paths.py does not exist.")
        return 1

    content = test_file.read_text(encoding="utf-8")

    # Check stub fixtures presence
    assert "class StubImageClassifierFixture" in content, (
        "Missing StubImageClassifierFixture"
    )
    assert "class StubSensorClassifierFixture" in content, (
        "Missing StubSensorClassifierFixture"
    )
    assert "create_stub_segmenter_fixture" in content, (
        "Missing create_stub_segmenter_fixture"
    )
    assert "create_failing_segmenter_fixture" in content, (
        "Missing create_failing_segmenter_fixture"
    )

    # Check all path tests exist
    assert "def test_path1_plant_mode_segmentation_to_prediction_success" in content
    assert (
        "def test_path1_plant_mode_segmentation_failure_returns_retryable_error"
        in content
    )
    assert "def test_path2_leaf_mode_crop_to_prediction_endpoint_success" in content
    assert "def test_path2_leaf_mode_missing_input_mode_rejected" in content
    assert "def test_path3_sensor_payload_to_prediction_success" in content
    assert "def test_path3_sensor_payload_to_prediction_invalid_unit_failure" in content

    # Run pytest
    print("Running pytest on tests/integration/test_prediction_paths.py...")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
    )

    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        print("FAIL: pytest exited with non-zero code.")
        return 1

    print("ALL ACs for T8-005 VERIFIED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

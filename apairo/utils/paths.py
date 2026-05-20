from pathlib import Path

# Resolve paths relative to this file so they work when the package is installed
_PACKAGE_ROOT = Path(__file__).parent.parent
dataset_profile_directory_path = str(_PACKAGE_ROOT / "parameter" / "dataset_profiles")
data_directory_path = "data"

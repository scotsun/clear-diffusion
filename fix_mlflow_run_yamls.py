import os

# CONFIGURATION
# 1. The root of your local mlruns folder (e.g., current directory)
MLRUNS_ROOT = os.path.abspath("./mlruns")

# 2. The old prefix to look for (from your error message)
OLD_PREFIX = "/hpc/home/ms1008/clear-diffusion/mlruns"
# Note: Sometimes it might just be "/hpc/..." without "file://".
# Check one of your meta.yaml files if this script doesn't catch it.


def fix_meta_files(root_dir):
    print(f"Scanning {root_dir}...")

    # Iterate through all files in mlruns
    for dirpath, _, filenames in os.walk(root_dir):
        if "meta.yaml" in filenames:
            file_path = os.path.join(dirpath, "meta.yaml")

            with open(file_path, "r") as f:
                content = f.read()

            # Check if the file contains the old prefix
            if OLD_PREFIX in content:
                # Create the new valid local path
                # We need to construct the correct path dynamically
                # New format: file:///abs/path/to/mlruns/EXP_ID/RUN_ID/artifacts

                # Determine relative path from mlruns root to this folder
                rel_path = os.path.relpath(dirpath, root_dir)

                # Construct new absolute prefix
                new_prefix = f"file://{MLRUNS_ROOT}"

                # Perform replacement
                new_content = content.replace(OLD_PREFIX, new_prefix)

                with open(file_path, "w") as f:
                    f.write(new_content)

                print(f"Fixed: {rel_path}/meta.yaml")


fix_meta_files(MLRUNS_ROOT)
print("Done.")

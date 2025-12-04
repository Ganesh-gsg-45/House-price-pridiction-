"""Create a deployable artifacts package.

Usage:
    python scripts/package_artifacts.py [--zip]

This script copies `artifacts/model.pkl` and `artifacts/preprocess.pkl` into
`deploy_artifacts/` under the project root. If `--zip` is passed, it will also
create `deploy_artifacts.zip`.
"""

import os
import shutil
import argparse


def package_artifacts(project_root=None, make_zip=False):
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts')
    deploy_dir = os.path.join(project_root, 'deploy_artifacts')

    files_to_copy = ['model.pkl', 'preprocess.pkl']

    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    os.makedirs(deploy_dir, exist_ok=True)

    copied = []
    for f in files_to_copy:
        src = os.path.join(artifacts_dir, f)
        dst = os.path.join(deploy_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(dst)
        else:
            print(f"Warning: {src} not found and will not be included in package.")

    if make_zip:
        zip_path = os.path.join(project_root, 'deploy_artifacts.zip')
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', deploy_dir)
        print(f"Created zip package: {zip_path}")

    return copied


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', dest='zip', action='store_true', help='Create a zip archive of the deploy_artifacts folder')
    args = parser.parse_args()

    try:
        copied_files = package_artifacts(make_zip=args.zip)
        print('Copied files:')
        for p in copied_files:
            print(' -', p)
    except Exception as e:
        print('Error packaging artifacts:', e)
        raise

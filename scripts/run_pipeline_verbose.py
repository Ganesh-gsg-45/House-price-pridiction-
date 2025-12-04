import runpy
import traceback
import sys

ERROR_FILE = 'scripts/pipeline_error.txt'

def main():
    try:
        # Ensure project root is on sys.path so `src` package imports resolve
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        runpy.run_path('pipeline.py', run_name='__main__')
    except Exception:
        with open(ERROR_FILE, 'w', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        # re-raise so the process still exits with non-zero code
        raise

if __name__ == '__main__':
    main()

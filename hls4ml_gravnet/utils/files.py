from pathlib import Path


def get_project_root_dir(project_name: str) -> Path:
    parts = list(Path.cwd().parts)
    for p in reversed(parts):
        if p == project_name:
            break
        else:
            parts.remove(p)
    return Path(*parts)


PROJECT_ROOT = get_project_root_dir('hls4ml-gravnet')
DATA_PATH = PROJECT_ROOT / 'data'
DATASET_PATH = DATA_PATH / 'dataset'
KERAS_MODEL_PATH = DATA_PATH / 'models'
RESULTS_PATH = DATA_PATH / 'results'
HLS4ML_OUT_PATH = DATA_PATH / 'hls4ml_out'

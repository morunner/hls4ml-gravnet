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

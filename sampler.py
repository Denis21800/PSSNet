import os
import shutil
from pathlib import Path


def split(root_dir, out_dir, sample_size=50000):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    dir_idx = 0
    total_count = 0
    for dir_, _, filenames in os.walk(root_dir):
        for file in filenames:
            source_path = os.path.join(dir_, file)
            target_dir = out_dir / f'sample_{dir_idx}'
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / file
            shutil.copy(source_path, target_path)
            total_count += 1
            if total_count % sample_size == 0:
                print(f'Processed {total_count}')
                dir_idx += 1

    print(f"Complete. Total: {total_count}")


ROOT_DIR = "D://Data//B-SPIKE/b-spike_out"
OUT_DIR = "D://Data//Proteins_structs//B=_sampled"
if __name__ == '__main__':
    split(ROOT_DIR, OUT_DIR)

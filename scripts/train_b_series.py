"""B-series Python entry point with original DE-MambaNet-style console logs.

Only adapts console/file presentation; the frozen training entry point is reused.
"""
from pathlib import Path
import hashlib
import json
import logging
import sys
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def validation_text(row):
    metrics = row['validation']
    if metrics is None:
        return 'val skipped'
    return (
        f"val mIoU {metrics['mIoU']:.4f} | F1 {metrics['F1']:.4f} | "
        f"clDice {metrics['clDice']:.4f} | BoundaryF1 {metrics['BoundaryF1']:.4f} | "
        f"ComponentError {metrics['ComponentError']:.3f} | structural {row['structural_score']:.4f}"
    )


def main():
    import b_series_study as study
    from b_series_runtime import engine

    config = study.loadcfg()
    original_append = engine.append_row
    states = {}

    def append_with_console(path, row):
        original_append(path, row)
        path = Path(path)
        if path.name != 'train.jsonl':
            return
        if path not in states:
            logger = logging.getLogger('b_series.console.' + str(path))
            logger.setLevel(logging.INFO)
            logger.propagate = False
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path.parent / 'train.log', encoding='utf-8')):
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            states[path] = [logger, time.perf_counter() - row['seconds'], None]
            (path.parent / 'console_logging.json').write_text(json.dumps({
                'entry_point': 'scripts/train_b_series.py',
                'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'console_every_epoch': True,
                'frozen_training_files_modified': False,
            }, indent=2), encoding='utf-8')
        logger, start, smoothed = states[path]
        smoothed = row['seconds'] if smoothed is None else .8 * smoothed + .2 * row['seconds']
        states[path][2] = smoothed
        logger.info('Epoch %03d/%03d | loss %.4f | %s | epoch %s | elapsed %s | ETA %s',
                    row['epoch'], config['training']['max_epochs'], row['train_loss'], validation_text(row),
                    duration(row['seconds']), duration(time.perf_counter() - start),
                    duration(smoothed * (config['training']['max_epochs'] - row['epoch'])))
        logger.info('Auto Protocol | global update %d/%d | LR %.8f | AMP skipped total %d',
                    row['global_update'], config['training']['target_updates'], row['lr'], row['amp_skipped_updates_total'])

    engine.append_row = append_with_console
    # The engine's only console print is its periodic duplicate JSON epoch row.
    engine.print = lambda *args, **kwargs: None
    study.main()


if __name__ == '__main__':
    main()

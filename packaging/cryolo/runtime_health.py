"""Read-only health probe for the private stock crYOLO runtime (Python 3.7)."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

BACKEND_SHA256 = '0eb044b4018be4b8c530389843d9650569751439fdab2880f3020e65ff2ff358'


def _stage(name, progress_report):
    record = {'stage': name, 'updated_at': time.time()}
    print('SARU runtime check: ' + name, file=sys.stderr, flush=True)
    if progress_report:
        path = Path(progress_report)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(record), encoding='utf-8')
        os.replace(str(temporary), str(path))


def check_runtime(progress_report=None):
    result = {'status': 'error', 'python': sys.executable, 'python_version': sys.version.split()[0]}
    try:
        _stage('Checking Python version', progress_report)
        if sys.version_info[:2] != (3, 7) or struct.calcsize('P') != 8:
            raise RuntimeError('The private crYOLO runtime requires Python 3.7 x64.')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['OMP_NUM_THREADS'] = '2'
        with contextlib.redirect_stdout(sys.stderr):
            _stage('Importing package metadata and NumPy', progress_report)
            import pkg_resources
            import numpy as np
            _stage('Importing TensorFlow', progress_report)
            import tensorflow as tf
            _stage('Importing Keras and crYOLO', progress_report)
            import keras
            import cryolo
            _stage('Importing crYOLO prediction and training modules', progress_report)
            from cryolo.frontend import YOLO  # noqa: F401
            import cryolo.predict  # noqa: F401
            import cryolo.train  # noqa: F401
            versions = {'cryolo': pkg_resources.get_distribution('cryolo').version,
                        'tensorflow': tf.__version__, 'keras': keras.__version__,
                        'numpy': np.__version__}
            expected = {'cryolo': '1.9.9', 'tensorflow': '1.15.4',
                        'keras': '2.3.1', 'numpy': '1.18.5'}
            if versions != expected:
                raise RuntimeError('Unexpected runtime versions: ' + repr(versions))
            _stage('Checking TensorFlow CPU execution', progress_report)
            with tf.Graph().as_default():
                value = tf.constant([2.0]) + tf.constant([3.0])
                with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0},
                        intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)) as session:
                    if session.run(value).tolist() != [5.0]:
                        raise RuntimeError('TensorFlow CPU execution failed.')
            _stage('Checking initialization weights', progress_report)
            backend = Path(cryolo.__file__).resolve().parent / 'full_yolo_backend.h5'
            digest = hashlib.sha256()
            with backend.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != BACKEND_SHA256:
                raise RuntimeError('The upstream crYOLO initialization weights are incomplete or invalid.')
        _stage('Ready', progress_report)
        result.update(status='ok', versions=versions, cpu_execution=True,
                      backend_sha256=digest.hexdigest())
    except Exception as exc:
        result['error'] = '{}: {}'.format(type(exc).__name__, exc)
        result['error_type'] = type(exc).__name__
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--progress-report')
    arguments = parser.parse_args()
    report = check_runtime(arguments.progress_report)
    print(json.dumps(report, sort_keys=True))
    sys.exit(0 if report['status'] == 'ok' else 1)

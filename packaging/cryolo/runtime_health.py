"""Read-only health probe for the private stock crYOLO runtime (Python 3.7)."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

BACKEND_SHA256 = '0eb044b4018be4b8c530389843d9650569751439fdab2880f3020e65ff2ff358'


def check_runtime():
    result = {'status': 'error', 'python': sys.executable, 'python_version': sys.version.split()[0]}
    try:
        if sys.version_info[:2] != (3, 7) or struct.calcsize('P') != 8:
            raise RuntimeError('The private crYOLO runtime requires Python 3.7 x64.')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['OMP_NUM_THREADS'] = '2'
        with contextlib.redirect_stdout(sys.stderr):
            import pkg_resources
            import numpy as np
            import tensorflow as tf
            import keras
            import cryolo
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
            with tf.Graph().as_default():
                value = tf.constant([2.0]) + tf.constant([3.0])
                with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0},
                        intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)) as session:
                    if session.run(value).tolist() != [5.0]:
                        raise RuntimeError('TensorFlow CPU execution failed.')
            backend = Path(cryolo.__file__).resolve().parent / 'full_yolo_backend.h5'
            digest = hashlib.sha256()
            with backend.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != BACKEND_SHA256:
                raise RuntimeError('The upstream crYOLO initialization weights are incomplete or invalid.')
        result.update(status='ok', versions=versions, cpu_execution=True,
                      backend_sha256=digest.hexdigest())
    except Exception as exc:
        result['error'] = '{}: {}'.format(type(exc).__name__, exc)
    return result


if __name__ == '__main__':
    report = check_runtime()
    print(json.dumps(report, sort_keys=True))
    sys.exit(0 if report['status'] == 'ok' else 1)

"""Exercise a stock crYOLO network and CLI with a synthetic, untrained model.

This is packaging/runtime validation only. No user data or trained model is used.
Run with the isolated Python 3.7 interpreter.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def verify(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '4'
    import numpy as np
    import tensorflow as tf
    from keras import backend as K
    from cryolo.frontend import YOLO
    from PIL import Image
    session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4,
                                               inter_op_parallelism_threads=2))
    K.set_session(session)
    started = time.monotonic()
    # This is the unmodified upstream API. It can initialize its feature
    # extractor from the upstream backend weights; the detection head is untrained.
    model = YOLO('PhosaurusNet', 320, 20, [80.0, 80.0])
    inputs = np.random.RandomState(29).normal(size=(1, 320, 320, 1)).astype('float32')
    boxes = np.zeros((1, 1, 1, 1, 20, 4), dtype='float32')
    prediction = model.model.predict([inputs, boxes])
    if prediction.size == 0 or not np.isfinite(prediction).all():
        raise RuntimeError('The stock crYOLO network did not return finite predictions.')
    weights = directory / 'synthetic-untrained.h5'
    from cryolo.ExtendedModelCheckpoint import ExtendedModelCheckpoint
    checkpoint = ExtendedModelCheckpoint(str(weights), monitor='val_loss', save_weights_only=True, save_best_only=True)
    checkpoint.set_model(model.model)
    checkpoint.set_anchors(model.anchors)
    checkpoint.set_fine_tune(False, 2)
    checkpoint.set_version('1.9.9')
    checkpoint.on_epoch_end(0, {'val_loss': 0.0})
    K.clear_session()
    image_directory = directory / 'images'
    image_directory.mkdir(exist_ok=True)
    image = np.random.RandomState(29).randint(80, 180, size=(320, 320)).astype('uint8')
    Image.fromarray(image, mode='L').save(str(image_directory / 'synthetic.png'))
    config = directory / 'synthetic-config.json'
    config.write_text(json.dumps({'model': {'architecture': 'PhosaurusNet', 'input_size': 320,
        'anchors': [80, 80], 'max_box_per_image': 20, 'norm': 'STANDARD'}}), encoding='utf-8')
    output = directory / 'prediction'
    command = [sys.executable, '-m', 'cryolo.predict', '-c', str(config), '-w', str(weights),
               '-i', str(image_directory), '-o', str(output), '-t', '0.99']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, timeout=240)
    (directory / 'synthetic-cli.log').write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('crYOLO prediction CLI failed: ' + result.stdout[-4000:])
    cboxes = list(output.rglob('synthetic.cbox'))
    if len(cboxes) != 1:
        raise RuntimeError('Stock crYOLO did not write exactly one synthetic CBOX result.')
    weights.unlink()  # Never deliver the synthetic untrained detector as a model.
    report = {'status': 'ok', 'platform': sys.platform, 'python': sys.version,
              'tensorflow': tf.__version__, 'network_output_shape': list(prediction.shape),
              'elapsed_seconds': time.monotonic() - started, 'cbox_created': True,
              'note': 'Real stock crYOLO network/CLI execution with synthetic noise and an untrained detection head. No capsid training, accuracy or private model validation.'}
    (directory / 'runtime-inference.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1]), indent=2))

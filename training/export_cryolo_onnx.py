"""Export a trained crYOLO PhosaurusNet checkpoint for offline CPU inference.

Run this build tool in an isolated Python 3.7 environment containing unchanged
crYOLO 1.9.9, TensorFlow 1.15.4, Keras 2.3.1, tf2onnx 1.14.0 and onnx 1.12.0.
It uses crYOLO's public model API and native checkpoint metadata. The resulting
model embeds its inference contract, so prediction needs no separate config.
Export does not grant additional rights to distribute third-party artifacts.

Example:
    python training/export_cryolo_onnx.py --weights trained.h5 \
        --config cryolo-config.json --output trained.onnx

Optional reference images produce private TensorFlow input/output tensors for
independent numerical verification. Keep those tensors out of source control.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


METADATA_KEY = 'saru_cryolo'
METADATA_FORMAT = 'saru-cryolo-onnx-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def model_metadata(source_sha256, input_size, anchors, input_name, output_name):
    """Build the restricted single-particle inference contract."""
    if not re.fullmatch(r'[0-9a-f]{64}', source_sha256):
        raise ValueError('The source H5 must have a lowercase SHA256 digest.')
    if (not isinstance(input_size, list) or input_size != [320, 320]
            or any(isinstance(n, bool) or not isinstance(n, int) for n in input_size)):
        raise ValueError('The verified SARU export contract requires input_size [320, 320].')
    anchors = [float(value) for value in anchors]
    if len(anchors) != 2 or any(not math.isfinite(x) or x <= 0 for x in anchors):
        raise ValueError('Exactly one finite positive learned anchor pair is required.')
    if not input_name or not output_name:
        raise ValueError('Input and output tensor names are required.')
    return {
        'format': METADATA_FORMAT,
        'architecture': 'PhosaurusNet',
        'source_h5_sha256': source_sha256,
        'input_name': input_name,
        'output_name': output_name,
        'input_size': list(input_size),
        'input_layout': 'NHWC',
        'output_layout': 'NHWAC',
        'anchors': anchors,
        'normalization': 'STANDARD',
        'preprocessing': 'pil-grayscale-bilinear-standard-v1',
        'labels': ['particle'],
        'filament_model': False,
        'opset': 13,
        'nms_threshold': 0.3,
        'estimated_size_rounding': 'truncate',
        'extraction_box_size': 1,
    }


def export(args):
    # Keep the legacy build dependencies out of ordinary application imports.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = str(args.cpu_threads)
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import backend as K
    from PIL import Image
    import cryolo
    from cryolo import utils
    from cryolo.frontend import YOLO
    import onnx
    import tf2onnx

    config = json.loads(args.config.read_text(encoding='utf-8'))
    network = config['model']
    if network.get('architecture') != 'PhosaurusNet':
        raise ValueError('This exporter is verified for PhosaurusNet only.')
    if network.get('norm', 'STANDARD') != 'STANDARD':
        raise ValueError('This exporter requires STANDARD normalization.')
    if network.get('num_patches', 1) != 1 or network.get('filter'):
        raise ValueError('Use one full-frame patch without additional filtering.')
    native_metadata = utils.get_custom_paramters_from_model(str(args.weights))
    if native_metadata.get('filament_model', False):
        raise ValueError('Filament checkpoints are not supported by this contract.')
    anchors = native_metadata.get('anchors')
    if anchors is None:
        raise ValueError('The checkpoint must contain native learned anchor metadata.')
    source_hash = sha256(args.weights)
    # Validate the configuration before allocating the model.
    model_metadata(source_hash, network['input_size'], anchors, 'input', 'output')
    session = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=args.cpu_threads,
        inter_op_parallelism_threads=args.cpu_threads))
    K.set_session(session)
    K.set_learning_phase(0)
    started = time.monotonic()
    temporary = args.output.with_name(args.output.name + '.part')
    try:
        parameters = {}
        if 'num_free_layers' in native_metadata:
            parameters['num_fine_tune_layers'] = int(native_metadata['num_free_layers'])
        model = YOLO(architecture='PhosaurusNet', input_size=list(network['input_size']),
                     max_box_per_image=int(network['max_box_per_image']),
                     anchors=anchors, input_depth=1, labels=['particle'],
                     pretrained_weights=str(args.weights), filament_model=False,
                     **parameters)
        input_name = model.model.inputs[0].name
        output_name = model.model.outputs[0].name
        contract = model_metadata(source_hash, network['input_size'], anchors,
                                  input_name, output_name)
        output_shape = list(model.model.output_shape)
        if len(output_shape) != 5 or output_shape[-2:] != [1, 6]:
            raise ValueError('Unexpected native single-particle output shape: ' + str(output_shape))
        references = []
        if args.reference_directory is not None:
            images = sorted(args.reference_directory.rglob('*.png'))
            if not images:
                raise ValueError('No reference PNG images found.')
            tensors = {}
            dummy = np.zeros((1, 1, 1, 1, int(network['max_box_per_image']), 4), np.float32)
            for index, path in enumerate(images):
                key = 'image_{:04d}'.format(index)
                with Image.open(path) as image:
                    size = image.size
                    pixels = np.array(image.convert('RGB').convert('L').resize(
                        tuple(network['input_size']), Image.BILINEAR))
                tensor = utils.normalize(pixels)[None, ..., None].astype(np.float32)
                raw = model.model.predict([tensor, dummy], batch_size=1)
                tensors[key + '_input'] = tensor
                tensors[key + '_raw'] = raw
                references.append({'key': key,
                    'relative_path': str(path.relative_to(args.reference_directory)),
                    'source_sha256': sha256(path), 'source_size': list(size)})
            args.reference_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(args.reference_output), **tensors)
        frozen = tf.graph_util.convert_variables_to_constants(
            session, session.graph.as_graph_def(), [output_name.split(':')[0]])
        converted, _ = tf2onnx.convert.from_graph_def(frozen,
            input_names=[input_name], output_names=[output_name], opset=13)
        onnx.helper.set_model_props(converted, {
            METADATA_KEY: json.dumps(contract, sort_keys=True, separators=(',', ':'))})
        onnx.checker.check_model(converted)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(converted, str(temporary))
        temporary.replace(args.output)
        report = {
            'format': 'saru-cryolo-onnx-export-v1', 'status': 'completed',
            'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'runtime': {'python': __import__('sys').version.split()[0],
                        'cryolo': cryolo.__version__, 'tensorflow': tf.__version__,
                        'keras': keras.__version__, 'tf2onnx': tf2onnx.__version__,
                        'onnx': onnx.__version__, 'numpy': np.__version__},
            'stock_package_modified': False,
            'source_config_sha256': sha256(args.config),
            'exporter_sha256': sha256(__file__), 'metadata': contract,
            'onnx_sha256': sha256(args.output), 'onnx_bytes': args.output.stat().st_size,
            'raw_output_shape': output_shape, 'elapsed_seconds': time.monotonic() - started,
            'reference_images': references,
            'reference_tensors_sha256': sha256(args.reference_output) if references else None,
            'scope': 'Format conversion. Numerical and detection equivalence require separate verification.',
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': 'completed', 'output': str(args.output),
                          'report': str(args.report), 'reference_count': len(references)}), flush=True)
        return report
    finally:
        if temporary.exists():
            temporary.unlink()
        K.clear_session()
        session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--reference-directory', type=Path)
    parser.add_argument('--reference-output', type=Path)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error('CPU thread count must be positive.')
    args.weights, args.config, args.output = (p.resolve() for p in
                                             (args.weights, args.config, args.output))
    if not args.weights.is_file() or not args.config.is_file():
        parser.error('Existing trained H5 weights and model configuration are required.')
    if args.output.suffix.lower() != '.onnx' or args.output.exists():
        parser.error('Output must be a new .onnx file; existing models are not overwritten.')
    if bool(args.reference_directory) != bool(args.reference_output):
        parser.error('Supply both --reference-directory and --reference-output, or neither.')
    if args.reference_directory:
        args.reference_directory = args.reference_directory.resolve()
        args.reference_output = args.reference_output.resolve()
        if not args.reference_directory.is_dir() or args.reference_output.exists():
            parser.error('Reference directory must exist and tensor output must be new.')
        if args.reference_output.suffix.lower() != '.npz':
            parser.error('Reference tensor output must end in .npz.')
    args.report = (args.report or args.output.with_suffix('.export.json')).resolve()
    if args.report.exists():
        parser.error('Export report must be a new file.')
    export(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Run the unmodified crYOLO CLI and record the actual training provenance.

Requires the private export produced by prepare_cryolo_data.py and a separately
installed crYOLO 1.9.9 Python environment. No crYOLO source or pretrained model
is redistributed by this script. Full-fit validation is in-sample monitoring.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_training_log(text):
    """Count completed epochs, not merely the printed start of an epoch."""
    completed = []
    for match in re.finditer(r'Epoch (\d+)/(\d+)(.*?)(?=Epoch \d+/\d+|\Z)', text, re.S):
        loss = re.findall(r'- loss: ([\d.eE+-]+) - val_loss: ([\d.eE+-]+)', match.group(3))
        if loss:
            train, valid = map(float, loss[-1])
            completed.append({'epoch': int(match.group(1)), 'loss': train,
                              'monitor_loss_in_sample': valid})
    selected = re.findall(r'Epoch\s+(\d+): val_loss improved[^\n]*saving model', text)
    return {'completed_epochs': completed,
            'selected_checkpoint_epoch': int(selected[-1]) if selected else None}


def runtime_versions(python):
    program = ("import json,sys,cryolo,tensorflow,numpy,keras; "
               "print(json.dumps({'python':sys.version.split()[0],"
               "'cryolo':cryolo.__version__,'tensorflow':tensorflow.__version__,"
               "'keras':keras.__version__,'numpy':numpy.__version__}))")
    result = subprocess.run([str(python), '-c', program], capture_output=True,
                            text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])



def native_target_audit(python, config_path):
    """Check the actual stock generator retains full-frame positive targets."""
    program = r"""
import json,sys,numpy as np
from pathlib import Path
from cryolo import preprocessing as pre,config_tools,utils
from cryolo.patchwisebatchgenerator2 import PatchwiseBatchGenerator
p=Path(sys.argv[1]).resolve(); c=json.loads(p.read_text())
pairs=pre.find_image_annotation_pairs_by_dir(str(p.parent/c['train']['train_annot_folder']),str(p.parent/c['train']['train_image_folder']))
grid=tuple(map(int,config_tools.get_gridcell_dimensions(c)))
parsed=pre.parse_annotation(pairs,grid_dims=(grid[0],grid[1],1),anchor_size=int(c['model']['anchors'][0]),input_size=c['model']['input_size'])
sizes=pre.get_image_size_distr([a[0] for a in pairs])
mode=utils.get_cryolo_mode(sizes,c['model']['input_size'])
wh=c['model']['input_size']; wh=wh if isinstance(wh,list) else [wh,wh]
g={'IMAGE_H':wh[1],'IMAGE_W':wh[0],'GRID_H':grid[1],'GRID_W':grid[0],'BOX':1,'LABELS':['particle'],'CLASS':1,'ANCHORS':utils.get_anchors(c,image_sizes=sizes),'BATCH_SIZE':1,'TRUE_BOX_BUFFER':c['model']['max_box_per_image'],'NUM_PATCHES':1}
np.random.seed(42)
gen=PatchwiseBatchGenerator(parsed['images'],g,np.ones((1,1)),cryolo_mode=mode,norm=utils.get_normalization_func('STANDARD'),jitter=False,overlap=0,name='audit',train_times=1,is_filament_training=False)
counts=[int(gen[i][1][...,4].sum()) for i in range(len(gen))]
print(json.dumps({'stock_mode':mode.name,'image_count':len(parsed['images']),'parsed_positive_boxes':sum(len(a['object']) for a in parsed['images']),'generator_positive_targets':sum(counts),'audit_augmentation':False,'audit_batch_size':1}))
"""
    result = subprocess.run([str(python), '-c', program, str(config_path)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def record_run(*, config_path, export_audit_path, log_path, output_path,
               python, returncode, elapsed_seconds, command, started_utc=None):
    config_path, log_path = Path(config_path), Path(log_path)
    config = json.loads(config_path.read_text())
    audit = json.loads(Path(export_audit_path).read_text())
    parsed = parse_training_log(log_path.read_text(errors='replace'))
    runfiles = sorted((config_path.parent / config.get('other', {}).get('log_path', 'logs') / 'runfiles').glob('*.json'))
    run = json.loads(runfiles[-1].read_text())['run'] if runfiles else {}
    case_ids = lambda key: sorted(int(Path(p).stem.split('_')[-1]) for p in run.get(key, []))
    train_ids, monitor_ids = case_ids('train_images'), case_ids('valid_images')
    weights = config_path.parent / config['train']['saved_weights_name']
    expected = sorted(audit['all_cases_used_for_full_fit_gradients'])
    target_audit = native_target_audit(python, config_path)
    completed = (returncode == 0 and weights.is_file()
                 and len(parsed['completed_epochs']) == config['train']['nb_epoch']
                 and train_ids == expected
                 and target_audit['stock_mode'] == 'FIXED_SIZE'
                 and target_audit['parsed_positive_boxes'] == target_audit['generator_positive_targets'])
    sources = {}
    for label, value in [('pretrained', config['train'].get('pretrained_weights')),
                         ('backend', config['model'].get('backend_weights'))]:
        if value:
            p = config_path.parent / value
            sources[label] = {'name': p.name, 'sha256': sha256(p), 'bytes': p.stat().st_size}
    report = {
        'format': 'SARU-crYOLO-training-run-v1',
        'status': 'completed' if completed else 'incomplete_or_failed',
        'started_utc': started_utc, 'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'elapsed_seconds': elapsed_seconds, 'returncode': returncode,
        'runtime': runtime_versions(python), 'command': command,
        'native_generator_audit': target_audit,
        'config_sha256': sha256(config_path), 'export_audit_sha256': sha256(export_audit_path),
        'training_log_sha256': sha256(log_path),
        'dataset_sha256': audit['dataset_sha256'], 'folds_sha256': audit['folds_sha256'],
        'source_decks': audit['source_decks'], 'initialization': sources,
        'model': config['model'],
        'training': {k:v for k,v in config['train'].items()
                     if not k.endswith('_folder') and k not in ('pretrained_weights', 'saved_weights_name')},
        'fine_tune_layers': int(command[command.index('-lft') + 1]) if '-lft' in command else None,
        'train_case_ids': train_ids, 'monitor_case_ids': monitor_ids,
        'all_cases_received_gradients_each_completed_epoch': train_ids == expected,
        'monitor_scope': 'in_sample' if train_ids == monitor_ids else 'separate_cases',
        'best_checkpoint_selection': 'Minimum stock crYOLO monitor loss; this is in-sample for the full fit.',
        'annotation_semantics': audit['background_semantics'],
        'confirmed_large_boxes': audit['confirmed_large_box_count'],
        'confirmed_blank_case_ids': audit['confirmed_blank_case_ids'],
        'unknown_only_case_ids': audit['unknown_only_case_ids'],
        'empty_annotations': audit['empty_annotation_count'],
        'augmentation': 'Stock crYOLO augmentation enabled unless --skip_augmentation is in command.',
        'exact_bitwise_reproducibility': False,
        'reproducibility_note': 'Stock --seed controls its dataset split; TensorFlow/augmentation execution can differ across runs.',
        **parsed,
    }
    # Configuration paths are kept in the private run directory, not the portable report.
    report['model'].pop('backend_weights', None)
    if weights.is_file():
        report['weights'] = {'name': weights.name, 'sha256': sha256(weights), 'bytes': weights.stat().st_size}
    Path(output_path).write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-directory', type=Path, required=True)
    p.add_argument('--cryolo-python', type=Path, required=True)
    p.add_argument('--pretrained-weights', type=Path, required=True)
    p.add_argument('--backend-weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--input-size', type=int, default=320)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--fine-tune-layers', type=int, default=2)
    a = p.parse_args(argv)
    if min(a.epochs, a.cpu_threads, a.fine_tune_layers) < 1 or a.input_size < 32 or a.input_size % 32:
        p.error('Use positive counts and an input size divisible by 32.')
    for required in (a.cryolo_python, a.pretrained_weights, a.backend_weights):
        if not required.is_file():
            p.error(f'Required runtime/model file is missing: {required}')
    base = a.export_directory.resolve()
    audit = base / 'export-audit.json'
    source = base / 'full' / 'config.json'
    if not audit.is_file() or not source.is_file():
        p.error('The export directory must contain export-audit.json and full/config.json.')
    a.output = a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output must be a new or empty directory; existing runs are not overwritten.')
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(source.read_text())
    for part, keys in [('train', ['train_image_folder','train_annot_folder']),
                       ('valid', ['valid_image_folder','valid_annot_folder'])]:
        for key in keys:
            config[part][key] = str((source.parent / config[part][key]).resolve())
    config['model']['input_size'] = [a.input_size, a.input_size]
    config['model']['backend_weights'] = str(a.backend_weights.resolve())
    config['train'].update(pretrained_weights=str(a.pretrained_weights.resolve()), nb_epoch=a.epochs,
                           saved_weights_name='capsid-cryolo.h5')
    config['other'] = {'log_path':'logs'}
    config_path = a.output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    command = [str(a.cryolo_python.resolve()), '-m', 'cryolo.train', '-c', 'config.json', '-w', '0',
               '--fine_tune', '-lft', str(a.fine_tune_layers), '-nc', str(a.cpu_threads),
               '-e', str(a.epochs+1)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', PYTHONUNBUFFERED='1')
    preflight = native_target_audit(a.cryolo_python, config_path)
    if (preflight['stock_mode'] != 'FIXED_SIZE'
            or preflight['parsed_positive_boxes'] != preflight['generator_positive_targets']):
        p.error('Stock generator did not preserve all full-frame labels; inspect the input size/annotations.')
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    start = time.monotonic()
    log_path = a.output / 'training.log'
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=a.output, env=env, stdout=log, stderr=subprocess.STDOUT)
    report = record_run(config_path=config_path, export_audit_path=audit, log_path=log_path,
                        output_path=a.output/'training-provenance.json', python=a.cryolo_python,
                        returncode=result.returncode, elapsed_seconds=time.monotonic()-start,
                        command=command, started_utc=started)
    print(json.dumps({'status':report['status'], 'output':str(a.output)}))
    return 0 if report['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

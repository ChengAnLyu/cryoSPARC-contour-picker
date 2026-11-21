"""Predict varying screenshot sizes with stock crYOLO, retaining one model.

Run with the separate crYOLO Python environment. Images are grouped by their
native dimensions because the official prediction API rejects mixed dimensions
within a call. Every CBOX is written by crYOLO's own writer and postprocessing.
"""
import argparse
import copy
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--images',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--backend-weights',type=Path)
    p.add_argument('--threshold',type=float,default=.3)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=1)
    a=p.parse_args(argv)
    if not 0 <= a.threshold <= 1 or min(a.cpu_threads,a.batch_size)<1:
        p.error('Threshold must be in [0,1]; thread and batch counts must be positive.')
    a.config=a.config.resolve();a.weights=a.weights.resolve();a.images=a.images.resolve();a.output=a.output.resolve()
    if not a.config.is_file() or not a.weights.is_file() or not a.images.is_dir():
        p.error('Configuration, weights and image directory must exist.')
    config=json.loads(a.config.read_text())
    size=config['model']['input_size']
    if (not isinstance(size,list) or len(size)!=2
            or any(not isinstance(n,int) or n<32 or n%32 for n in size)):
        p.error('Use explicit fixed input_size [width,height], each divisible by32.')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a new or empty output directory to avoid stale predictions.')
    a.output.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
    from keras import backend as K
    from PIL import Image
    import cryolo
    from cryolo import config_tools,utils
    from cryolo.frontend import YOLO
    from cryolo.predict import do_prediction
    K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.cpu_threads,inter_op_parallelism_threads=a.cpu_threads)))
    if config_tools.get_number_patches(config)!=1:
        p.error('This validated screenshot wrapper requires one patch.')
    # Same extraction-size override as SARU: preserve edge capsids while using
    # CBOX EstWidth/EstHeight for the learned size. Model anchors are unchanged.
    config['model']['particle_diameter']=1
    backend=str(a.backend_weights.resolve()) if a.backend_weights else config['model'].get('backend_weights')
    if backend:
        config['model']['backend_weights']=str((a.config.parent/backend).resolve())
        if not Path(config['model']['backend_weights']).is_file():
            p.error('Backend initialization weights do not exist.')
    (a.output/'inference-config.json').write_text(json.dumps(config,indent=2)+'\n')
    paths=sorted(a.images.glob('*.png'))
    if not paths:
        p.error('No PNG screenshots were found.')
    groups=defaultdict(list)
    for path in paths:
        with Image.open(path) as image:
            if image.mode!='L':
                p.error('Expected exported single-channel PNG: '+path.name)
            groups[image.size].append(str(path))
    meta=utils.get_custom_paramters_from_model(str(a.weights))
    if 'anchors' not in meta:
        p.error('Weights must have native crYOLO anchor metadata from real training.')
    if meta.get('filament_model',False):
        p.error('This capsid prediction client requires a single-particle model.')
    kwargs={}
    if 'num_free_layers' in meta:kwargs['num_fine_tune_layers']=meta['num_free_layers']
    started=time.monotonic()
    model=YOLO(architecture=config['model']['architecture'],input_size=list(config['model']['input_size']),
               input_depth=1,labels=['particle'],max_box_per_image=config['model']['max_box_per_image'],
               anchors=meta['anchors'],backend_weights=config['model'].get('backend_weights'),
               pretrained_weights=str(a.weights),filament_model=meta.get('filament_model',False),**kwargs)
    model_load_seconds=time.monotonic()-started
    summaries=[]
    for (width,height),group in sorted(groups.items()):
        group_started=time.monotonic()
        results,about=do_prediction(config_path=str(a.config),weights_path=str(a.weights),input_path=group,
            num_patches=1,obj_threshold=a.threshold,write_empty=True,config_pre=copy.deepcopy(config),
            overlap=0,filament_mode=False,num_images_batch_prediction=a.batch_size,num_gpus=1,
            num_cpus=a.cpu_threads,yolo=model,otf=False,normalization=config['model'].get('norm','STANDARD'),
            normalization_margin=0,skip_picked=False,write_direct=True,min_distance=0,
            outdir=str(a.output),min_size=None,max_size=None)
        if len(results)!=len(group):
            raise RuntimeError('Native result count differs from source image count.')
        summaries.append({'width':width,'height':height,'images':len(group),
                          'elapsed_seconds':time.monotonic()-group_started,
                          'filtered_detections':sum(len(boxes) for result in results for boxes in result['boxes'])})
    missing=[path.name for path in paths if not (a.output/'CBOX'/(path.stem+'.cbox')).is_file()]
    report={'format':'SARU-stock-crYOLO-batch-prediction-v1','cryolo_version':cryolo.__version__,
            'tensorflow_version':tf.__version__,'image_count':len(paths),'shape_groups':summaries,
            'threshold':a.threshold,'cpu_threads':a.cpu_threads,'prediction_batch_size':a.batch_size,
            'model_instances':1,'model_load_seconds':model_load_seconds,
            'elapsed_seconds_including_model_load':time.monotonic()-started,
            'weights_sha256':sha256(a.weights),'config_sha256':sha256(a.config),
            'inference_config_sha256':sha256(a.output/'inference-config.json'),
            'extraction_diameter_override':1,'learned_box_sizes':'CBOX EstWidth and EstHeight',
            'learned_anchors':[float(value) for value in meta['anchors']],
            'scope':'Prediction provenance only; no held-out accuracy claim.',
            'outputs':[{'image':path.name,'image_sha256':sha256(path),
                        'cbox_sha256':sha256(a.output/'CBOX'/(path.stem+'.cbox'))}
                       for path in paths if (a.output/'CBOX'/(path.stem+'.cbox')).is_file()],
            'missing_cbox':missing,'status':'completed' if not missing else 'incomplete'}
    (a.output/'prediction-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    K.clear_session()
    print(json.dumps({'status':report['status'],'image_count':len(paths),'missing_cbox':missing}))
    return 1 if missing else 0


if __name__=='__main__':
    raise SystemExit(main())

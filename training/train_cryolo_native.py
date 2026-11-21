"""Train unchanged crYOLO APIs with explicit full-dataset loss aggregation.

Run in the separately installed crYOLO 1.9.9 environment (Python 3.7).
The architecture, loss, augmentation, target generator and checkpoint writer
come directly from that installation. This driver averages every monitoring
batch explicitly; monitoring on the full-fit data is not held-out validation.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--fine-tune-layers',type=int,default=2)
    p.add_argument('--snapshot-every',type=int,default=5)
    a=p.parse_args(argv)
    a.config=a.config.resolve();base=a.config.parent
    c=json.loads(a.config.read_text())
    if not isinstance(c['model']['input_size'],list) or len(c['model']['input_size'])!=2:
        p.error('Explicit [width,height] fixed-size input is required.')
    if min(a.cpu_threads,a.fine_tune_layers,a.snapshot_every)<1:p.error('Counts must be positive.')
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import backend as K
    from keras.optimizers import Adam
    import cryolo
    from cryolo import preprocessing as pre,utils,config_tools
    from cryolo.frontend import YOLO,warmup_bs
    from cryolo.patchwisebatchgenerator2 import PatchwiseBatchGenerator
    from cryolo.ExtendedModelCheckpoint import ExtendedModelCheckpoint
    K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=a.cpu_threads,inter_op_parallelism_threads=a.cpu_threads)))
    paths=lambda part,key:str((base/c[part][key]).resolve())
    train_pairs=pre.find_image_annotation_pairs_by_dir(paths('train','train_annot_folder'),paths('train','train_image_folder'))
    valid_pairs=pre.find_image_annotation_pairs_by_dir(paths('valid','valid_annot_folder'),paths('valid','valid_image_folder'))
    grid=tuple(map(int,config_tools.get_gridcell_dimensions(c)))
    parse=lambda pairs:pre.parse_annotation(pairs,grid_dims=(grid[0],grid[1],1),anchor_size=int(c['model']['anchors'][0]),input_size=c['model']['input_size'])['images']
    train_images,valid_images=parse(train_pairs),parse(valid_pairs)
    if not train_images or not valid_images:p.error('Both training and monitoring images are required.')
    sizes=pre.get_image_size_distr([pair[0] for pair in train_pairs])
    mode=utils.get_cryolo_mode(sizes,c['model']['input_size'])
    if mode.name!='FIXED_SIZE':p.error('Full-frame fixed-size mode is required.')
    batch=c['train']['batch_size']
    if len(train_images)%batch or len(valid_images)%batch:
        p.error('Use a batch size dividing both dataset sizes so no examples are duplicated for padding.')
    if config_tools.get_number_patches(c)!=1:p.error('This full-frame driver requires one patch.')
    weights=(base/c['train']['pretrained_weights']).resolve()
    backend=(base/c['model']['backend_weights']).resolve()
    if not weights.is_file() or not backend.is_file():p.error('Official initialization files must exist.')
    anchors=utils.get_anchors(c,image_sizes=sizes)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
    model=YOLO(architecture=c['model']['architecture'],input_size=c['model']['input_size'],
               max_box_per_image=c['model']['max_box_per_image'],anchors=anchors,
               backend_weights=str(backend),pretrained_weights=str(weights),fine_tune=True,
               num_fine_tune_layers=a.fine_tune_layers,input_depth=1,labels=['particle'])
    model.batch_size=batch;model.warmup_epochs=0;model.debug=False
    for key in ['object_scale','no_object_scale','coord_scale','class_scale']:setattr(model,key,c['train'][key])
    K.set_value(warmup_bs,0)
    model.model.compile(optimizer=Adam(lr=c['train']['learning_rate'],beta_1=.9,beta_2=.999,epsilon=1e-8,decay=0),loss=model.custom_loss)
    wh=c['model']['input_size']
    gen_config={'IMAGE_H':wh[1],'IMAGE_W':wh[0],'GRID_H':grid[1],'GRID_W':grid[0],'BOX':1,'LABELS':['particle'],
                'CLASS':1,'ANCHORS':anchors,'BATCH_SIZE':batch,'TRUE_BOX_BUFFER':c['model']['max_box_per_image'],'NUM_PATCHES':1}
    def generator(images,jitter,shuffle):
        return PatchwiseBatchGenerator(images,gen_config,np.ones((1,1)),cryolo_mode=mode,shuffle=shuffle,
             norm=utils.get_normalization_func(c['model'].get('norm','STANDARD')),jitter=jitter,
             overlap=0,name='train' if jitter else 'monitor',train_times=1,is_filament_training=False)
    train_gen=generator(train_images,True,True);valid_gen=generator(valid_images,False,False)
    parsed_targets=sum(len(image['object']) for image in valid_images)
    assigned_targets=sum(int(valid_gen[i][1][...,4].sum()) for i in range(len(valid_gen)))
    if parsed_targets!=assigned_targets:p.error('Monitoring generator lost positive targets; inspect box/grid collisions.')
    run_dir=base/c.get('other',{}).get('log_path','logs')/'runfiles';run_dir.mkdir(parents=True,exist_ok=True)
    run={'train_images':[x['filename'] for x in train_images],'train_annot':[{'path':x['boxpath']} for x in train_images],
         'valid_images':[x['filename'] for x in valid_images],'valid_annot':[{'path':x['boxpath']} for x in valid_images]}
    (run_dir/'native-api-training.json').write_text(json.dumps({'run':run},indent=2)+'\n')
    output=base/c['train']['saved_weights_name'];last=output.with_name(output.stem+'-last'+output.suffix)
    def checkpoint(path,best_only):
        cb=ExtendedModelCheckpoint(str(path),monitor='val_loss',verbose=1,save_best_only=best_only,save_weights_only=True,mode='min')
        cb.set_model(model.model);cb.set_anchors(anchors);cb.set_fine_tune(True,a.fine_tune_layers);cb.set_version(cryolo.__version__)
        return cb
    best_cb=checkpoint(output,True);last_cb=checkpoint(last,False)
    epochs=c['train']['nb_epoch'];records=[];best=float('inf');selected=None
    metadata={'format':'SARU-crYOLO-native-API-training-v1','status':'running','started_utc':started,
        'runtime':{'python':__import__('sys').version.split()[0],'cryolo':cryolo.__version__,'tensorflow':tf.__version__,'keras':keras.__version__,'numpy':np.__version__},
        'model_class':'cryolo.frontend.YOLO','loss':'Unmodified YOLO.custom_loss','generator':'Unmodified PatchwiseBatchGenerator',
        'checkpoint_writer':'Unmodified ExtendedModelCheckpoint','stock_package_modified':False,
        'monitor_aggregation':'Arithmetic mean of all equal-size monitoring batch test_on_batch losses, each with reset_metrics=True.',
        'monitor_scope':'in_sample' if sorted(run['train_images'])==sorted(run['valid_images']) else 'separate_cases',
        'image_count':len(train_images),'positive_boxes':sum(len(x['object']) for x in train_images),
        'monitor_positive_targets':assigned_targets,'batch_size':batch,'steps_per_epoch':len(train_gen),
        'epochs_requested':epochs,'fine_tune_layers':a.fine_tune_layers,'cpu_threads':a.cpu_threads,
        'config_sha256':sha256(a.config),'driver_sha256':sha256(__file__),
        'initialization':{'pretrained_sha256':sha256(weights),'backend_sha256':sha256(backend)},
        'learning_rate':c['train']['learning_rate'],'loss_weights':{k:c['train'][k] for k in ['object_scale','no_object_scale','coord_scale','class_scale']},
        'augmentation':'Stock crYOLO augmentation enabled','input_size':wh,'native_mode':mode.name,'epochs':records}
    state=base/'native-training-state.json'
    print('Native API training: {} images, {} positive boxes, mode {}, {} monitoring targets'.format(len(train_images),metadata['positive_boxes'],mode.name,assigned_targets),flush=True)
    for epoch in range(1,epochs+1):
        epoch_clock=time.monotonic();losses=[]
        print('Epoch {}/{}'.format(epoch,epochs),flush=True)
        for i in range(len(train_gen)):
            inputs,targets=train_gen[i]
            loss=float(model.model.train_on_batch(inputs,targets,reset_metrics=True));losses.append(loss)
            if not math.isfinite(loss):raise RuntimeError('Non-finite training loss; refusing to publish weights.')
        monitors=[];target_count=0;gt_conf=[];max_conf=[]
        for i in range(len(valid_gen)):
            inputs,targets=valid_gen[i]
            loss=float(model.model.test_on_batch(inputs,targets,reset_metrics=True));monitors.append(loss)
            target_count+=int(targets[...,4].sum())
            if not math.isfinite(loss):raise RuntimeError('Non-finite monitoring loss.')
        mean_train=float(np.mean(losses));mean_monitor=float(np.mean(monitors))
        record={'epoch':epoch,'train_loss_mean':mean_train,'monitor_loss_mean':mean_monitor,
                'monitor_positive_targets':target_count,'elapsed_seconds':time.monotonic()-epoch_clock}
        records.append(record)
        if mean_monitor<best:best=mean_monitor;selected=epoch
        print('{0}/{0} - loss: {1:.9g} - val_loss: {2:.9g}'.format(len(train_gen),mean_train,mean_monitor),flush=True)
        logs={'loss':mean_train,'val_loss':mean_monitor};best_cb.on_epoch_end(epoch-1,logs);last_cb.on_epoch_end(epoch-1,logs)
        if epoch%a.snapshot_every==0:
            snap=output.with_name(output.stem+'-epoch{:02d}'.format(epoch)+output.suffix)
            checkpoint(snap,False).on_epoch_end(epoch-1,logs)
        metadata.update(epochs_completed=epoch,selected_checkpoint_epoch=selected,elapsed_seconds=time.monotonic()-clock)
        state.write_text(json.dumps(metadata,indent=2)+'\n')
        train_gen.on_epoch_end()
    metadata.update(status='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    weights={'name':output.name,'sha256':sha256(output),'bytes':output.stat().st_size},
                    last_weights={'name':last.name,'sha256':sha256(last),'bytes':last.stat().st_size})
    state.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'status':'completed','epochs':epochs,'selected_epoch':selected,'weights':str(output)}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

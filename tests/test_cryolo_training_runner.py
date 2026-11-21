from training.train_cryolo import parse_training_log


def test_training_log_does_not_count_started_or_partial_epoch():
    text = ('Epoch 1/20\n40/40 [===] - 44s 1s/step - loss: 0.9994 - val_loss: 5.2156\n'
            'Epoch 00001: val_loss improved from inf to 5.21560, saving model to weights_tmp.h5\n'
            'Epoch 2/20\n 4/40 [=] - ETA: 20s - loss: 0.25\n')
    result = parse_training_log(text)
    assert result['completed_epochs'] == [{'epoch':1, 'loss':0.9994, 'monitor_loss_in_sample':5.2156}]
    assert result['selected_checkpoint_epoch'] == 1


def test_selection_tracks_actual_checkpoint_not_final_epoch():
    text = ('Epoch 1/2\n40/40 [===] - loss: 0.2 - val_loss: 0.1\n'
            'Epoch 00001: val_loss improved from inf to 0.1000, saving model to weights_tmp.h5\n'
            'Epoch 2/2\n40/40 [===] - loss: 0.1 - val_loss: 0.2\n'
            'Epoch 00002: val_loss did not improve from 0.1000\n')
    result = parse_training_log(text)
    assert len(result['completed_epochs']) == 2
    assert result['selected_checkpoint_epoch'] == 1


def test_partial_fit_is_not_reported_complete_when_weights_exist(tmp_path, monkeypatch):
    import json
    import training.train_cryolo as runner
    config = {'model':{'input_size':[320,320]},
              'train':{'nb_epoch':20,'saved_weights_name':'capsid-cryolo.h5'},
              'other':{'log_path':'logs'}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'capsid-cryolo.h5').write_bytes(b'existing checkpoint from a partial run')
    log = ('Epoch 1/20\n40/40 - loss: 2.0 - val_loss: 0.9\n'
           'Epoch 00001: val_loss improved from inf to 0.9000, saving model to capsid-cryolo.h5\n')
    (tmp_path/'training.log').write_text(log)
    runfiles=tmp_path/'logs'/'runfiles';runfiles.mkdir(parents=True)
    (runfiles/'run.json').write_text(json.dumps({'run':{
        'train_images':['case_001.png','case_002.png'],
        'valid_images':['case_001.png','case_002.png']}}))
    audit={'all_cases_used_for_full_fit_gradients':[1,2],'dataset_sha256':'dataset',
           'folds_sha256':'folds','source_decks':[], 'background_semantics':'stock background',
           'confirmed_large_box_count':1, 'confirmed_blank_case_ids':[2],
           'unknown_only_case_ids':[], 'empty_annotation_count':1}
    (tmp_path/'export-audit.json').write_text(json.dumps(audit))
    monkeypatch.setattr(runner,'runtime_versions',lambda _: {'cryolo':'1.9.9'})
    monkeypatch.setattr(runner,'native_target_audit',lambda *args:{'stock_mode':'FIXED_SIZE',
        'parsed_positive_boxes':1,'generator_positive_targets':1})
    result=runner.record_run(config_path=tmp_path/'config.json',
        export_audit_path=tmp_path/'export-audit.json',log_path=tmp_path/'training.log',
        output_path=tmp_path/'report.json',python='unused',returncode=0,
        elapsed_seconds=1,command=['python','train_cryolo_native.py','--fine-tune-layers','2'])
    assert result['status']=='incomplete_or_failed'
    assert result['selected_checkpoint_epoch']==1
    assert result['fine_tune_layers']==2
    assert result['monitor_scope']=='in_sample'

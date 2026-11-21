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

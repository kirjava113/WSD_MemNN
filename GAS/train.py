# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2017/7/1
"""

import glob
import os
import sys
import time
import numpy as np
import pickle
import tensorflow as tf

from GAS.data_preparing import batch_generator, data_loading

sys.path.insert(0, "..")  # make sure to import utils and config successfully

# from utils.data import *
# from utils.glove import *
from utils import path
from utils import store_result
from GAS.config import MemNNConfig
import GAS.model as model

_path = path.WSD_path()
config = MemNNConfig()

# os.environ["CUDA_VISIBLE_DEVICES"] = '3,1,2,0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

best_info = {
    'i': 0,
    'acc_val': 0.0,
    'acc_train': 0.0,
}


def feed_dict(batch_data, is_training=True):
    [xf, xb, xfb, _, _, _, target_ids, sense_ids, instance_ids, glosses_ids, glosses_lenth, sense_mask] = batch_data

    feeds = {
        model.is_training: is_training,  # control keep_prob for dropout
        model.inputs_f: xf,
        model.inputs_b: xb,
        model.inputs: xfb,
        model.target_ids: target_ids,
        model.sense_ids: sense_ids,
        model.glosses: glosses_ids,
        model.glosses_lenth: glosses_lenth,
        model.sense_mask: sense_mask,
    }
    return feeds


def evaluate(eval_data, step=None):
    total_loss = []
    total_acc = []
    total_pred = []

    for batch_id, batch_data in enumerate(
            batch_generator(True, config.batch_size, eval_data, dict_data, word_to_id['<pad>'],
                            config.n_step_f, config.n_step_b, pad_last_batch=True)):
        feeds = feed_dict(batch_data, is_training=False)
        acc, loss, correct, pred, summary = \
            session.run([model.accuracy_op, model.loss_op, model.correct, model.predictions, model.summary_op], feeds)
        total_loss.append(loss)
        total_acc.append(acc)

        # map to true sense_id
        for i, predicted_sense_id in enumerate(pred):
            if batch_id * config.batch_size + i < len(eval_data):
                instance_id = batch_data[8][i]  # 8 is index of instance_ids
                predicted_sense = target_id_to_sense_id_to_sense[batch_data[6][i]][predicted_sense_id]
                total_pred.append([instance_id, predicted_sense])

    # tag='accuracy': coincide with variable name in the training process to show leaning curves in same graph
    if step is not None:
        summary_acc_average = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=np.mean(total_acc))])
        summary_val_writer.add_summary(summary_acc_average, global_step=step)
        summary_loss_average = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=np.mean(total_loss))])
        summary_val_writer.add_summary(summary_loss_average, global_step=step)

    return np.mean(total_acc), np.mean(total_loss), total_pred


def train():
    total_batch = 0  # same to global step
    n_batch = len(train_data) / config.batch_size
    last_improved = 0

    for i in range(1, config.n_epochs + 1):
        print ('::: EPOCH: %d :::' % i)
        for batch_id, batch_data in enumerate(batch_generator(True, config.batch_size, train_data, dict_data,
                                                              word_to_id['<pad>'], config.n_step_f, config.n_step_b,
                                                              pad_last_batch=True)):
            total_batch += 1
            feeds = feed_dict(batch_data, is_training=True)
            acc_train, loss_train, cuorrect, pred, step, summary, _ = \
                session.run([model.accuracy_op, model.loss_op, model.correct, model.predictions, model.global_step,
                             model.summary_op, model.train_op], feeds)

            if step % config.store_log_gap == 0:
                summary_train_writer.add_summary(summary, step)

            if step % config.evaluate_gap == 0:  # show evaluation result
                acc_val, loss_val, pred_val = evaluate(val_data, step)
                if acc_val > best_info['acc_val']:
                    last_improved = 1
                    best_info['i'] = i
                    best_info['acc_val'] = acc_val
                    best_info['acc_train'] = acc_train
                    saver.save(session, model_save_dir, global_step=step)  # path: model_save_prefix-global_step
                else:
                    last_improved += 1

                if config.print_batch:
                    print('step:%s (batch:%s/%s)' % (step, batch_id, n_batch))
                    msg = '%s--->\tacc:%.5f\tloss:%.5f\t'
                    print(msg % ('Train', acc_train, loss_train))
                    print(msg % ('Val', acc_val, loss_val))

            # early stop
            if last_improved / n_batch >= config.min_no_improvement:
                print('**>> Early stop <<**')
                return best_info


def test(result_path=None):
    _, _, pred_test = evaluate(test_data)
    if result_path is None:
        result_path = '../tmp/result.txt'
    store_result.write_result(pred_test, back_off_result, result_path)


if __name__ == "__main__":

    print(sys.argv)

    if len(sys.argv) > 1:
        config.random_config()  # random search to find best parameters
    print (vars(config))

    # Load data
    # ==================================================================================================================
    n_senses_from_target_id, init_emb, test_data, back_off_result, target_id_to_sense_id_to_sense, word_to_id, \
    dict_data, train_data, val_data = data_loading()

    # Training & Test
    # ==================================================================================================================

    # === Initial model
    Model = model.Model
    model = Model(config, n_senses_from_target_id, init_emb)

    print("Configuring TensorBoard and Saver")
    # === Configuring TensorBoard(Summary)
    # train summary
    train_log_dir = '../tmp/tf.log/GAS/train'

    if os.path.exists(train_log_dir):
        for old_file in glob.glob(train_log_dir + '/*'):
            os.remove(old_file)
    else:
        os.makedirs(train_log_dir)
    # val summary
    val_log_dir = '../tmp/tf.log/GAS/val'
    if os.path.exists(val_log_dir):
        for old_file in glob.glob(val_log_dir + '/*'):
            print ('Remove' + old_file)
            os.remove(old_file)
    else:
        os.makedirs(val_log_dir)

    summary_train_writer = tf.summary.FileWriter(train_log_dir)
    summary_val_writer = tf.summary.FileWriter(val_log_dir)

    print('Configuring model saver')
    # === Configuring model saver
    model_save_dir = '../tmp/model/GAS/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print("Saver configured")
    saver = tf.train.Saver(max_to_keep=5)

    pickle.dump(config, open(model_save_dir + 'conf.pkl', 'w'))

    # === Create session
    print('Create Session')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  # allocate dynamically
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    print("Set tf session")
    session = tf.Session(config=tf_config)
    print("Session runing..")
    session.run(tf.global_variables_initializer())
    print("Session in run")
    summary_train_writer.add_graph(session.graph)

    t1 = time.time()
    print("Check layers")
    print(tf.layers)

    print('Start training..')
    train()

    print('Training cost time: %.2f h' % ((time.time() - t1) / 3600))

    print('Start testing..')
    test()

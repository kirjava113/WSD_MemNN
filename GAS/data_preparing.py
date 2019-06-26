from utils.data import *
from utils.glove import *
import numpy as np
import math
import random


def batch_generator(is_training, batch_size, data, dict_data, pad_id, n_step_f, n_step_b, pad_last_batch=False):
    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)

    if is_training:
        random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]

        # context word
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)
        xfbs.fill(pad_id)

        # context pos
        pfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        pbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        pfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)  # 0 is pad for pos, no need pad_id

        # x forward backward
        for j in range(batch_size):
            if i * batch_size + j < data_len:
                n_to_use_f = min(n_step_f, len(batch[j].xf))
                n_to_use_b = min(n_step_b, len(batch[j].xb))
                if n_to_use_f:
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                    pfs[j, -n_to_use_f:] = batch[j].pf[-n_to_use_f:]
                if n_to_use_b:
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                    pbs[j, -n_to_use_b:] = batch[j].pb[-n_to_use_b:]
                xfbs[j] = np.concatenate((xfs[j], [batch[j].target_word_id], xbs[j][::-1]), axis=0)
                pfbs[j] = np.concatenate((pfs[j], [batch[j].target_pos_id], pbs[j][::-1]), axis=0)

        # id
        instance_ids = [inst.id for inst in batch]

        # labels
        target_ids = [inst.target_id for inst in batch]
        sense_ids = [inst.sense_id for inst in batch]

        if len(target_ids) < batch_size:  # padding
            n_pad = batch_size - len(target_ids)
            # print('Batch padding size: %d'%(n_pad))
            target_ids += [0] * n_pad
            sense_ids += [0] * n_pad
            instance_ids += [0] * n_pad  # instance_ids += [''] * n_pad

        target_ids = np.array(target_ids, dtype=np.int32)
        sense_ids = np.array(sense_ids, dtype=np.int32)

        glosses_ids = [dict_data[0][target_ids[i]] for i in range(batch_size)]
        glosses_lenth = [dict_data[1][target_ids[i]] for i in range(batch_size)]
        sense_mask = [dict_data[2][target_ids[i]] for i in range(batch_size)]

        yield (xfs, xbs, xfbs, pfs, pbs, pfbs, target_ids,
               sense_ids, instance_ids, glosses_ids, glosses_lenth, sense_mask)


def batch_generator_hyp(is_training, batch_size, data, dict_data, pad_id, n_step_f, n_step_b, n_hyper, n_hypo,
                        pad_last_batch=False):
    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)

    if is_training:
        random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]

        # context word
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)
        xfbs.fill(pad_id)

        # context pos
        pfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        pbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        pfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)  # 0 is pad for pos, no need pad_id

        # x forward backward
        for j in range(batch_size):
            if i * batch_size + j < data_len:
                n_to_use_f = min(n_step_f, len(batch[j].xf))
                n_to_use_b = min(n_step_b, len(batch[j].xb))
                if n_to_use_f:
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                    pfs[j, -n_to_use_f:] = batch[j].pf[-n_to_use_f:]
                if n_to_use_b:
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                    pbs[j, -n_to_use_b:] = batch[j].pb[-n_to_use_b:]
                xfbs[j] = np.concatenate((xfs[j], [batch[j].target_word_id], xbs[j][::-1]), axis=0)
                pfbs[j] = np.concatenate((pfs[j], [batch[j].target_pos_id], pbs[j][::-1]), axis=0)

        # id
        instance_ids = [inst.id for inst in batch]

        # labels
        target_ids = [inst.target_id for inst in batch]
        sense_ids = [inst.sense_id for inst in batch]

        if len(target_ids) < batch_size:  # padding
            n_pad = batch_size - len(target_ids)
            # print('Batch padding size: %d'%(n_pad))
            target_ids += [0] * n_pad
            sense_ids += [0] * n_pad
            instance_ids += [0] * n_pad  # instance_ids += [''] * n_pad

        target_ids = np.array(target_ids, dtype=np.int32)
        sense_ids = np.array(sense_ids, dtype=np.int32)

        # [gloss_to_id, gloss_lenth, sense_mask, hyper_lenth, hypo_lenth]
        glosses_ids = [dict_data[0][target_ids[i]] for i in
                       range(batch_size)]  # [batch_size, max_n_sense, n_hyp, max_gloss_words]
        glosses_lenth = [dict_data[1][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]
        sense_mask = [dict_data[2][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense, mask_size]
        hyper_lenth = [dict_data[3][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]
        hypo_lenth = [dict_data[4][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]

        yield (xfs, xbs, xfbs, pfs, pbs, pfbs, target_ids, sense_ids, instance_ids, glosses_ids,
               glosses_lenth, sense_mask, hyper_lenth, hypo_lenth)


def data_loading(_path, config):
    print('Loading all-words task data...')
    train_dataset = _path.ALL_WORDS_TRAIN_DATASET[0]  # semcor
    print('train_dataset: ' + train_dataset)
    val_dataset = _path.ALL_WORDS_VAL_DATASET  # semeval2007
    print('val_dataset: ' + val_dataset)
    test_dataset = _path.ALL_WORDS_TEST_DATASET[0]  # ALL
    print('test_dataset: ' + test_dataset)

    train_data = load_train_data(train_dataset)
    val_data = load_val_data(val_dataset)
    test_data = load_test_data(test_dataset)
    print('O Original dataset size (train/val/test): %d / %d / %d' % (len(train_data), len(val_data), len(test_data)))

    # === Using back-off strategy
    back_off_result = []
    if train_dataset in _path.ALL_WORDS_TRAIN_DATASET:
        test_data_lenth_pre = len(test_data)
        train_data, test_data, word_to_senses, back_off_result = data_postprocessing(
            train_dataset, test_dataset, train_data, test_data, 'FS', config.min_sense_freq, config.max_n_sense)
        val_data = data_postprocessing_for_validation(val_data, word_to_senses)
        print ('1 Filtered dataset size (train/val/test): %d / %d / %d' % (
            len(train_data), len(val_data), len(test_data)))
        print ('***Test using back-off instance: %d' % (len(back_off_result)))
        missed = test_data_lenth_pre - (len(test_data) + len(back_off_result))
        missed_ratio = float(missed) / test_data_lenth_pre
        print ('***Test missed instance(not in MFS/FS): %d/%d = %.3f' % (
            (missed, test_data_lenth_pre, missed_ratio)))

    # === Build vocab utils
    word_to_id = build_vocab(train_data)
    config.vocab_size = len(word_to_id)
    target_word_to_id, target_sense_to_id, n_senses_from_target_id, word_to_sense = build_sense_ids(word_to_senses)
    print ('Vocabulary size: %d' % len(word_to_id))
    tot_n_senses = sum(n_senses_from_target_id.values())
    average_sense = float(tot_n_senses) / len(n_senses_from_target_id)
    assert average_sense >= 2.0  # ambiguous word must have two sense
    print ('Avg n senses per target word: %.4f' % average_sense)
    with open('../tmp/target_word.txt', 'w') as f:
        for word, id in target_word_to_id.items():
            f.write('{}\t{}\n'.format(word, id))

    # === Make numeric
    train_data = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Train')
    val_data = convert_to_numeric(val_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Val')
    test_data = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Test')
    print ('2 After convert_to_numeric dataset size (train/val/test): %d / %d / %d' % (
        len(train_data), len(val_data), len(test_data)))

    target_id_to_word = {id: word for (word, id) in target_word_to_id.iteritems()}
    target_id_to_sense_id_to_sense = [{sense_id: sense for (sense, sense_id) in sense_to_id.iteritems()} for
                                      (target_id, sense_to_id) in enumerate(target_sense_to_id)]

    # get dic and make numeric
    gloss_dict = load_dictionary(train_dataset, target_word_to_id, config.gloss_expand_type)
    mask_size = 2 * config.n_lstm_units
    dict_data, config.max_n_sense = bulid_dictionary_id(gloss_dict, target_sense_to_id, word_to_id, word_to_id['<pad>'],
                                                        mask_size, config.max_gloss_words)

    if config.use_pre_trained_embedding:
        print('Load pre-trained word embedding...')
        # TODO method for changing word_to_id to glove
        init_emb = fill_with_gloves(word_to_id, _path.GLOVE_VECTOR)
    else:
        init_emb = None

    return n_senses_from_target_id, init_emb, test_data, back_off_result, target_id_to_sense_id_to_sense, word_to_id, \
           dict_data, train_data, val_data

from typing import Type
import time
import os
import logging
import tensorflow.compat.v1 as tf

import sys
print(sys.path)
from ..newData.sparsegraph import load_data
from .model import Model
from .earlystopping import EarlyStopping, stopping_args


def train_model(
        dataName: str, model_class: Type[Model], build_args: dict,
        stopping_args: dict = stopping_args, print_interval: int = 20) -> dict:

    adj, features, labels, train_idx, val_idx, test_idx, num_features, num_labels = load_data(dataName, 0.6, 0.2)
    tf.reset_default_graph()

    sess = tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    model = model_class(features, adj, labels, sess)
    model.build_model(**build_args)

    train_inputs = {
            model.idx: train_idx,
            model.isTrain: True}
    train_inference_inputs = {
            model.idx: train_idx,
            model.isTrain: False}
    stopping_inputs = {
            model.idx: val_idx,
            model.isTrain: False}
    valtest_inputs = {
            model.idx: test_idx,
            model.isTrain: False}

    init = tf.global_variables_initializer()
    sess.run(init)


    early_stopping = EarlyStopping(model, **stopping_args)

    start_time = time.time()
    last_time = start_time
    for step in range(early_stopping.max_steps):

        _, train_loss = sess.run(
            [model.train_op, model.loss],
            feed_dict=train_inputs)

        train_acc, train_str = sess.run(
            [model.accuracy, model.summary], feed_dict=train_inference_inputs)
        stopping_loss, stopping_acc, stopping_str = sess.run(
            [model.loss, model.accuracy, model.summary],
            feed_dict=stopping_inputs)
        if step % print_interval == 0:
            duration = time.time() - last_time
            last_time = time.time()
            logging.info(
                "Step {}: Train loss = {:.2f}, train acc = {:.1f}, "
                "early stopping loss = {:.2f}, early stopping acc = {:.1f} ({:.3f} sec)"
                    .format(step, train_loss, train_acc * 100,
                            stopping_loss, stopping_acc * 100, duration))
        if len(early_stopping.stop_vars) > 0:
            stop_vars = sess.run(
                early_stopping.stop_vars, feed_dict=stopping_inputs)
            if early_stopping.check(stop_vars, step):
                break
    runtime = time.time() - start_time
    runtime_perepoch = runtime / (step + 1)

    if len(early_stopping.stop_vars) == 0:
        logging.log(22, "Last step: {} ({:.3f} sec)".format(step, runtime))
    else:
        logging.log(22, "Last step: {}, best step: {} ({:.3f} sec)"
                    .format(step, early_stopping.best_step, runtime))
        model.set_vars(early_stopping.best_trainables)

    train_accuracy, train_f1_score = sess.run(
            [model.accuracy, model.f1_score],
            feed_dict=train_inputs)

    stopping_accuracy, stopping_f1_score = sess.run(
            [model.accuracy, model.f1_score],
            feed_dict=stopping_inputs)
    logging.log(21, "Early stopping accuracy: {:.1f}%, early stopping F1 score: {:.3f}"
                .format(stopping_accuracy * 100, stopping_f1_score))

    valtest_accuracy, valtest_f1_score = sess.run(
            [model.accuracy, model.f1_score],
            feed_dict=valtest_inputs)

    valtest_name = 'Test'
    logging.log(22, "{} accuracy: {:.1f}%, test F1 score: {:.3f}"
                .format(valtest_name, valtest_accuracy * 100, valtest_f1_score))


    result = {}
    result['predictions'] = model.get_predictions()
    result['vars'] = early_stopping.best_trainables
    result['train'] = {'accuracy': train_accuracy, 'f1_score': train_f1_score}
    result['early_stopping'] = {'accuracy': stopping_accuracy, 'f1_score': stopping_f1_score}
    result['valtest'] = {'accuracy': valtest_accuracy, 'f1_score': valtest_f1_score}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime_perepoch
    sess.close()
    return result

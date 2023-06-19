import json
import os
import time
import pickle
import random

from absl import flags
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from etcmodel.models import input_utils, modeling
from etcmodel.models.docred import run_docred_lib

tf.compat.v1.disable_v2_behavior()
tf.enable_eager_execution()

flags = tf.flags
FLAGS = flags.FLAGS

GLOBAL_SIZE=80
NUM_ENTITIES=50
NUM_LABELS=96
BATCH_SIZE=1

#predictions have one extra, the threshold, those predicted above it are predicted positive, below it, negative
def adaptiveFocalLoss(y_true,y_pred):
    gamma=0.5
    threshold=y_pred[:,:,:,NUM_LABELS:]
    thresholds=[]
    for i in range(NUM_LABELS):
        thresholds.append(threshold)
    thresholds=tf.cast(tf.concat(thresholds,axis=-1),"float32")
    y_true=tf.cast(y_true,"float32")
    y_pred=y_pred[:,:,:,:NUM_LABELS]
    positive_log_probs=tf.math.log_sigmoid(tf.subtract(y_pred,thresholds))
    y_pred_no_positive=tf.subtract(y_pred,1000*y_true)
    threshold_log_prob=tf.math.log_sigmoid(-1*tf.math.reduce_logsumexp(tf.subtract(y_pred_no_positive,thresholds),axis=-1,keepdims=True))
    gamma_term=tf.math.pow(tf.subtract(tf.ones(tf.shape(y_pred),"float32"),tf.math.exp(positive_log_probs)),gamma*tf.ones(tf.shape(y_pred),"float32"))
    #Is gamma_term causing the NaN issue?
    final=tf.multiply(y_true,tf.multiply(positive_log_probs,gamma_term))
    loss=-1*tf.add(tf.reduce_sum(final),tf.reduce_sum(threshold_log_prob))
    print(y_true)
    print(y_pred)
    print(threshold)
    print(thresholds)
    #print(positive_class)
    print(positive_log_probs)
    print(y_pred_no_positive)
    print(threshold_log_prob)
    #print(thresholds_log_probs)
    print(gamma_term)
    print(final)
    print(loss)
    return loss
#precision and recall for adaptive focal loss
class AFLPrecision(tf.keras.metrics.Metric):
    def __init__(self,name="AFLprecision",**kwargs):
        super().__init__(name=name,**kwargs)
        self.metric=tf.keras.metrics.Precision()
    def update_state(self,y_true,y_pred,sample_weight=None):
        threshold=y_pred[:,:,:,NUM_LABELS:]
        thresholds=[]
        for i in range(NUM_LABELS):
            thresholds.append(threshold)
        thresholds=tf.cast(tf.concat(thresholds,axis=-1),"float32")
        y_pred=y_pred[:,:,:,:NUM_LABELS]
        y_pred=tf.math.greater(tf.cast(y_pred,"float32"),tf.cast(thresholds,"float32"))
        self.metric.update_state(y_true,y_pred,sample_weight)
        print(y_true)
        print(y_pred)
    def result(self):
        return self.metric.result()
    def reset_state(self):
        self.metric.reset_state()
class AFLRecall(tf.keras.metrics.Metric):
    def __init__(self,name="AFLrecall",**kwargs):
        super().__init__(name=name,**kwargs)
        self.metric=tf.keras.metrics.Recall()
    def update_state(self,y_true,y_pred,sample_weight=None):
        threshold=y_pred[:,:,:,NUM_LABELS:]
        thresholds=[]
        for i in range(NUM_LABELS):
            thresholds.append(threshold)
        thresholds=tf.cast(tf.concat(thresholds,axis=-1),"float32")
        y_pred=y_pred[:,:,:,:NUM_LABELS]
        y_pred=tf.math.greater(tf.cast(y_pred,"float32"),tf.cast(thresholds,"float32"))
        self.metric.update_state(y_true,y_pred,sample_weight)
        print(y_true)
        print(y_pred)
    def result(self):
        return self.metric.result()
    def reset_state(self):
        self.metric.reset_state()

y_true=[]
y_pred=[]
for i in range(NUM_LABELS):
    y_true.append(0)
    y_pred.append(random.random()*20-10)
y_pred.append(random.random()*20-10)
y_true[10]=1
y_true[57]=1
y_true[85]=1
y_true=tf.reshape(tf.constant(y_true),[1,1,1,NUM_LABELS])
y_pred=tf.reshape(tf.constant(y_pred),[1,1,1,NUM_LABELS+1])
adaptiveFocalLoss(y_true,y_pred)

import json
import os
import time
import pickle
import sys

from absl import flags
import numpy as np
import pandas as pd

#from tensorflow.compat.v1 import estimator as tf_estimator

#import tensorflow.python.tools.inspect_checkpoint
from tensorflow.python.training import py_checkpoint_reader

from etcmodel.models import input_utils, modeling
from etcmodel.models.docred import run_docred_lib

#tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Required parameters
flags.DEFINE_string('TOKENIZER',"BERT","Which tokenizer we're using")
flags.DEFINE_boolean('ANNOTATE',False,"Are we annotating the input?")
flags.DEFINE_boolean('FULL',False,"Are we including all non-leaf constit tokens, at the cost of removing wsd tokens?")
flags.DEFINE_boolean('TRUEFULL',False,"Are we adding the WSD tokens back in for the true full annotation?")
flags.DEFINE_string('IGN_PATH',"./ign_mask.pickle","Path to tensor of relations to ignore in devset")
flags.DEFINE_integer('GLOBAL_SIZE',80,"Size of non-ONF global input")
flags.DEFINE_integer('NUM_ENTITIES',50,"Number of entities in global input")
flags.DEFINE_integer('NUM_LABELS',96,"Number of possible relation labels")
flags.DEFINE_integer('BATCH_SIZE',5,"Size of batches during training")
flags.DEFINE_integer('ONE_LABEL',-1,"Which label to focus solely on(-1 if we're not doing this)")
flags.DEFINE_integer('NUM_HIDDEN_LAYERS',12,"Number of ETC layers")
flags.DEFINE_float('LEARNING_RATE',0.0001,"Learning rate")
flags.DEFINE_integer('NUM_EPOCHS',10,"Number of training epochs")
flags.DEFINE_float('ZERO_WEIGHT',0.25,"Weight of negative examples")
flags.DEFINE_boolean('DO_MASK',True,"Remove the masks if this is false, for testing")
flags.DEFINE_string('LOSS_FUNC',"BCE","Loss function: BCE,AFL,or F1")
flags.DEFINE_integer('NUM_FINAL_LAYERS',5,"Number of final Dense ReLU layers")
flags.DEFINE_float('GAMMA',1.0,"Gamma hyperparam for AFL")
flags.DEFINE_string('PRETRAIN_FILE',"BERTmymodels/pretrainedmodel","Pretrained model to load from")
flags.DEFINE_boolean('PRETRAIN',True,"Whether to pretrain")
flags.DEFINE_integer('RUN_NUMBER',-1,"Use for statistical runs, so that the results are printed and distinguished, but the models are not saved")

flags.DEFINE_boolean('CONTINUE',False,"Contine from saved model with current parameters?")
flags.DEFINE_string('OPTIMIZER',"Adam","Use Adam or AdamW?")

FLAGS=flags.FLAGS
FLAGS(sys.argv)

if FLAGS.OPTIMIZER=="AdamW":
    import tensorflow as tf
elif FLAGS.OPTIMIZER=="Adam":
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
    tf.enable_eager_execution()
else:
    print("Choose Adam or AdamW for optimizer!")
    exit()

TOKENIZER=FLAGS.TOKENIZER#BERT, ALBERT, or ROBERTA-BASE
ANNOTATE=FLAGS.ANNOTATE
FULL=FLAGS.FULL
TRUEFULL=FLAGS.TRUEFULL
if not ANNOTATE:
    INPUT_TRAIN_PATH="./"+TOKENIZER+"examples/tf_example"
    INPUT_VAL_PATH="./"+TOKENIZER+"devexamples/tf_example"
elif ANNOTATE:
    if not FULL:
        INPUT_TRAIN_PATH="./"+TOKENIZER+"annoexamples/tf_example"
        INPUT_VAL_PATH="./"+TOKENIZER+"annodevexamples/tf_example"
    elif FULL and not TRUEFULL:
        INPUT_TRAIN_PATH="./"+TOKENIZER+"constitfullexamples/tf_example"
        INPUT_VAL_PATH="./"+TOKENIZER+"constitfulldevexamples/tf_example"
    elif TRUEFULL:
        INPUT_TRAIN_PATH="./"+TOKENIZER+"truefullexamples/tf_example"
        INPUT_VAL_PATH="./"+TOKENIZER+"truefulldevexamples/tf_example"
IGN_PATH=FLAGS.IGN_PATH
GLOBAL_SIZE=FLAGS.GLOBAL_SIZE#Only for the non-onf portion of global size
NUM_ENTITIES=FLAGS.NUM_ENTITIES
NUM_LABELS=FLAGS.NUM_LABELS
BATCH_SIZE=FLAGS.BATCH_SIZE
ONE_LABEL=FLAGS.ONE_LABEL
NUM_HIDDEN_LAYERS=FLAGS.NUM_HIDDEN_LAYERS
LEARNING_RATE=FLAGS.LEARNING_RATE
NUM_EPOCHS=FLAGS.NUM_EPOCHS
ZERO_WEIGHT=FLAGS.ZERO_WEIGHT
DO_MASK=FLAGS.DO_MASK
LOSS_FUNC=FLAGS.LOSS_FUNC#BCE, AFL, or F1
NUM_FINAL_LAYERS=FLAGS.NUM_FINAL_LAYERS
GAMMA=FLAGS.GAMMA
OPTIMIZER=FLAGS.OPTIMIZER
CONTINUE=FLAGS.CONTINUE
if TOKENIZER=="BERT":
    CONFIG_DIR="./etcmodel/pretrained/etc_base/"
else:
    CONFIG_DIR="./etcmodel/pretrained/"
#PRETRAIN_FILE=CONFIG_DIR+"model.ckpt"
PRETRAIN_FILE=FLAGS.PRETRAIN_FILE
PRETRAIN=FLAGS.PRETRAIN

MODEL_FILENAME="./"+TOKENIZER+"mymodels/"
if ANNOTATE:
    if TRUEFULL:
        MODEL_FILENAME+="TRUEFULL"
    else:
        MODEL_FILENAME+="ANNO"
        if FULL:
            MODEL_FILENAME+="FULL"
MODEL_FILENAME+=str(NUM_HIDDEN_LAYERS)+"L"+str(BATCH_SIZE)+"B"+str(NUM_EPOCHS)+"E"+str(LEARNING_RATE)+"LR"
if LOSS_FUNC=="AFL":
    MODEL_FILENAME+=str(GAMMA)+"gAFL"
elif LOSS_FUNC=="BCE":
    MODEL_FILENAME+=str(ZERO_WEIGHT)+"BCE"
elif LOSS_FUNC=="F1":
    MODEL_FILENAME+="F1"
if NUM_FINAL_LAYERS>1:
    MODEL_FILENAME+=str(NUM_FINAL_LAYERS)+"Fin"
if PRETRAIN:
    MODEL_FILENAME+="Pre"
if OPTIMIZER=="AdamW":
    MODEL_FILENAME+="AdamW"
if FLAGS.RUN_NUMBER>=0:
    MODEL_FILENAME+=str(FLAGS.RUN_NUMBER)
print(MODEL_FILENAME)

model_config=input_utils.get_model_config(
        model_dir=CONFIG_DIR,
#        source_file=SOURCE_MODEL_CONFIG_FILE,
        write_from_source=False)

#Define the model we're using
class MyModel(tf.keras.Model):
    def __init__(self,model_config,ckpt=None):
        super().__init__()
        if ckpt:
            self.etcmodel=tf.keras.models.load_model(ckpt)
        else:
            self.etcmodel=modeling.EtcModel(model_config)
        self.final_layers=[]
        if NUM_FINAL_LAYERS>1:
            for i in range(1,NUM_FINAL_LAYERS):
                self.final_layers.append(tf.keras.layers.Dense(units=model_config.hidden_size*2,activation=tf.keras.activations.relu))
        if LOSS_FUNC=="AFL":
            self.final_layer=tf.keras.layers.Dense(units=NUM_LABELS+1)#96 relations and threshold(for Adaptive Focal Loss
        elif ONE_LABEL==-1:
            self.final_layer=tf.keras.layers.Dense(units=NUM_LABELS)
        else:
            self.final_layer=tf.keras.layers.Dense(units=1)
    def call(self,inputs,training=False):
        _,global_output=self.etcmodel(**inputs,training=training)
        entities=global_output[:,GLOBAL_SIZE-NUM_ENTITIES:GLOBAL_SIZE,:]
        all_pairs_shape=tf.stack([tf.shape(entities)[0],tf.shape(entities)[1],tf.shape(entities)[1],tf.shape(entities)[2]],axis=0)
        global_broadcast1=tf.broadcast_to(entities[:,:,tf.newaxis,:],all_pairs_shape)
        global_broadcast2=tf.broadcast_to(entities[:,tf.newaxis,:,:],all_pairs_shape)
        pairs_embedding=tf.concat([global_broadcast1,global_broadcast2],axis=-1)
        for i in range(1,NUM_FINAL_LAYERS):
            pairs_embedding=self.final_layers[i-1](pairs_embedding,training=training)
        result=self.final_layer(pairs_embedding,training=training)
        return result#return as logits in this case

def oneRelationLoss(y_true,y_pred):
    y_true=y_true[:,:,:,1]
    y_pred=y_pred[:,:,:,1]
    zeroweight=1/1000.0
    is_zero=tf.equal(tf.cast(y_true,"int32"),tf.zeros(tf.shape(y_true),"int32"))
    weights=tf.add(tf.ones(tf.shape(y_true),"float32"),(zeroweight-1.0)*tf.cast(is_zero,"float32"))
    cross_entropy=tf.add(-1*tf.math.log_sigmoid(y_pred),tf.multiply(y_pred,tf.cast(is_zero,"float32")))
    #probs=tf.math.sigmoid(y_pred)
    #cross_entropy=-1*tf.add(tf.multiply(tf.cast(y_true,"float32"),tf.math.log(probs)),tf.multiply(tf.cast(is_zero,"float32"),tf.math.log(tf.add(tf.ones(tf.shape(probs),"float32"),-1*probs))))
    loss=tf.reduce_sum(tf.multiply(weights,cross_entropy))
    print(y_true)
    print(y_pred)
    print(is_zero)
    print(weights)
    print(cross_entropy)
    print(loss)
    return loss
def oneThresholdPrecision(threshold):
    class thresholdMetric(tf.keras.metrics.Metric):
        def __init__(self,name=str(threshold)+"precision",**kwargs):
            super().__init__(name=name,**kwargs)
            self.metric=tf.keras.metrics.Precision()
        def update_state(self,y_true,y_pred,sample_weight=None):
            y_true=y_true[:,:,:,1]
            y_pred=y_pred[:,:,:,1]
            y_pred=tf.math.greater(tf.cast(y_pred,"float32"),threshold*tf.ones(tf.shape(y_pred),"float32"))
            y_pred=tf.cast(y_pred,"int32")
            self.metric.update_state(y_true,y_pred,sample_weight)
            print(y_true)
            print(y_pred)
        def result(self):
            return self.metric.result()
        def reset_state(self):
            self.metric.reset_state()
    return thresholdMetric
def oneThresholdRecall(threshold):
    class thresholdMetric(tf.keras.metrics.Metric):
        def __init__(self,name=str(threshold)+"recall",**kwargs):
            super().__init__(name=name,**kwargs)
            self.metric=tf.keras.metrics.Recall()
        def update_state(self,y_true,y_pred,sample_weight=None):
            y_true=y_true[:,:,:,1]
            y_pred=y_pred[:,:,:,1]
            y_pred=tf.math.greater(tf.cast(y_pred,"float32"),threshold*tf.ones(tf.shape(y_pred),"float32"))
            y_pred=tf.cast(y_pred,"int32")
            self.metric.update_state(y_true,y_pred,sample_weight)
            print(y_true)
            print(y_pred)
        def result(self):
            return self.metric.result()
        def reset_state(self):
            self.metric.reset_state()
    return thresholdMetric



def customWeightedCategoricalCrossentropy(y_true,y_pred):
    zeroweight=1/500.0
    y_true=tf.squeeze(y_true,axis=-1)
    is_zero=tf.equal(tf.cast(y_true,"int32"),tf.zeros(tf.shape(y_true),"int32"))
    weights=tf.add(tf.ones(tf.shape(y_true),"float32"),(zeroweight-1.0)*tf.cast(is_zero,"float32"))
    cross_entropy=-1*tf.multiply(tf.one_hot(y_true,depth=98),tf.math.log(y_pred))
    cross_entropy=tf.reduce_sum(cross_entropy,axis=-1)
    loss=tf.reduce_sum(tf.multiply(weights,cross_entropy))
    print(y_true)
    print(y_pred)
    print(is_zero)
    print(weights)
    print(cross_entropy)
    print(loss)
    return loss

#Calculate cross entropy: take negative log-sigmoid of predictions, add prediction logits for zeros
#a cool math trick to avoid rounding errors
def customWeightedBinaryCrossentropy(y_true,y_pred):
    y_true=tf.cast(y_true,"int32")
    y_pred=tf.cast(y_pred,"float32")
    zeroweight=ZERO_WEIGHT
    is_zero=tf.equal(y_true,tf.zeros(tf.shape(y_true),"int32"))
    weights=tf.add(tf.ones(tf.shape(y_true),"float32"),(zeroweight-1.0)*tf.cast(is_zero,"float32"))
    cross_entropy=tf.add(-1*tf.math.log_sigmoid(y_pred),tf.multiply(y_pred,tf.cast(is_zero,"float32")))
    #probs=tf.math.sigmoid(y_pred)
    #cross_entropy=-1*tf.add(tf.multiply(tf.cast(y_true,"float32"),tf.math.log(probs)),tf.multiply(tf.cast(is_zero,"float32"),tf.math.log(tf.add(tf.ones(tf.shape(probs),"float32"),-1*probs))))
    loss=tf.reduce_sum(tf.multiply(weights,cross_entropy))
    print(y_true)
    print(y_pred)
    print(is_zero)
    print(weights)
    print(cross_entropy)
    print(loss)
    return loss

#predictions have one extra, the threshold, those predicted above it are predicted positive, below it, negative
def adaptiveFocalLoss(y_true,y_pred):
    gamma=GAMMA
    threshold_log_prob_weight=1.0
    threshold=y_pred[:,:,:,NUM_LABELS:]
    thresholds=[]
    for i in range(NUM_LABELS):
        thresholds.append(threshold)
    thresholds=tf.cast(tf.concat(thresholds,axis=-1),"float32")
    y_true=tf.cast(y_true,"float32")
    y_pred=y_pred[:,:,:,:NUM_LABELS]
    positive_log_probs=tf.math.log_sigmoid(tf.subtract(y_pred,thresholds))
    y_pred_no_positive=tf.subtract(y_pred,1000.0*y_true)
    threshold_log_prob=tf.math.log_sigmoid(-1.0*tf.math.reduce_logsumexp(tf.subtract(y_pred_no_positive,thresholds),axis=-1,keepdims=True))
    gamma_term=tf.math.pow(tf.subtract(tf.ones(tf.shape(y_pred),"float32"),tf.math.exp(positive_log_probs)),gamma*tf.ones(tf.shape(y_pred),"float32"))
    #Is gamma_term causing the NaN issue?
#    final=tf.multiply(y_true,positive_log_probs)
    final=tf.multiply(y_true,tf.multiply(positive_log_probs,gamma_term))
    loss=-1*tf.add(tf.reduce_sum(final),threshold_log_prob_weight*tf.reduce_sum(threshold_log_prob))
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

#try loss function based on the F1 metric?
def softF1Loss(y_true,y_pred):
    y_true=tf.cast(y_true,"float32")
    y_probs=tf.math.sigmoid(tf.cast(y_pred,"float32"))
    y_negative_probs=tf.math.subtract(tf.ones(tf.shape(y_probs),"float32"),y_probs)
    y_negative=tf.math.subtract(tf.ones(tf.shape(y_true),"float32"),y_true)
    true_positives=tf.reduce_sum(tf.multiply(y_true,y_probs))
    false_positives=tf.reduce_sum(tf.multiply(y_negative,y_probs))
    false_negatives=tf.reduce_sum(tf.multiply(y_true,y_negative_probs))
    print(y_true)
    print(y_probs)
    print(y_negative_probs)
    print(y_negative)
    print(true_positives)
    print(false_positives)
    print(false_negatives)
    return (false_positives+false_negatives)/(2*true_positives+false_positives+false_negatives+0.0001)

#precision and recall for adaptive focal loss
class AFLPrecision(tf.keras.metrics.Metric):
    def __init__(self,name="AFLprecision",ign=False,**kwargs):
        super().__init__(name=name,**kwargs)
        self.metric=tf.keras.metrics.Precision()
        self.ign=ign
    def update_state(self,y_true,y_pred,sample_weight=None):
        threshold=y_pred[:,:,:,NUM_LABELS:]
        thresholds=[]
        for i in range(NUM_LABELS):
            thresholds.append(threshold)
        thresholds=tf.cast(tf.concat(thresholds,axis=-1),"float32")
        y_pred=y_pred[:,:,:,:NUM_LABELS]
        y_pred=tf.math.greater(tf.cast(y_pred,"float32"),tf.cast(thresholds,"float32"))
        if self.ign:
            sample_weight=tf.greater(y_true,-1*tf.ones(tf.shape(y_true),"int32"))
            sample_weight=tf.cast(sample_weight,"int32")
        else:
            y_true=tf.add(y_true,2*tf.cast(tf.greater(tf.zeros(tf.shape(y_true),"int32"),y_true),"int32"))
        self.metric.update_state(y_true,y_pred,sample_weight)
        print(y_true)
        print(y_pred)
    def result(self):
        return self.metric.result()
    def reset_state(self):
        self.metric.reset_state()
class AFLRecall(tf.keras.metrics.Metric):
    def __init__(self,name="AFLrecall",ign=False,**kwargs):
        super().__init__(name=name,**kwargs)
        self.metric=tf.keras.metrics.Recall()
        self.ign=ign
    def update_state(self,y_true,y_pred,sample_weight=None):
        threshold=y_pred[:,:,:,NUM_LABELS:]
        thresholds=[]
        for i in range(NUM_LABELS):
            thresholds.append(threshold)
        thresholds=tf.cast(tf.concat(thresholds,axis=-1),"float32")
        y_pred=y_pred[:,:,:,:NUM_LABELS]
        y_pred=tf.math.greater(tf.cast(y_pred,"float32"),tf.cast(thresholds,"float32"))
        if self.ign:
            sample_weight=tf.greater(y_true,-1*tf.ones(tf.shape(y_true),"int32"))
            sample_weight=tf.cast(sample_weight,"int32")
        else:
            y_true=tf.add(y_true,2*tf.cast(tf.greater(tf.zeros(tf.shape(y_true),"int32"),y_true),"int32"))
        self.metric.update_state(y_true,y_pred,sample_weight)
        print(y_true)
        print(y_pred)
    def result(self):
        return self.metric.result()
    def reset_state(self):
        self.metric.reset_state()

class MyPrecision(tf.keras.metrics.Metric):
    def __init__(self,name="precision",**kwargs):
        super().__init__(name=name,**kwargs)
        self.true_positives=self.add_weight(name="precisionTruePositives",initializer="zeros")
        self.total_positives=self.add_weight(name="precisionPositives",initializer="zeros")
#        self.metric=tf.keras.metrics.Precision()
    def update_state(self,y_true,y_pred,sample_weight=None):
        #get prediction from logits
        y_true=tf.squeeze(y_true,axis=-1)
        y_pred=tf.argmax(y_pred,axis=-1)
#        match=tf.equal(tf.cast(y_true,"int32"),tf.cast(y_pred,"int32"))
#        actual_positive=tf.math.greater(tf.cast(y_true,"int32"),tf.zeros(tf.shape(y_true),"int32"))
#        true_predicted_positive=tf.math.logical_and(match,actual_positive)
#        self.metric.update_state(tf.cast(actual_positive,"int32"),tf.cast(true_predicted_positive,"int32"))
        match=tf.equal(tf.cast(y_true,"int32"),tf.cast(y_pred,"int32"))
        positive=tf.math.greater(tf.cast(y_pred,"float32"),tf.zeros(tf.shape(y_pred),"float32"))
        true_positive=tf.math.logical_and(match,positive)
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_positive,"float32")))
        self.total_positives.assign_add(tf.reduce_sum(tf.cast(positive,"float32")))
        print(y_true)
        print(y_pred)
        print(match)
        print(positive)
        print(true_positive)
    def result(self):
        result=tf.math.divide_no_nan(self.true_positives,self.total_positives)
        return result
#        return self.metric.result()
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.total_positives.assign(0.0)
#        self.metric.reset_state()
class MyRecall(tf.keras.metrics.Metric):
    def __init__(self,name="recall",**kwargs):
        super().__init__(name=name,**kwargs)
        self.true_positives=self.add_weight(name="recallTruePositives",initializer="zeros")
        self.total_positives=self.add_weight(name="recallPositives",initializer="zeros")
#        self.metric=tf.keras.metrics.Recall()
    def update_state(self,y_true,y_pred,sample_weight=None):
        #get prediction from logits
        y_true=tf.squeeze(y_true,axis=-1)
        y_pred=tf.argmax(y_pred,axis=-1)
#        match=tf.equal(tf.cast(y_true,"int32"),tf.cast(y_pred,"int32"))
#        actual_positive=tf.math.greater(tf.cast(y_true,"int32"),tf.zeros(tf.shape(y_true),"int32"))
#        true_predicted_positive=tf.math.logical_and(match,actual_positive)
#        self.metric.update_state(tf.cast(actual_positive,"int32"),tf.cast(true_predicted_positive,"int32"))
        match=tf.equal(tf.cast(y_true,"int32"),tf.cast(y_pred,"int32"))
        positive=tf.math.greater(tf.cast(y_true,"float32"),tf.zeros(tf.shape(y_true),"float32"))
        true_positive=tf.math.logical_and(match,positive)
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_positive,"float32")))
        self.total_positives.assign_add(tf.reduce_sum(tf.cast(positive,"float32")))
        print(y_true)
        print(y_pred)
        print(match)
        print(positive)
        print(true_positive)
    def result(self):
        result=tf.math.divide_no_nan(self.true_positives,self.total_positives)
        return result
#        return self.metric.result()
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.total_positives.assign(0.0)
#        self.metric.reset_state()
#read in features and labels from input files

def ThresholdPrecision(threshold):
    class thresholdMetric(tf.keras.metrics.Metric):
        def __init__(self,name=str(threshold)+"precision",**kwargs):
            super().__init__(name=name,**kwargs)
            self.metric=tf.keras.metrics.Precision()
        def update_state(self,y_true,y_pred,sample_weight=None):
            y_pred=tf.math.greater(tf.cast(y_pred,"float32"),threshold*tf.ones(tf.shape(y_pred),"float32"))
            y_pred=tf.cast(y_pred,"int32")
            self.metric.update_state(y_true,y_pred,sample_weight)
#            print(y_true)
#            print(y_pred)
        def result(self):
            return self.metric.result()
        def reset_state(self):
            self.metric.reset_state()
    return thresholdMetric
def ThresholdRecall(threshold):
    class thresholdMetric(tf.keras.metrics.Metric):
        def __init__(self,name=str(threshold)+"recall",**kwargs):
            super().__init__(name=name,**kwargs)
            self.metric=tf.keras.metrics.Recall()
        def update_state(self,y_true,y_pred,sample_weight=None):
            y_pred=tf.math.greater(tf.cast(y_pred,"float32"),threshold*tf.ones(tf.shape(y_pred),"float32"))
            y_pred=tf.cast(y_pred,"int32")
            self.metric.update_state(y_true,y_pred,sample_weight)
#            print(y_true)
#            print(y_pred)
        def result(self):
            return self.metric.result()
        def reset_state(self):
            self.metric.reset_state()
    return thresholdMetric

def IGNPrecision(threshold,ign):
    class thresholdMetric(tf.keras.metrics.Metric):
        def __init__(self,name=str(threshold)+"precision",**kwargs):
            super().__init__(name=name,**kwargs)
            self.metric=tf.keras.metrics.Precision()
        def update_state(self,y_true,y_pred,sample_weight=None):
            y_pred=tf.math.greater(tf.cast(y_pred,"float32"),threshold*tf.ones(tf.shape(y_pred),"float32"))
            y_pred=tf.cast(y_pred,"int32")
            if ign:
                sample_weight=tf.greater(y_true,-1*tf.ones(tf.shape(y_true),"int32"))
                sample_weight=tf.cast(sample_weight,"int32")
            else:
                y_true=tf.add(y_true,2*tf.cast(tf.greater(tf.zeros(tf.shape(y_true),"int32"),y_true),"int32"))
            self.metric.update_state(y_true,y_pred,sample_weight)
#            print(y_true)
#            print(y_pred)
        def result(self):
            return self.metric.result()
        def reset_state(self):
            self.metric.reset_state()
    return thresholdMetric
def IGNRecall(threshold,ign):
    class thresholdMetric(tf.keras.metrics.Metric):
        def __init__(self,name=str(threshold)+"recall",**kwargs):
            super().__init__(name=name,**kwargs)
            self.metric=tf.keras.metrics.Recall()
        def update_state(self,y_true,y_pred,sample_weight=None):
            y_pred=tf.math.greater(tf.cast(y_pred,"float32"),threshold*tf.ones(tf.shape(y_pred),"float32"))
            y_pred=tf.cast(y_pred,"int32")
            if ign:
                sample_weight=tf.greater(y_true,-1*tf.ones(tf.shape(y_true),"int32"))
                sample_weight=tf.cast(sample_weight,"int32")
            else:
                y_true=tf.add(y_true,2*tf.cast(tf.greater(tf.zeros(tf.shape(y_true),"int32"),y_true),"int32"))
            self.metric.update_state(y_true,y_pred,sample_weight)
#            print(y_true)
#            print(y_pred)
        def result(self):
            return self.metric.result()
        def reset_state(self):
            self.metric.reset_state()
    return thresholdMetric



def get_input_data(input_file_pattern,ign_file=None):
    list_features=[]
    list_labels=[]
    counter=0
    #For validation, add the ign_mask to the labels so we know which labels to ignore
    if ign_file:
        ign=pickle.load(open(ign_file,'rb'))
    #for counter in range(1):
    dataset=None
    while True:
        print(counter)
        filename=input_file_pattern+str(counter)+".pickle"
        if os.path.exists(filename):
            with open(filename,'rb') as f:
                p=pickle.load(f)
                if DO_MASK==False:
                    p[0]['l2l_att_mask']=tf.zeros(tf.shape(p[0]['l2l_att_mask']),"int32")
                    p[0]['l2g_att_mask']=tf.zeros(tf.shape(p[0]['l2g_att_mask']),"int32")
                    p[0]['g2l_att_mask']=tf.zeros(tf.shape(p[0]['g2l_att_mask']),"int32")
                    p[0]['g2g_att_mask']=tf.zeros(tf.shape(p[0]['g2g_att_mask']),"int32")
                features=p[0]
                if ONE_LABEL==-1:
                    labels=p[1]['global_label_id']
                else:
                    labels=p[1]['global_label_id'][:,:,:,ONE_LABEL]
                #Will it fix it if I remove the batch dimension so it can be readded?
                for key in features:
                    tmp=tf.squeeze(features[key],0)
                    #if tf.equal(tf.size(list_features[i][key]),tf.size(tmp)):
                    features[key]=tmp
                    #else:#This only works properly if the first dimension is only size one and can be removed without data loss
                    #    print("Error: Inputs should be unbatched initially")
                    #    exit()
                if ign_file:
                    labels=tf.subtract(labels,2*tf.cast(tf.equal(tf.zeros(tf.shape(labels),"int32"),ign[counter]),"int32"))
            list_features.append(features)
            list_labels.append(labels)
            if len(list_features)==100 or not os.path.exists(input_file_pattern+str(counter+1)+".pickle"):
                feature_d=tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(list_features).to_dict(orient="list"))
                label_d=tf.data.Dataset.from_tensor_slices(tf.concat(list_labels,0))
                d=tf.data.Dataset.zip((feature_d,label_d))
                if dataset==None:
                    dataset=d
                else:
                    dataset=dataset.concatenate(d)
                list_features=[]
                list_labels=[]
        else:
            break
        counter+=1
    return dataset

#Create model
model_config.num_hidden_layers=NUM_HIDDEN_LAYERS
model=MyModel(model_config)

#Compile the model
#We may need to change loss/metrics later
print("Compiling model")
#testing metrics at various thresholds for the logits
if OPTIMIZER=="Adam":
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
elif OPTIMIZER=="AdamW":
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=LEARNING_RATE)
if LOSS_FUNC=="AFL":
    metrics=[AFLPrecision(),AFLRecall(),AFLPrecision(name="ignAFLprecision",ign=True),AFLRecall(name="ignAFLrecall",ign=True)]#,tf.keras.metrics.BinaryAccuracy()]
    loss=adaptiveFocalLoss
else:
    if LOSS_FUNC=="BCE":
        loss=customWeightedBinaryCrossentropy
    elif LOSS_FUNC=="F1":
        loss=softF1Loss
    metrics=[tf.keras.metrics.BinaryAccuracy()]
    for i in range(21):
        x=i*0.5-5
        metrics.append(IGNPrecision(x,False)(name=str(x)+"precision"))
        metrics.append(IGNRecall(x,False)(name=str(x)+"recall"))
        metrics.append(IGNPrecision(x,True)(name="ign"+str(x)+"precision"))
        metrics.append(IGNRecall(x,True)(name="ign"+str(x)+"recall"))
#    metrics.append(tf.keras.metrics.Precision(thresholds=[i*0.5-10]))
#    metrics.append(tf.keras.metrics.Recall(thresholds=[i*0.5-10]))
model.compile(
        optimizer=optimizer,
#        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#        loss=customWeightedCategoricalCrossentropy,
#        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#        loss=customWeightedBinaryCrossentropy,
#        loss=softF1Loss,
#        loss=adaptiveFocalLoss,
        loss=loss,
        metrics=metrics,
#        run_eagerly=True
        )
#If continuing, load weights from filename
if CONTINUE and RUN_NUMBER==-1:
    model.load_weights(MODEL_FILENAME)
    MODEL_FILENAME+="+"+str(NUM_EPOCHS)
#load weights from checkpoint, line up with Keras model
elif PRETRAIN:
    #load test input to build the model
    testinputfile="./"
    testinputfile+=TOKENIZER
    if ANNOTATE and not FULL:
        testinputfile+="anno"
    elif ANNOTATE and FULL:
        testinputfile+="constitfull"
    testinputfile+="examples/tf_example0.pickle"
    testinput=pickle.load(open(testinputfile,'rb'))[0]
    model(testinput,training=False)
    #get weights from ckpt
    reader=py_checkpoint_reader.NewCheckpointReader(PRETRAIN_FILE)
    d={}
    var_to_shape_map=reader.get_variable_to_shape_map()
    for key, value in sorted(var_to_shape_map.items()):
        d[key]=reader.get_tensor(key)
        print(key)
    #CONTINUE with the tests below, and there also may be more weights we need outside the 12 layers
    #make sure the key/value projection layers are Dense
    print(model.etcmodel.global_local_transformer.fused_att_layers[0].inner_layer.g2g_key_projection)
    for weight in model.etcmodel.global_local_transformer.fused_att_layers[0].inner_layer.g2g_key_projection.weights:
        print(weight.name)
    #see which way the qkv_attention layer goes
    print(model.etcmodel.global_local_transformer.fused_att_layers[0].inner_layer.global_qkv_attention)
    for weight in model.etcmodel.global_local_transformer.fused_att_layers[0].inner_layer.global_qkv_attention.weights:
        print(weight.name)

    #load weights from BERT pretrained model
    for i in range(12):
        #feed-forward weights
        for j in range(2):
            kernel=d["etc_document_bert/global_local_transformer_layers/feed_forward_layer_"+str(i)+"/dense_layers/layer_"+str(j)+"/kernel"]
            bias=d["etc_document_bert/global_local_transformer_layers/feed_forward_layer_"+str(i)+"/dense_layers/layer_"+str(j)+"/bias"]
            model.etcmodel.global_local_transformer.long_feed_forward_layers[i].inner_layer.layers[j].set_weights([kernel,bias])
            gamma=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/layer_norm_"+str(j)+"/gamma"]
            beta=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/layer_norm_"+str(j)+"/beta"]
            model.etcmodel.global_local_transformer.fused_att_layers[i].normalization_layers[j].set_weights([gamma,beta])
        beta=d["etc_document_bert/global_local_transformer_layers/feed_forward_layer_"+str(i)+"/layer_norm/beta"]
        gamma=d["etc_document_bert/global_local_transformer_layers/feed_forward_layer_"+str(i)+"/layer_norm/gamma"]
        #gamma is initialized to one, beta to zero, I printed the initialized weights w/ get_weights, gamma goes first.
        model.etcmodel.global_local_transformer.long_feed_forward_layers[i].normalization_layers[0].set_weights([gamma,beta])
        #fused-attention weights
        for j in ["g","l"]:#key/value projection
            for k in ["g","l"]:
                for l in ["key","value"]:
                    kernel=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/"+j+"2"+k+"_"+l+"_projection/linear/kernel"]
                    bias=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/"+j+"2"+k+"_"+l+"_projection/linear/bias"]
                    setweightcommand="model.etcmodel.global_local_transformer.fused_att_layers["+str(i)+"].inner_layer."+j+"2"+k+"_"+l+"_projection.set_weights([kernel,bias])"
                    exec(setweightcommand)
        for j in ["global","long"]:#global/local output_projection/qkv_attention/query_projection
            kernel=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/"+j+"_output_projection/kernel"]
            bias=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/"+j+"_output_projection/bias"]
            setweightcommand="model.etcmodel.global_local_transformer.fused_att_layers["+str(i)+"].inner_layer."+j+"_output_projection.set_weights([kernel,bias])"
            exec(setweightcommand)
            kernel=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/"+j+"_query_projection/linear/kernel"]
            bias=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/"+j+"_query_projection/linear/bias"]
            setweightcommand="model.etcmodel.global_local_transformer.fused_att_layers["+str(i)+"].inner_layer."+j+"_query_projection.linear.set_weights([kernel,bias])"
            exec(setweightcommand)
        emb=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/global_qkv_attention/relative_emb_table"]
        bias=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/global_qkv_attention/relative_bias_table"]
        #which way does it go in set_weights? Can't determine without experiment
        model.etcmodel.global_local_transformer.fused_att_layers[i].inner_layer.global_qkv_attention.set_weights([emb,bias])
        emb=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/long_qkv_attention/qkv_relative_attention/relative_emb_table"]
        bias=d["etc_document_bert/global_local_transformer_layers/fused_att_layer_"+str(i)+"/fused_global_local_att/long_qkv_attention/qkv_relative_attention/relative_bias_table"]
        #which way does it go in set_weights? Can't determine without experiment
        model.etcmodel.global_local_transformer.fused_att_layers[i].inner_layer.long_qkv_attention.qkv_relative_attention.set_weights([emb,bias])
#    for weight in weights:
#        print(weight,tf.shape(weight))
    #weights outside the 12 layers
    gamma=d["etc_document_bert/global_emb_layer_norm/gamma"]
    beta=d["etc_document_bert/global_emb_layer_norm/beta"]
    model.etcmodel.global_token_embedding_norm.set_weights([gamma,beta])
    gamma=d["etc_document_bert/long_emb_layer_norm/gamma"]
    beta=d["etc_document_bert/long_emb_layer_norm/beta"]
    model.etcmodel.token_embedding_norm.set_weights([gamma,beta])
    segment_table=d["etc_document_bert/segment_emb_lookup/embedding_table"]
    token_table=d["etc_document_bert/token_emb_lookup/embedding_table"]
    model.etcmodel.segment_embedding.set_weights([segment_table])
    model.etcmodel.token_embedding.set_weights([token_table])
    aflstr=""
    if LOSS_FUNC=="AFL":
        aflstr="AFL"
#    model.save_weights(TOKENIZER+"mymodels/pretrainedmodel"+aflstr+str(NUM_FINAL_LAYERS)+"Fin")
    #initialize the model with the pretrained model
    #CONTINUE: find another way with the tensor dict we just created

#Load data
print("Loading train data")
train_dataset=get_input_data(INPUT_TRAIN_PATH)
print("Shuffling train data")
train_dataset=train_dataset.shuffle(buffer_size=4096).batch(BATCH_SIZE)
print("Loading val data")
val_dataset=get_input_data(INPUT_VAL_PATH,IGN_PATH)
val_dataset=val_dataset.batch(BATCH_SIZE)

#train the model
print("Training model")
callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="mymodel_{epoch}")]
model.fit(train_dataset,validation_data=val_dataset,epochs=NUM_EPOCHS)#,callbacks=callbacks)
print("Training completed:")
print(MODEL_FILENAME)
if FLAGS.RUN_NUMBER==-1:
    try:
        model.save_weights(MODEL_FILENAME)
    except Exception as e:
        print(e)
#test=pickle.load(open("examples/tf_example0.pickle",'rb'))
#testinput=test[0]
#y_true=test[1]['global_label_id']
#y_pred=model(testinput,training=False)
#print(y_true.numpy())
#print(y_pred.numpy())
#evaluate the model
#print("Evaluating model")
#result=model.evaluate(val_dataset)
#dict(zip(model.metrics_names,result))

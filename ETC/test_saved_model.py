import json
import os
import time
import pickle

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
BATCH_SIZE=5
ONE_LABEL=-1#Which label to focus solely on? If we aren't focusing on only one, this should be -1
#Define the model we're using
NUM_HIDDEN_LAYERS=4
MODEL_FILENAME="./ALBERTmymodels/4Layers5Batch2Epoch.0001LRg2l"
class MyModel(tf.keras.Model):
    def __init__(self,model_config,ckpt=None):
        super().__init__()
        if ckpt:
            self.etcmodel=tf.keras.models.load_model(ckpt)
        else:
            self.etcmodel=modeling.EtcModel(model_config)
#        self.final_layer=tf.keras.layers.Dense(units=NUM_LABELS+1)#96 relations and threshold(for Adaptive Focal Loss
        if ONE_LABEL==-1:
            self.final_layer=tf.keras.layers.Dense(units=NUM_LABELS)
        else:
            self.final_layer=tf.keras.layers.Dense(units=1)
    def call(self,inputs,training=False):
        _,global_output=self.etcmodel(**inputs,training=training)
        entities=global_output[:,GLOBAL_SIZE-NUM_ENTITIES:GLOBAL_SIZE,:]
        #print the entity representations
        #for i in range(NUM_ENTITIES):
        #    for j in range(1024):
        #        print(entities[0,i,j].numpy(),end=' ')
        #    print('\n')
        all_pairs_shape=tf.stack([tf.shape(entities)[0],tf.shape(entities)[1],tf.shape(entities)[1],tf.shape(entities)[2]],axis=0)
        global_broadcast1=tf.broadcast_to(entities[:,:,tf.newaxis,:],all_pairs_shape)
        global_broadcast2=tf.broadcast_to(entities[:,tf.newaxis,:,:],all_pairs_shape)
        pairs_embedding=tf.concat([global_broadcast1,global_broadcast2],axis=-1)
        result=self.final_layer(pairs_embedding,training=training)
        return result#return as logits in this case


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
    zeroweight=1/200.0
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

#predictions have one extra, the threshold, those predicted above it are predicted positive, below it, negative
def adaptiveFocalLoss(y_true,y_pred):
    gamma=0.0
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
    final=tf.multiply(y_true,positive_log_probs)
#    final=tf.multiply(y_true,tf.multiply(positive_log_probs,gamma_term))
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
            print(y_true)
            print(y_pred)
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
            print(y_true)
            print(y_pred)
        def result(self):
            return self.metric.result()
        def reset_state(self):
            self.metric.reset_state()
    return thresholdMetric


def get_input_data(input_file_pattern):
    list_features=[]
    list_labels=[]
    counter=0
    while True:
        filename=input_file_pattern+str(counter)+".pickle"
        print(counter)
        counter+=1
        if os.path.exists(filename):
            with open(filename,'rb') as f:
                p=pickle.load(f)
                list_features.append(p[0])
                if ONE_LABEL==-1:
                    list_labels.append(p[1]['global_label_id'])
                else:
                    list_labels.append(p[1]['global_label_id'][:,:,:,ONE_LABEL])
        else:
            break
    #Will it fix it if I remove the batch dimension so it can be readded?
    for i in range(len(list_features)):
        for key in list_features[i]:
            tmp=tf.squeeze(list_features[i][key],0)
            #if tf.equal(tf.size(list_features[i][key]),tf.size(tmp)):
            list_features[i][key]=tmp
            #else:#This only works properly if the first dimension is only size one and can be removed without data loss
            #    print("Error: Inputs should be unbatched initially")
            #    exit()

    feature_d=tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(list_features).to_dict(orient="list"))
    labels=tf.concat(list_labels,0)
    #reshape the labels so that each value has its own dimension of one
    #labels=tf.expand_dims(labels,-1)
    #change labels to one-hot for precision/recall
    #labels=tf.one_hot(labels,depth=98,axis=-1)
    label_d=tf.data.Dataset.from_tensor_slices(labels)
    d=tf.data.Dataset.zip((feature_d,label_d))
    return d
model_config=input_utils.get_model_config(
        model_dir="./etcmodel/pretrained",
#        source_file="./etcmodel/pretrained/etc_config.json",
        write_from_source=True)
model_config.num_hidden_layers=NUM_HIDDEN_LAYERS
model=MyModel(model_config)
#Compile the model
#We may need to change loss/metrics later
print("Compiling model")
#testing metrics at various thresholds for the logits
#metrics=[AFLPrecision(),AFLRecall()]#,tf.keras.metrics.BinaryAccuracy()]
metrics=[tf.keras.metrics.BinaryAccuracy()]
for i in range(21):
    metrics.append(ThresholdPrecision(i-10)())
    metrics.append(ThresholdRecall(i-10)())
#    metrics.append(tf.keras.metrics.Precision(thresholds=[i*0.5-10]))
#    metrics.append(tf.keras.metrics.Recall(thresholds=[i*0.5-10]))
model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
#        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#        loss=customWeightedCategoricalCrossentropy,
#        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=customWeightedBinaryCrossentropy,
#        loss=softF1Loss,
#        loss=adaptiveFocalLoss,
        metrics=metrics
        )
#model.save_weights("./mymodels/untrainedmodel")
model.load_weights(MODEL_FILENAME)
val_dataset=get_input_data("./devexamples/tf_example")
val_dataset=val_dataset.batch(BATCH_SIZE)
result=model.evaluate(val_dataset)
dict(zip(model.metrics_names,result))
#test=pickle.load(open("examples/tf_example0.pickle",'rb'))
#testinput=test[0]
#y_true=test[1]['global_label_id']
#y_pred=model(testinput,training=False)
#if ONE_LABEL!=-1:
#    y_true=y_true[:,:,:,ONE_LABEL]
#for i in range(NUM_ENTITIES):
#    for j in range(NUM_ENTITIES):
#        print(y_true[0,i,j].numpy(),end=' ')
#    print('\n')
#for i in range(NUM_ENTITIES):
#    for j in range(NUM_ENTITIES):
#        print(y_pred[0,i,j,0].numpy(),end=' ')
#    print('\n')
#

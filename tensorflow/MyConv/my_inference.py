import tensorflow as tf
class ConvInference(object):

    def inference(input_tensor):
    
class TrainModule(object):
    ConvInference train_inference
    Batch_Size
    Learning_Rate_Base
    Learning_Rate_Decay
    Input
    Output
    x_ph
    y_ph
    mad
    ema
    step
    global_step = tf.Variable(0,trainable=false)
    cross_entropy
    Learning_Rate
    def __init__(self,ConvInference inference,num):
        self.train_inference = inference
        return self
    def setBatchSize(self,num):
        self.Batch_Size = num
        x_ph = tf.placeholder(tf.float32,[None,784],name="x_input")
        y_ph = tf.placeholder(tf.float32,[None,10],name="y_input")
        return self
    def setStep(self,num):
        self.step = num        
    def setLearningRate(self,b,d):
        self.Learning_Rate_Base = b
        self.Learning_Rate_Decay = d
        Learning_Rate = tf.train.exponentia_decay(
            Learning_Rate_Base,
            global_step
            mnist.train.num_examples/Batch_Size,
            Learning_Rate_Decay)
        return self
    def setREGU(self,num):
        self.REGU = num
        return self
    def setMovingAverageDecay(self,num):
        self.mad = num
        ema = tf.train.ExponentialMovingAverage(mad,global_step)
        return self
    def setIO(self,inin,out):
        self.Input = inin
        self.Output = out
        return self
    def train(input_tensor):
        y = train_inference.inference(input_tensor)
        ema+op = ema.apple(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, lables = argmax(y_,1))
        

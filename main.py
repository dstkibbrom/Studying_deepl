import tensorflow as tf
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n

    def add(self,*args):
        self.data=[a+float(b) for a, b in zip(self.data,args)]
    def __getitem__(self, index):
        return self.data[index]

W=tf.Variable(tf.random.normal(shape=(784,10)),mean=0,stddev=0.01)
b=tf.Variable(tf.zeros(shape=(10)))

def Softmax(X):   # 256,10
    X_exp=tf.exp(X)
    summa=tf.reduce_sum(X,axis=1,keepdims=True)

    return X_exp/summa
def net(X):
    return Softmax(tf.matmul(X,W)+b)

def cross_entropy (y_hat,y): # (256,10)  (256,1)
    return -tf.math.log(tf.boolean_mask(y_hat,tf.one_hot(y,depth=y_hat.shape[-1])))


class Updator():
    def __init__(self,params,lr):
        self.params=params
        self.lr=lr

    def __call__(self, batch_size, grads):
        sgd(self.params,grads,self.lr,batch_size)




def train_epoch_ch3(net, train_iter, loss, updater):
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
            print(X.shape[0])
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model (defined in Chapter 3)."""

    for epoch in range(num_epochs):
        print(len(list(train_iter)))
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        print("epoch:",epoch,"acc:",train_acc,"loss:",train_loss)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
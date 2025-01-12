import jax
import jax.numpy as jnp
import jax.nn as nn
from jax import random, grad
from functools import partial


def initialise_params(dim, seed=0):
    """
    This function initialises the parameters of a fully connected neural network.

    Parameters:
    dim -- The dimensions of the neural network.
    seed -- The seed for random numbers.

    Return:
    The initialised parameters of the neural network.
    """

    key = random.key(seed)
    params = {}
    for i in range(len(dim)-1):
        key, subkey = random.split(key)
        params[f"W{i}"] = random.normal(subkey, shape=(dim[i+1], dim[i])) * jnp.sqrt(2.0 / dim[i]) # he initialisation
        params[f"b{i}"] = jnp.zeros(dim[i+1]) # note this is a row vector
    return params


@jax.jit
def forward(params, inputs):
    """
    This is the forward pass function of a fully connected neural network.

    Parameters:
    params -- The parameters of the network.
    inputs -- The input vector fed into the network.

    Return:
    The output vector of the network.
    """

    outputs = inputs
    L = int(len(params)/2) # Number of layers
    for i in range(L):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        if i != L-1:
            outputs = nn.relu(outputs @ jnp.transpose(W) + b)
        else:
            outputs = nn.softmax(outputs @ jnp.transpose(W) + b)
    return outputs
        

@jax.jit
def cross_entropy(outputs, labels):
    """
    Cross entropy function to compare outputs of a network with true labels.

    Parameters:
    outputs -- The output vector of a neural network.
    labels -- A vector of true labels to compare the outputs with.

    Return:
    The average cross entropy between the output values and the true labels
    """

    epsilon = 1e-12
    outputs_clipped = jnp.clip(outputs, epsilon, 1 - epsilon)
    labels_flat = labels.flatten()
    
    one_hot_labels = jax.nn.one_hot(labels_flat, num_classes=outputs_clipped.shape[1])
    cross_entropy = -jnp.sum(one_hot_labels * jnp.log(outputs_clipped)) / outputs_clipped.shape[0]

    # This version of cross entropy with an extra term doesn't seem to work properly and I'm yet to figure out why
    # cross_entropy = -jnp.sum(one_hot_labels * jnp.log(outputs_clipped) + (1 - one_hot_labels) * jnp.log(1 - outputs_clipped)) / outputs_clipped.shape[0]

    return cross_entropy


@partial(jax.jit, static_argnames=['loss_func'])
def calc_loss(loss_func, params, inputs, labels):
    """
    This function calculates the loss of neural network in representing it's intended function

    Parameters:
    loss_func -- A function with which to calculate the loss with respect to.
    params -- The parameters of the neural network.
    inputs -- The input vector of the network.
    labels -- A vector of true labels to compare an output vector with.

    Return:
    The loss of the network with respect to the loss function and true labelled data.
    """

    outputs = forward(params, inputs)
    return loss_func(outputs, labels)



def gen_batches(key, inputs, batch_size):
    """
    This function generates batches of data to be used for e.g. stochastic gradient descent.

    Parameters:
    key -- A random key used to handle randomness in jax.
    inputs -- Input data to be batched (2D vector).
    batch_size -- The size of batches to split the input data into.

    Return:
    An array of batches. (Note there may be one batch that is smaller than the rest if the input data cannot be evenly divided by the batch_size)
    """

    permutations = random.permutation(key, inputs)
    splits = jnp.arange(batch_size, inputs.shape[0], batch_size)
    return jnp.split(permutations, splits, axis=0)


@jax.jit
def batch_size_linear(batch_size_initial, batch_size_final, total_epochs, current_epoch):
    """
    This function linearly interpolates some batch size between some initial and final size throughout the training.

    Parameters:
    batch_size_initial -- The initial batch size at start of training.
    batch_size_final -- The final batch size at end of traing.
    total_epochs -- The number of epochs in the training process.
    current_epoch -- The current epoch of the training process.

    Return:
    The current batch size to use for a given epoch.
    """

    return jnp.round(batch_size_initial + ((current_epoch-1)/(total_epochs-1))*(batch_size_final - batch_size_initial)).astype(int)


@jax.jit
def learning_rate_decay(lr_initial, lr_final, tau, current_epoch):
    """
    This function adjusts the learning rate with an expontential rate of decay throughout training.

    Parameters:
    lr_initial -- The initial learning rate.
    lr_final -- The final learning rate.
    tau -- The time constant for decay.
    current_epoch -- The current epoch of the training process.

    Return:
    The current learning rate to use for a given epoch.
    """

    return (lr_final + (lr_initial -  lr_final)*jnp.exp(-(current_epoch-1)/tau))


def train(params, inputs_train, labels_train, inputs_test, labels_test, epochs, batch_size_func, lr_func, loss_func=cross_entropy, seed=0):
    """
    This function is the main training loop for the neural network.

    Parameters:
    params -- The parameters of the network.
    inputs_train -- The training data inputs.
    labels_train -- The training data labels.
    inputs_test -- The test data inputs.
    labels_test -- The test data labels.
    epochs -- The number of epochs over which to train.
    batch_size_func -- A function that takes in a single argument of the current epoch and returns the current batch_size.
    lr_func -- A function that takes in a single argument of the current epoch and returns the current learning rate.
    loss_func -- A function with which to calculate the loss with respect to.
    seed -- The seed for random numbers.

    Return:
    params -- The updated parameters of the trained neural network.
    train_losses -- An array of the average training loss at each epoch.
    test_losses -- An array of the average test loss at each epoch.
    test_percents -- An array of the percentage of correctly classified test data at each epoch.
    """

    key = random.key(seed)
    train_losses = jnp.array([])
    test_losses = jnp.array([])
    test_percents = jnp.array([])

    for epoch in range(1, epochs+1):
        batch_size = batch_size_func(epoch)
        lr = lr_func(epoch)
        key, subkey = random.split(key)
        batches_inputs = gen_batches(subkey, inputs_train, batch_size)
        batches_labels = gen_batches(subkey, labels_train, batch_size)
        train_loss = 0 # Average loss per sample
        num_samples_seen = 0

        for batch_inputs, batch_labels in zip(batches_inputs, batches_labels):

            batch_labels = batch_labels.flatten()

            num_batch_samples = batch_inputs.shape[0]
            train_loss = ((num_samples_seen * train_loss) + (num_batch_samples * calc_loss(loss_func, params, batch_inputs, batch_labels))) / (num_samples_seen + num_batch_samples)
            num_samples_seen += num_batch_samples
            grads = grad(lambda x: calc_loss(loss_func, x, batch_inputs, batch_labels))(params)

            params = {p: q - lr * grads[p] for p, q in params.items()}
        
        test_loss, test_percent = evaluate(params, inputs_test, labels_test)

        train_losses = jnp.append(train_losses, train_loss)
        test_losses = jnp.append(test_losses, test_loss)
        test_percents = jnp.append(test_percents, test_percent)

        # if epoch % 5 == 0:
        #     print(f"Epoch: {epoch}, Loss: {loss:.4f}, Percent correct: {test_percent:.2f}%")

    return params, train_losses, test_losses, test_percents


@jax.jit
def evaluate(params, inputs_test, labels_test, loss_func=cross_entropy):
    """
    This function evaluates the loss and percentage of correctly classified data points.

    Parameters:
    params -- The parameters of the neural network.
    inputs_test -- The input data. (Test data)
    labels_test -- The data labels. (Test data).
    loss_func -- A function with which to calculate the loss with respect to.

    Return:
    test_loss -- The average loss of the test data.
    percent_correct -- The percentage of correctly classified test data.
    """
    
    # Test loss
    test_loss = calc_loss(loss_func, params, inputs_test, labels_test)

    # Percetage correct
    outputs_test = jnp.argmax(forward(params, inputs_test), axis=1)
    correct = sum(outputs_test == labels_test.flatten())
    total = labels_test.shape[0]
    percent_correct = 100 * correct / total

    return test_loss, percent_correct

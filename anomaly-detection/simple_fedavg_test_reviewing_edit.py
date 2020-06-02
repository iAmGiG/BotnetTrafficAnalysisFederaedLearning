# Lint as: python3
# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end example testing Federated Averaging against the MNIST model."""
# """pulled from: experimental file under: 'tensorflow_federated\python\research\simple_fedavg' """

import collections
import functools
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.simple_fedavg import simple_fedavg_tf
from tensorflow_federated.python.research.simple_fedavg import simple_fedavg_tff


def _create_test_cnn_model(only_digits=True):
    """A simple CNN model for test."""
    data_format = 'channels_last'
    input_shape = [28, 28, 1]

    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding='same',
        data_format=data_format)
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu)

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=input_shape),
        max_pool(),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(10 if only_digits else 62),
        # tf.keras.layers.Activation(tf.nn.softmax),
        tf.keras.layers.Dense(int(0.75 * 10), activation="tanh", input_shape=(10,)),
        tf.keras.layers.Dense(int(0.5 * 10), activation="tanh"),
        tf.keras.layers.Dense(int(0.33 * 10), activation="tanh"),
        tf.keras.layers.Dense(int(0.25 * 10), activation="tanh"),
        tf.keras.layers.Dense(int(0.33 * 10), activation="tanh"),
        tf.keras.layers.Dense(int(0.5 * 10), activation="tanh"),
        tf.keras.layers.Dense(int(0.75 * 10), activation="tanh"),
        tf.keras.layers.Dense(10)
    ])

    return model


def _create_random_batch():
    return collections.OrderedDict(
        x=tf.random.uniform(tf.TensorShape([1, 28, 28, 1]), dtype=tf.float32),
        y=tf.constant(1, dtype=tf.int32, shape=[1]))


def _model_fn():
    keras_model = _create_test_cnn_model(only_digits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
        y=tf.TensorSpec([None], tf.int32))
    return simple_fedavg_tf.KerasModelWrapper(keras_model, input_spec, loss)


MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def create_mnist_variables():
    return MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name='bias',
            trainable=True),
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def mnist_forward_pass(variables, batch):
    y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(
        tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32))

    num_examples = tf.cast(tf.size(batch['y']), tf.float32)

    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions


def get_local_mnist_metrics(variables):
    return collections.OrderedDict(
        num_examples=variables.num_examples,
        loss=variables.loss_sum / variables.num_examples,
        accuracy=variables.accuracy_sum / variables.num_examples)


# core on what to study
# here a return is done, rather then the main creation of a computation being sent....
@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
    return collections.OrderedDict(
        num_examples=tff.federated_sum(metrics.num_examples),
        loss=tff.federated_mean(metrics.loss, metrics.num_examples),
        accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))


class MnistModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def weights(self):
        return simple_fedavg_tf.ModelWeights(
            trainable=self.trainable_variables,
            non_trainable=self.non_trainable_variables)

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 784], tf.float32),
            y=tf.TensorSpec([None, 1], tf.int32))

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, _ = mnist_forward_pass(self._variables, batch)
        return loss

    @tf.function
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

    # this property has the ability to call tff.fed_computations....
    @property
    def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients


def create_client_data():
    emnist_batch = collections.OrderedDict(
        label=[5], pixels=np.random.rand(28, 28))

    output_types = collections.OrderedDict(label=tf.int32, pixels=tf.float32)

    output_shapes = collections.OrderedDict(
        label=tf.TensorShape([1]),
        pixels=tf.TensorShape([28, 28]),
    )

    dataset = tf.data.Dataset.from_generator(lambda: (yield emnist_batch),
                                             output_types, output_shapes)

    def client_data():
        return tff.simulation.models.mnist.keras_dataset_from_emnist(
            dataset).repeat(2).batch(2)

    return client_data


# unit test:
#
class SimpleFedAvgTest(tf.test.TestCase):

    def test_something(self):
        """
        @it_precess: is a federated averaging process, note the inputs for build fed avg process, we are given the
            option to build that process with any server/client optimizer via anonymous functions, this way the passing
            of custom optimizer is fluid.
        @fed_data_type: the parameter of the next in the iterative process through its type signature.
        :return:
        """
        it_process = simple_fedavg_tff.build_federated_averaging_process(_model_fn)
        self.assertIsInstance(it_process, tff.templates.IterativeProcess)
        federated_data_type = it_process.next.type_signature.parameter[1]
        self.assertEqual(
            str(federated_data_type),
            '{<x=float32[?,28,28,1],y=int32[?]>*}@CLIENTS')

    def test_simple_training(self):
        it_process = simple_fedavg_tff.build_federated_averaging_process(_model_fn)
        server_state = it_process.initialize()
        Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name

        # Test out manually setting weights:
        keras_model = _create_test_cnn_model(only_digits=True)

        def deterministic_batch():
            return Batch(
                x=np.ones([1, 28, 28, 1], dtype=np.float32),
                y=np.ones([1], dtype=np.int32))

        batch = tff.tf_computation(deterministic_batch)()
        federated_data = [[batch]]

        def keras_evaluate(state):
            tff.learning.assign_weights_to_keras_model(keras_model,
                                                       state.model_weights)
            keras_model.predict(batch.x)

        loss_list = []
        for _ in range(3):
            keras_evaluate(server_state)
            server_state, loss = it_process.next(server_state, federated_data)
            loss_list.append(loss)
        keras_evaluate(server_state)

        self.assertLess(np.mean(loss_list[1:]), loss_list[0])

    def test_self_contained_example_custom_model(self):

        client_data = create_client_data()
        train_data = [client_data()]

        trainer = simple_fedavg_tff.build_federated_averaging_process(MnistModel)
        state = trainer.initialize()
        losses = []
        for _ in range(2):
            state, loss = trainer.next(state, train_data)
            # Track the loss.
            losses.append(loss)
        self.assertLess(losses[1], losses[0])


# helper function
def server_init(model, optimizer):
    """Returns initial `ServerState`.

    Args:
      model: A `tff.learning.Model`.
      optimizer: A `tf.train.Optimizer`.

    Returns:
      A `ServerState` namedtuple.
    """
    simple_fedavg_tff._initialize_optimizer_vars(model, optimizer)
    return simple_fedavg_tf.ServerState(
        model_weights=model.weights,
        optimizer_state=optimizer.variables(),
        round_num=0)


# Server testing
# here the TestCase from tf is being used as a way to generate a common tf test.
class ServerTest(tf.test.TestCase):

    def _assert_server_update_with_all_ones(self, model_fn):
        """
        @model_function: a param function that is used to define the expected model on the server.
        @optimizer_fn: a stocastic gradient decent optimizer function.
        @state: an initial server state, made up of the model and optimizer
        @weights_delta: a map_structure,
            tf.ones_llike - "Creates a tensor of all ones that has the same shape as the input."
            model.trainable_variables - variables.weights and self._variables.bias - see above define locally.
        :param model_fn:
        :return:
        """
        optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
        model = model_fn()
        optimizer = optimizer_fn()
        state = server_init(model, optimizer)
        weights_delta = tf.nest.map_structure(tf.ones_like,
                                              model.trainable_variables)

        """
        @ '_' : indicates a through away variable
        @ range from 0 - 2
        the state on each run will update with the new model, optimizer, state, weights_delta
        """
        for _ in range(2):
            state = simple_fedavg_tf.server_update(model, optimizer, state,
                                                   weights_delta)
        '''
        this will "Evaluates tensors and returns numpy values."
        
        '''
        model_vars = self.evaluate(state.model_weights)
        train_vars = model_vars.trainable
        self.assertLen(train_vars, 2)
        self.assertEqual(state.round_num, 2)
        # weights are initialized with all-zeros, weights_delta is all ones,
        # SGD learning rate is 0.1. Updating server for 2 steps.
        self.assertAllClose(train_vars, [np.ones_like(v) * 0.2 for v in train_vars])

    def test_self_contained_example_custom_model(self):
        self._assert_server_update_with_all_ones(MnistModel)


# client testing
# this class will take in the tf.test.testcase, (some kinda of test case made by tf.test.TestCase)
class ClientTest(tf.test.TestCase):

    def test_self_contained_example(self):
        """
        @client_data: uses the method creat client data, see for details
        @model: retrieves the test model, this will be needed as a way to break down how to design a model for simulation
        @optimizer_function: is a standard keras optimizer, in this case a stochastic gradient decent
        @losses: this list appears to be the list to hold all losses from the clients
        :return: this method never needs to return, it acts as the functional main
        """
        client_data = create_client_data()

        model = MnistModel()
        optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
        losses = []

        '''
        this component is very important for the main simulation running
        present is a range where range(2) means: the creation of a sequence of numbers between 0 and n (where n = 2)
        @optimizer pulls in the anonymous function that is only usable in this scope (why not just do this down bellow?)
        @simple_fedavg_tff._initialize_optimizer_vars(model, optimizer): this comes from the simple_fedavg_tff py
            (see for details)
        @server_message: this broadcast message will push out hte model.weights and the round number, 
            why model_weights?
            round number: might be the need to know which round for logging
        @outputs: client update - uses the model, client data, the server message and the optimizer
        @losses: appends the outputs to the list. model_output converted to numpy()(what this does exactly?)        
        '''
        for r in range(2):
            optimizer = optimizer_fn()
            simple_fedavg_tff._initialize_optimizer_vars(model, optimizer)
            server_message = simple_fedavg_tf.BroadcastMessage(
                model_weights=model.weights, round_num=r)
            outputs = simple_fedavg_tf.client_update(model,
                                                     client_data(),
                                                     server_message,
                                                     optimizer)
            losses.append(outputs.model_output.numpy())
        # asserting that the outputs.client_weight.numpy is 2..... need to know why? or not...
        self.assertAllEqual(int(outputs.client_weight.numpy()), 2)
        # assert that the loss at 1 are less than the loss at 0
        self.assertLess(losses[1], losses[0])
        # for these assertions they should be to test that all the simulated clients will produce the same results
        # meaning all assert fails should indicate and error during runtime.
        # cannot confirm


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()

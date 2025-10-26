# Development Notes - 2020 Research Period

**Author Notes**: These are consolidated development notes from May-June 2020 during the initial implementation of federated learning for IoT botnet detection. Preserved for historical reference and understanding the original implementation approach.

---

## Federated Learning Concepts (Original Notes)

### What is Federated Learning?

From my original understanding in 2020:

* Putting the computation on the device
* Where each device trains on its own model
* As there is not enough data on each device to truly train a full model, the contributions of other devices elevate this issue
* This is also mitigated by pretraining the model at the server level on proxy data
* Issue here is that as data becomes outdated, the model may not pick up the new data as well when raising issues

### Learning Together - The Process

1. Start with server model
2. Distribute this model to some of the clients
3. Clients produce a new model, and then the model is pushed back to the server for aggregation
4. The average model reflects the training from every model
5. Then repeat for each round (where a round = a session of learning)

---

## TensorFlow Federated (TFF) Implementation

### Architecture Overview

* There exist interfaces to represent federated data sets
* Built out of FL core and FL API

### Federated Learning Outline

**Training is represented as a pair of computations:**
1. Initialize state
2. Single round execution

Both can be executed like functions in Python, and **when we do, they by default execute in a local simulation** ← **IMPORTANT for paper writing**

**The state includes:**
* The model
* The training data

**Status note**: *At the time of this writing debugging has not begun fully (mid-May 2020)*

### The Botnet FL Situation

My conceptualization of the problem:

* Take that we have clients that are mirroring the traffic data going through a security checkpoint (in this case network signals going through a router to IoT devices)
* All the collected network signals can be viewed as a single federated value
* Where we say a federated value is a **multi-set**
* In TFF the multi-set has types - the types consist of the identity of the devices that host the value(s)
* This is the **placement** - the client placement
* On the server, in this case some global aggregation handler

### Federated Operations

1. **Broadcast to clients** through `tff.federated_broadcast` - the first federated action
2. **Task performed on clients**
3. **Clients feedback** the `tff.federated_average`

---

## Development Timeline & Logs

### May 28, 2020 - Simulation Discovery

Managed to review some test code from the experimental/simulation section of TFF. This code may produce simulation results.

**Key insight**: This simulation will reflect the intended results of federated learning without the need to set up an entire production pipeline to run single experiments.

**IMPORTANT NOTE**: This code is marked as a unit test - production and simulation are not the same and the intended results may differ.

### June 1, 2020 - Test Case Analysis

Analyzed `tf.test.TestCase`: *"Base class for tests that need to test TensorFlow."*

**Method analyzed**: `ClientTest.test_self_contained_example(self)`

Components:
* `@client_data`: Uses the method `create_client_data`
* `@model`: Retrieves the test model (needed to design a model for simulation)
* `@optimizer_function`: Standard keras optimizer (stochastic gradient descent)
* `@losses`: List to hold all losses from the clients
* Returns nothing - acts as the functional main

**The critical for loop**:
```python
for r in range(2):  # Creates sequence of numbers between 0 and n (where n=2)
    optimizer = optimizer_fn()  # Anonymous function usable only in this scope
    simple_fedavg_tff._initialize_optimizer_vars(model, optimizer)
    server_message = # Broadcasts model.weights and round number
    outputs = client_update  # Uses model, client data, server message, optimizer
    losses.append(outputs.model_output.numpy())
```

**Why these components?**
* **model_weights**: For aggregation
* **round number**: Needed for logging
* **optimizer from anonymous function**: Keeps value in local memory, accessible only in that scope
* **losses appended**: Outputs converted to numpy array

**My speculation at the time**: That anonymous function `lambda: tf.keras.optimizers.SGD()` may be used as a way to keep the value in local memory only accessible in that scope, in case there were to be the creation of that object elsewhere. However, not necessarily best practice - original developer short-cutting.

---

## Command-Line Flags for TFF Experiments

**Note**: Not all flags work with all experiments

```bash
--experiment_name=temp
--client_optimizer=sgd
--client_learning_rate=0.2
--server_optimizer=sgd
--server_learning_rate=1.0
--use_compression=True
--broadcast_quantization_bits=8
--aggregation_quantization_bits=8
--use_sparsity_in_aggregation=True
--total_rounds=200
--rounds_per_eval=1
--rounds_per_checkpoint=50
--rounds_per_profile=0
--root_output_dir=B:/projects/openProjects/githubprojects/BotnetTrafficAnalysisFederaedLearning/anomaly-detection/logs/fed_out/
```

---

## Implementation Details & Dependencies

### Environment Setup (May 2020 Notes)

* `download.py` can be modified to work - many libs were not needed or became outdated
* Dependencies are not simple to work with:
  * **TensorFlow Federated (as of May 2020)** still does not work with Python 3.8 (sure to change soon)
  * When working with code, be sure to either containerize with Docker or use conda environments
  * This makes working with code inside local IDEs (such as PyCharm) smoother
  * Most code should work right away with notebook runners such as Google Colab
  * Most/all testing done during modification was done in PyCharm (historical reasons)

**Conda environment recommendation**: Work with Python 3.7.x until notice about TensorFlow Federated working with 3.8+

**Anaconda Navigator (as of May 2020)**: Will not have `tensorflow_federated` within the repos
* Any setups will require you to use your pip terminal/cmd to install TFF
* Recommendation: Create your conda environment, then install TFF first thing
* This will install normal base TensorFlow as well and dependencies
* If using Google Colab you should be fine

**Device names**: Still work the same on input, but left the `sys.args` input as-is for the moment (historical reasons)

---

## PySyft Decision Log

### Why TensorFlow Federated was chosen over PySyft

**TFF was chosen** due to general compatibility with existing TensorFlow implementation.

**PySyft analysis (May 2020)**:
* PySyft, as of May 2020, still does not have a well-defined solution with TensorFlow
* `syft-tensorflow` is (as of writing this) very young and fresh - not enough examples to learn from effectively
* However, this may change sooner as updates appear to be frequent enough
* The main reason solutions such as TFF and Syft are not as popular: **need and age**
  * Not enough projects (independent, research, and public) exist with either material
  * Lack of demand exists to push development (on the surface this demand is low)
* **The alternative would be to convert the TF parts to PyTorch**, but... yeah, not doing that.

---

## Code Refactoring Notes

### Changes Made

* Lots of refactoring done to move closer to an **MVC (Model-View-Control)** framework
* Major changes will be for the TFF material
* Left the OG (original) file in place for reference

### Testing Status (June 1, 2020)

* Haven't touched testing YET
* Testing as of 6/1/2020 is done through `tf.test.TestCase`
* Testing under a federated lens will need to be reevaluated as another feature
* Each function of this botnet traffic analysis will have to be its own conversion project

---

## Files Referenced

Original draft files (now consolidated here):
* `README++.md` (root) - General FL concepts and update log
* `anomaly-detection/README++.MD` - Detailed TFF implementation notes
* `classification/README++.md` - Empty placeholder

These notes have been consolidated to preserve the development thought process and technical decisions made during the 2020 research period.

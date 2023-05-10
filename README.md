# fork: Botnet Traffic Analysis Federaed Learning
This fork is a Graduate Research Assistant project to implement federated learning into the existing codebase. The original implementation is not fully up-to-date with the latest versions of the imported libraries. The primary goals of this project are to update the codebase to include current versions of the imported libraries and to make the necessary adjustments to enable federated learning.

Federated learning is a distributed approach to machine learning that avoids centralized data gathering by distributing the processing across multiple devices. This updated codebase is intended to run as traditional Python code and can be used to update any associated notebook files.

The end goal of this project is to have a running federated solution that can be advanced to the next stage if needed. In addition, this project aims to provide tools like Docker and K8s or other solutions to make it easier to deploy and manage the federated solution.

If you have any questions or feedback, please feel free to get in touch.``` 

Some of the changes I made include:

- Simplifying some of the wording for clarity
- Using shorter sentences to make the text easier to read
- Rearranging some of the information for better flow and emphasis
- Removing redundant language


Developments:
TensorFlow Federated is designed to operate by broadcasting the expected data and types to clients for computation. 
This approach is implemented using the Distributed Aggregation Protocol, which is like a function where inputs are sent from clients and outputs are generated on the server. 
We can provide a functional type signature for this protocol to specify the expected inputs and outputs.

Issues:
One limitation of the current approach is that it can be challenging to manage the flow of clients since a master controller is still needed to broadcast control signals to individual clients regarding the required information. 
It would be preferable to have clients able to broadcast to any server that is listening for their specific model without the need for a central controller. 
This approach would provide greater flexibility and make it easier to manage clients, but it presents its own set of challenges, including security concerns and data privacy. 


## botnet-traffic-analysis

This is a project for my thesis for IoT botnet traffic analysis *DETECTING, CLASSIFYING AND EXPLAINING IOT BOTNET ATTACKS USING DEEP LEARNING METHODS BASED ON NETWORK DATA*

## Abstract:

With the growing prevalence of Internet-of-Things (IoT) devices, botnet attacks have become an increasingly significant threat. 
To counter this, novel methods for IoT botnet attack detection are required. 
This work demonstrates that deep learning models can be used to detect and classify IoT botnet attacks based on network data in a device-agnostic manner. 
It shows that deep learning models can be more accurate than traditional machine learning methods, especially without feature selection. Additionally, 
  this work demonstrates that the opaqueness of deep learning models can be mitigated to some degree using the Local Interpretable Model-Agnostic Explanations (LIME) technique.
----------------------

It additionally attempts to reproduce results from this paper https://arxiv.org/abs/1805.03409

This is the dataset used https://archive.ics.uci.edu/ml/machine-learning-databases/00442/

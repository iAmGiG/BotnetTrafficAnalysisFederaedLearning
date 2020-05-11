# fork: Botnet Traffic Analysis Federaed Learning
This fork is intended to implement a federed learning into this existing code, as apart of the Graduate research Assistant work. 
The origianl implementation is not fully uptodate with the latest versions of the imported libraries.
Some of the main goals include updateding the original code to include current versions of the import Libs as well as adjust for changes required to implement a federed learning solution. 
Federated learning is to have a distibution of the intended processing rather than data gathering to a centeralized solution. 
This updated code is intended to run as traditional python code, then updated any notebook files as needed.
The end goal is to have a running federed solution that could be brought to the next step if needed.
Additional goals are to have tools such as docker and K8s or other like solutions. 

developments: 
the core way that tensorflow federated is intended to work is to send clients, via broad-cast, the expected data and types to be computed.

issues:
this way does not make a flow of client first easy to approach. this is because you still need a master controller to boradcast controls to 
the individual clients the information that is required, rather than have the ablity for clients to broadcast to any server that is listening out for
there specific model. 

## botnet-traffic-analysis

This is a project for my thesis for IoT botnet traffic analysis *DETECTING, CLASSIFYING AND EXPLAINING IOT BOTNET ATTACKS USING DEEP LEARNING METHODS BASED ON NETWORK DATA*

Abstract:

The growing adoption of Internet-of-Things devices brings with it the increased participation of said devices in botnet attacks, and as such novel methods for IoT botnet attack detection are needed. This work demonstrates that deep learning models can be used to detect and classify IoT botnet attacks based on network data in a device agnostic way and that it can be more accurate than some more traditional machine learning methods, especially without feature selection. Furthermore, this works shows that the opaqueness of deep learning models can mitigated to some degree with Local Interpretable Model-Agnostic Explanations technique.

----------------------

It additionally attempts to reproduce results from this paper https://arxiv.org/abs/1805.03409

This is the dataset used https://archive.ics.uci.edu/ml/machine-learning-databases/00442/

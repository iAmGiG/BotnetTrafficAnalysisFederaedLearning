# README ++ botnet traffic analysis change log and updates (General)
## what is federated learning
* putting the computation on the device, 
*	where each device trains on its own model
* as there is not enough data on each device to truly train up a full model, the contributions of other devices 
	elevate this issue.
*	this is also mitigated by pretraining the model at the server level. on proxy data...
*		issue here is that as data becomes outdated, the model may not pick up the new data as well when raising issues.

## learning together
* start with server model.
* distributed this model to some of the clients.
* clients produce a new model, and then the model is pushed back to the server for aggregation. 
* the average model reflects the training from every model.
* then repeat for each round.
*	were a round being a session of learning

## TFF (TensorFlow federated)
* there exist interfaces to represent federated data sets.
* built out of FL core and FL api
### fed outline
    * training is representing as a part of computations
    * one the initialize state
    * two the single round execution
    * both can be executed like functions in python.
    * and when we do, they by default execute in a local simulation <------------------important for paper writing
    * and perform small simulation groups.
    * the state includes the model and the train data (both of which are above have above)
    * *at the time of this writing debugging as not begun fully mid May2020*
	
## the situation:
* take that we have clients that are mirroring the traffic data going through a security checkpoint
*	(in this case network signals going through a router to an IoT devices)
* all the collected network signals can be viewed as a single federated value.
* 	where we say a federated value is a multi-set
* in TFF the multi-set has types, the types consist of the identity of the devices that host the value(s)
* 	this is the placement, the client placement.
* on the server, in this case some global aggregation handler. 
 
## start
* broadcast to the clients through tff.federated_broadcast
* the first federated action.
* the task is then performed on the clients
* the clients feedback the tff.federated_average


# update log:
* managed to review some test code from the experimental/simulation section.
* this code will may produce simulation results 
* see anomaly-detection/simple_fedagv_test_revieing_edit.py 
* see anomaly-detection/README for details
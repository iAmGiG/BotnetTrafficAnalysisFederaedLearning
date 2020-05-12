# README ++ botnet traffic anaysis change log and updates (General)
## wwhat is federated learning
* putting the computation on the device, 
*	where each device trains on its own model
* as there is not enough data on each device to trully train up a full model, the contibutsions of other devices 
	eleviate this issue.
*	this is also mitigated by pretraining the model at the server level. on proxie data...
*		issue here is that as data becomes outdata, the model may not pick up the new data as well when raising issues.

## learning together
* start with server model.
* distirbuted this model to some of the clients.
* clients produce a new model, and then the model is pushed back to the server for aggigation. 
* the average model reflects the training from every model.
* then repeat for each round.
*	were a round is a session of learning

## TFF (tensorflow federated)
* there exist interfaces to represent federated data sets.
* built out of FL core and FL api
### fed outline
    * training is represent as a par of computations
    * one the initialize state
    * two the single round execution
    * both can be executed like functions in python.
    * and when we do they by default execute in a local simulation <------------------important for paper writing
    * and perform small simulation groups..
    * the state includes the model and the train data (both of which are above have above)
    * *at the time of this writing debugging as not begun fully mid May2020*
	
## the situation:
* take that we have clients that are mirroring the traffic data going through a security checkpoint
*	(in this case network signals going through a router to an IoT devices)
* all the collected network singals can be seen as a single federated value.
* 	where we say a federated values is a multi-set
* in TFF the multi-set has types, the types consist of the identity of the devices that host the value(s)
* 	this the placement, particular the client placement.
* on the server, in this case a some gobal aggigation handelar. 
* 
## start
* broadcast to the clients through tff.federated_broadcast
* the first federated action.
* the task is then perfromed on the clients
* the clients feed back the tff.federated_average
* 
# This read me details many of the upgrades that have occurred during the implemetnation oftensorflow federated,
## includes some details on pysyft-tensorflow and what the end plan is intended to be.
### includes running details that have/where updated from the original README in THIS folder.
* the original README was becoming data and many updates to the code had to be made.
* terms:
	* TFF tensorflow_federated
	* TF tensorflow
	* I = [fork author GIGA]

# training: 
details on training.
###
## updates
* the download.py can be modified to work, the issues were many of the libs where not needed
	or became outdated. more details in another README file(maybe)
* the dependencies are not simple to work with.
	* tensorflow federated (as of May 2020) still does not work with python 3.8 (sure to change soon(maybe))
	* when working with any of the code, be sure to either containerize with tools such as docker or use conda environments
		* this will make working with the code inside local IDE's (such as PyCharm) smother
	* most of the code should work right away with notebook runners such as google colab.
	* but most to all the testing done during modification was done in PyCharm (historical raisins)
* when working with environments such as conda environs, be sure to work with 3.7.x until notice about TensorFlow federated working with 3.7 < [beyond]
* more on conda: anaconda navigator (as of May 2020) will not have tensorflow_federated within the repos,
	* so, any set ups will require you to use your pip terminal/cmd to install TFF
	*	I recommend just creating your conda environment(s) and then installing TFF first thing, 
	* 	this will install normal base tensorflow as well and dependencies (again do this from the environ terminal if using local IDE)
	* 	if you are using google co-lab you should be fine.
* device names will still work the same on input, but I left the sys.args input as is for the moment (historical raisins)

** log 5/28/2020: digging through the experiment/research section of tff federated as revealed simulation experiments
*   this simulation will reflect the intended results of federated learning without the need to set up
*   an entire product pipeline to run single experiments.
*   this WILL BE DETAILED in the corresponding implementation as production and simulation are not the same and the intended 
*   results may differ. 
*   TO MAIN NOTE THIS: this code is marked as an unit test, as it runs as a unit test callable to quick answer what its doing


** log 6/1/2020: class ClientTest(tf.test.TestCase):
*  tf.test.TestCase: """Base class for tests that need to test TensorFlow.""" as per doc
*  method: of clientTest: test_self_contained_example(self):
*  this methods is a simple tester that primarily runs a built in test case under tensorflow
*    """
        @client_data: uses the method creat client data, see for details
        @model: retrieves the test model, this will be needed as a way to break down how to design a model for simulation
        @optimizer_function: is a standard keras optimizer, in this case a stochastic gradient decent
        @losses: this list appears to be the list to hold all losses from the clients
        :return: this method never needs to return, it acts as the functional main
    """
*   the for loop:
*   for r in range(2):
*   '''
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
    * the optimizer is retrieved from the anonymous function optimizer_fn().
    * the server message would appear to send the model_weights and the round number, this from the client perspective
    * helps with logging and aggregation.
    * the output .....
    * the losses are appended to the list, and are the ouptuts.model_output.numpy() array
* as will be detailed in changes: I will need to adjust teh create_client_data, re-focus the model making and ajust the 
* SDG value: as this is the value I already am closest, I get a better understand of why to use that anonamous function
* only speculation: that anonymous function (lambda: tf.keras.optimizers.SGD()) may be used as way to keep the value in local memory only 
* accessible in that scope, in case there were to be the creation of that object else where, will however, look
* into switching out that anonymous function for a standard function, as that anonymous function isn't nessesariy best 
* practice but rather the original developer sort cutting.  

## changes
* lots of refactoring done to move closer to an MVC (model view control) framework.
* the major changes will be for the TFF material and left the OG (original) file in place for reference.


## tensorflow details
* TFF was chosen as of the general compatibility with existing tensorflow implementation.
* yep.

## pysyft log
* pysyft, as of (May 2020) still does not have a well-defined solution with TF.
* syft-tensorflow is (as of writing this) every young and fresh meaning not enough examples to learn from effectily.
* however, it would seam this may change sooner as updates appear to be frequent enough and as more updates come 
* the chances of working implementation might be possible. 
* the main reason solutions such as TFF and syft are not as popular is the need and age
* not enough projects (independent, research and public) is out there with either material, along with the lack of demand exist to push development (on the surface this demand is low)
* the alternative would be to convert the TF parts to pytorch, but that... yeah not doing that.


# testing:
* haven't touched it YET.*
* it testing as of 6/1/2020 is done though the tf.test.TestCase, as the testing under a federated lenses
* will need to be reevaluated as another feature, 
* this also includes each function of this botnet traffic analysis, will have to be its own conversion project.


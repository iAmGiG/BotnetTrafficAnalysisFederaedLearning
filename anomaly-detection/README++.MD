# This read me details many of the upgrades that have occurred during the implemetnation oftensorflow federated,
## includes some details on pysyft-tensorflow and what the end plan is intended to be.
### includes running details that have/where updated from the original README in THIS folder.
* the original README was becomeing data and many updates to the code had to be made.
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
	* 	if you are using google colab you should be fine.
* device names will still work the same on input, but I left the sys.args input as is for the moment (historical raisins)

## changes
* lots of refactoring done to move closer to an MVC (model view control) framework.
* the major changes will be for the TFF material and left the OG (original) file in place for reference.


## tensorflow details
* TFF was chosen as of the general compatibility with existing tensorflow implementation.
* yep.

## pysyft log
* pysyft, as of (May 2020) still does not have a well-defined solution with TF.
* syft-tensorflow is (as of writing this) every young and fresh.
* the main reason solutions such as TFF and syft are not as popular is the need and age
* not enough projects (independent, research and public) is out there with either material, along with the lack of demand exist to push development (on the surface this demand is low)
* the alternative would be to convert the TF parts to pytorch, but that... yeah not doing that.


# testing:
*haven't touched it YET.*

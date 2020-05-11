# This read me details many of the upgrades that have occured during the upgrades to tensorflow fedeated,
## inclues some details on pysyft-tensorflow and what happend
### inclues runningdetails that have/where updated from the original README in THIS folder.
* the original README was outdated durning the intial fork of the repo, and many updates to the code had to be made.
*** TFF tensorflow_federated
*** TF tensorflow

# training: 
### details on training.
## updates
* the download.py can be modified to work, the issues was many of the libs where not needed
	or became outdated. more details in another README file(maybe)
* the dependecies are not simple to work with.
	** tensorflow fedeated (as of May 2020) still does not work with python 3.8 (sure to change soon(maybe))
	* when working with any of the code, be sure to either contianerize with tools such as docker or use conda environments
		this  will make working with the code inside local IDE's (such as pycharm) smother
	* most of the code should work right away with notbook runners such as google colab.
	** but most to all of the testing done during modification was done in pycharm (historical raisins)
* when working with environments such as conda envions, besure to work with 3.7.x untill notic about tensorflo fedeated is working with 3.7< [beyond]
* more on conda: anaconda navigator (as of may 2020) will not have tensorflow_federated within the repos,
	** so any set ups will require you to use your pip terminal/cmd to install TFF
	*	GIGA: recomend just creating your conda environ and then installing TFF first thing, 
	* 	this will install normal base tensorflow as well and dependecies (again do this from the envorn terminal if using local IDE)
	* 	if you are using google colab you should be fine.
* device names will still work the same on input, but GIGA left the sys.args input as is for the moment(historical raisins)

## changes
** lots of refactoring done to move clsoer to an MVC (model view control) framework.
* the major changes will be for the TFF material, and left the OG (original) file in place for reference.


## tensorflow details
* TFF was chosen as of the genral compatability with exsiting tensorflow implementation.
* yep.

## pysyft log
* pysyft, as of (May 2020) still does not have a well defined solution with TF.
* syf-tensorflow is (as of writing this) every young and fresh.
* the main reason solutions such as TFF and syft are not as popoulat is the need and age
* not enough is outthere with either material, and not enough demand exist to push development (on the surface this demand is low)
* the alternative would be to convert the TF parts to pytorch, but that... yeah not doing that.


# testing:
*** haven't touched it YET.

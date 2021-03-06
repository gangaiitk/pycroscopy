.. contents::

v 1.0 goals
-----------
1. mostly done - reogranize to have a .core submodule with the core functionality, .contrib to have non-verified code

  * This is perhaps the last oppurtunity for major restructuring and renaming
  * hdf_utils is becoming very big and all the functions deal with h5 in some form whether it is for reading or writing. Perhaps it should be split into read_utils and write_utils? hdf is implied.
  * Think about whether the rest of the code should be organized by instrument
    * One possible strategy - .core, .process (science independent), .instrument?. For example px.instrument.AFM.BE would contain translators under a .translators, the two analysis modules and accompanying functions under .analysis and visualization utilities under a .viz submodule. The problem with this is that users may find this needlessly complicated. Retaining existing package structure means that all the modalities are mixed in .analysis, .translators and .viz. 
2. mostly done - Make core as robust as possible with type / value checking, raising exceptions. 
3. mostly done - unit tests for .core io

 * measure coverage using codecov.io and codecov package
4. mostly done - good utilities for interrogating data - pycro data, what about the rest of the file?
5. partly done - good documentation for both users and developers

  * Need one per module in .core + finish plot_utils tour
  * (for developers) explaining what is where and why + io utils + hdf utils tour etc.
  * Need to add the ability to swap out the data for user provided data in the examples - Eg FFT filtering
  * Need to add statement - shift + enter to advance to next cell / link to jupyter notebook operation within each notebook
  * Upload clean exports of paper notebooks
  * comprehensive getting started page that will point everyone towards all necessary prerequisites including python, data analytics, jupyter, pycharm, git, etc.
  
6. DONE - generic visualizer - we now have something that can visualize up to 4D datasets reliably.
7. mostly done - good utils for generating publishable plots - make certain functions more generic / extendable - easy < 1 day
8. DONE - Fitter must absorb new features in Process if it is not possible to extend it
9. Examples within docs for popular functions <-- just use the examples from the tests!
10. almost done - a single function that will take numpy arrays to create main and ancillary datasets in the HDF5 file and link everything.  
 
  * Allow the user to specify an empty dataset - this will become very handy for all Processes and Analysis classes. This will mean that we cannot check to see if the sizes of the said dimensions in the descriptors / h5 ancilllary datasets match with the data dimensions. 
11. Restructure Process such that:
  * It allows the user to apply the process to a single unit of the data to tweak the parameters
  * This requires a new function to update the parameters and does whatever init already does
  * The compute should be detached from writing in Cluster, Decomposition etc. If the results are not satisfactory, discard them, change parameters and try again until one is happy with the results at which point the write results can be manually called.
12. Lower the communication barrier by starting a twitter account - Rama?
13. file dialog for Jupyter not working on Mac OS
14. Get pycroscopy on Anaconda / conda installation (to include other packages like opencv)
15. Test all translators, Processes and Analyses to make sure they still work.

v 1.1 goals
-----------
1. Deploy on cluster - consider MPI4py or ipyparallel....

Documentation
-------------

Fundamental tutorials on how to use pycroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* A tour of what is where and why
* A tour of all utils in core.io at the very minimum:
  
  * hdf_utils: paartially done - functions used for writing h5 files since these functions need data to show / explain them - chunking the main dataset
  * io_utils: DONE
  * dtype_utils : in progress
  * hdf_writer + VirtualData : we already have something. Needs to be updated
  * Numpy translator : Done in some way
  * write_utils: Not done
* How to write your own analysis class based on the (to-be simplified) Model class

Rama's (older and more applied / specific) tutorial goals
~~~~~~~~~~~~~~~~~~~~
1. Open a translated and fitted FORC-PFM file, and plot the SHO Fit from cycle k corresponding to voltage p, along with the raw spectrogram for that location and the SHO guess. Plot both real and imaginary, and do so for both on and off-field.
2. Continuing above, determine the average of the quality factor coming from cycles 1,3,4 for spatial points stored in vector b for the on-field part for a predetermined voltage range given by endpoints [e,f]. Compare the results with the SHO guess and fit for the quality factor.
3. After opening a h5 file containing results from a relaxation experiment, plot the response at a particular point and voltage, run exponential fitting and then store the results of the fit in the same h5 file using iohdf and/or numpy translators.
4. Take a FORC IV ESM dataset and break it up into forward and reverse branches, along with positive and negative branches. Do correlation analysis between PFM and IV for different branches and store the results in the file, and readily access them for plotting again.
5. A guide to using the model fitter for parallel fitting of numpy array-style datasets. This one can be merged with number 

New features
------------
Core development
~~~~~~~~~~~~~~~~
* function for saving sub-tree to new h5 file
* Windows compatible function for deleting sub-tree
* Chris - Demystify analyis / optimize. Use parallel_compute instead of optimize and guess_methods and fit_methods
* Chris - Image Processing must be a subclass of Process and implement resuming of computation and checking for old (both already handled quite well in Process itself)
* Consistency in the naming of and placement of attributes (chan or meas group) in all translators - Some put attributes in the measurement level, some in the channel level! hyperspy appears to create datagroups solely for the purpose of organizing metadata in a tree structure! 

Long-term
^^^^^^^^^
* A sister package with the base labview subvis that enable writing pycroscopy compatible hdf5 files. The actual acquisition can be ignored.
* multi-node computing capability in parallel_compute
* Intelligent method (using timing) to ensure that process and Fitter compute over small chunks and write to file periodically. Alternatively expose number of positions to user and provide intelligent guess by default
* Consider developing a generic curve fitting class a la `hyperspy <http://nbviewer.jupyter.org/github/hyperspy/hyperspy-demos/blob/master/Fitting_tutorial.ipynb>`_

GUI
~~~~~~~~~~~
*	Convert all existing notebooks to interactive plotting

Plot Utils
~~~~~~~~~
* move plot_image_cleaning_results to a application specific module
* move save_fig_filebox_button and export_fig_data to jupyter_utils
* ensure most of these functions result in publication-ready plots (good proportions, font sizes, etc.)
* allow setting of c-axis limits for all plot utils functions
* plot_map 

  1. allow the tick labels to be specified instead of just the x_size and y_size. 

* plot_loops
 
  1. Legend at the bottom
  
* plot_map_stack:

  1. Add ability to manually specify x and y tick labels - see plot_cluster_results_together for inspiration
  2. See all other changes that were made for the image cleaning paper

* plot_cluster_results_together

  1. Use plot_map and its cleaner color bar option
  2. Option to use a color bar for the centroids instead of a legend - especially if number of clusters > 7
  3. See G-mode IV paper to see other changes

* plot_cluster_results_separate
  
  1. Use same guidelines as above

* plot_cluster_dendrogram - this function has not worked recently to my knowledge. Fortunately, it is not one of the more popular functions so it gets low priority for now. Use inspiration from image cleaning paper

* plot_histograms - not used frequently. Can be ignored for this pass

External user contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Incorporate sliding FFT into pycroscopy
* Create an IR analysis notebook
* Li Xin classification code 
* Ondrej Dyck’s atom finding code – written well but needs to work on images with different kinds of atoms
* Nina Wisinger’s processing code (Tselev) – in progress
* Sabine Neumeyer's cKPFM code
* Iaroslav Gaponenko's Distort correct code from - https://github.com/paruch-group/distortcorrect.
* Port everything from IFIM Matlab -> Python translation exercises
* Other workflows/functions that already exist as scripts or notebooks

Formatting changes
------------------
*	Fix remaining PEP8 problems
*	Ensure code and documentation is standardized

Testing
-------
*	Write test code for scientific functions in addition to just core
*	Longer tests using data (real or generated) for the workflow tests

Software Engineering
--------------------
* Consider releasing bug fixes (to onsite CNMS users) via git instead of rapid pypi releases 
   * example release steps (incl. git tagging): https://github.com/cesium-ml/cesium/blob/master/RELEASE.txt
* Use https://docs.pytest.org/en/latest/ instead of nose (nose is no longer maintained)
* Add requirements.txt
* Consider facilitating conda installation in addition to pypi

Scaling to clusters
-------------------
We have two kinds of large computational jobs and one kind of large I/O job:

* I/O - reading and writing large amounts of data
   * Dask and MPI are compatible. Spark is probably not
* Computation
   1. Machine learning and Statistics
   
      1.1. Use custom algorithms developed for BEAM
         * Advantage - Optimized (and tested) for various HPC environments
         * Disadvantages:
            * Need to integarate non-python code
            * We only have a handful of these. NOT future compatible            
      1.2. OR continue using a single FAT node for these jobs
         * Advantages:
            * No optimization required
            * Continue using the same scikit learn packages
         * Disadvantage - Is not optimized for HPC
       1.3. OR use pbdR / write pbdPy (wrappers around pbdR)
         * Advantages:
            * Already optimized / mature project
            * In-house project (good support) 
         * Disadvantages:
            * Dependant on pbdR for implementing new algorithms
            
   2. Parallel parametric search - analyze subpackage and some user defined functions in processing. Can be extended using:
   
      * Dask - An inplace replacement of multiprocessing will work on laptops and clusters. More elegant and easier to write and maintain compared to MPI at the cost of efficiency
         * simple dask netcdf example: http://matthewrocklin.com/blog/work/2016/02/26/dask-distributed-part-3
      * MPI - Need alternatives to Optimize / Process classes - Better efficiency but a pain to implement

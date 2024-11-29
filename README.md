===========
BUILD & RUN
===========
. These programs have all been written on a Linux computer and not tested elsewhere
. Python used a conda environment when running the programs
. The following commands should be run from the .../main_program/ folder


. To compile the cython files into C, the setup_* files must be run with;
        python setup_* build_ext -fi
. Alternatively, I have a made a bash script "build_all.sh" that can be run to build all (openMP and MPI) setup files too;
        bash build_all.sh


. Then, the corresponding program can be run using the command;
        python barneshut_core_OpenMP_py.py <mode_number> <thread_number> <particle_number> <iteration_number>
OR
        mpirun -np <ranks> python barneshut_core_MPI_py.py <mode_number> <particle_number> <iteration_number>
. This run will either the program that implements the OpenMP or MPI approach (and the Hybrid approach is run in the MPI version too for modes implementing the "main_mixed()" method)

. Mode 0 for both the OpenMP and MPI programs performs the main simulation

=============
VISUALISATION
=============
. Visualised particle data shown in the report is all generated using the code I have written in the language 'Processing', available at https://processing.org/
. The particle data is stored in a file 'data_info.txt' after running
. This is read by the Processing file to draw particles at the correct positions (Note** must be copied from the main_program folder to the root of the sketch folder)
. The Processing file (.pde) can be run by opening the 'octree_core_visualiser.pde' file through the Processing.exe editor and running the sketch
. The animated frame can be then be stepped through using keys '1' & '2', and started and reset using the keys '4' & '5'
. The "quadtree_experimental_visualiser" and "octree_experimental_visualiser" programs are also included which were used to initially test the tree generation 
    and provide two figures used in the report showing the 2D and 3D tree structure. The "void key_pressed()" functions inside the program can be read to infer 
    some of the simple controls used.


=================
PROGRAM STRUCTURE
=================
The following files are included in this project;

.   "main_program" folder
.       "BC4_sample_jobs" folder
.           Contains a series of sbatch scripts used in BC4 for some of the data collection (often these scripts were simply overwritten to get specific plots)
.       data_info.txt file containing particle information
.       Core progam for both OpenMP and MPI in a python .py script
.           OpenMP calculations functions in a cythonised .pyx script
.           OpenMP setup/build script in as a .py files
.       MPI calculations functions in a cythonised .pyx script
.       MPI setup/build script in as a .py files
.           The "output_data_reader.py" program used to produce the plots used in the report (however, this file reads specifically generated data files and produces 
            specific plots according to an ID system, and so will not be usable without this original data)

.   "visualising_programs" folder
.       "octree_core_visualiser" sketch folder
.       "octree_experimetnal_visualiser" sketch folder
.       "quadtree_experimental_visualiser" sketch folder

. As well as the approporiate compiled .c scripts and additional files generated after being built
. There is also a "build_all.sh" bash script to run all "setup_*" files in the folder


================
ADDITIONAL NOTES
================
. This folder has been tested being built prior to submission

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

updated version of C/C++ compiler
CMake v3.4 with support for FindCUDA module
CUDA Toolkit version >= 7
NVIDIA Kepler or Maxwell GPU

--------------------------------------------------------------------------------
DOWNLOAD
--------------------------------------------------------------------------------
if necessary create a ssh key
https://confluence.atlassian.com/bitbucket/set-up-ssh-for-git-728138079.html

$git clone git@bitbucket.org:federico_busato/appagato.git

or download the package directly from bitbucket website
--------------------------------------------------------------------------------
COMPILE
--------------------------------------------------------------------------------

$ cd appagato
$ mkdir build
$ cd build
$ cmake [ -DARCH=<your_GPU_compute_cabability> ] ..
$ make -j

example:
$ cd build
$ cmake -DARCH=35 ..
$ make

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

$ ./SeqAppagato_info <TARGET_GRAPH> <QUERY_GRAPH> <NUMBER_OF_SOLUTIONS> [ Label_Similarity_Matrix ]

example:
$cd build
$./SeqAppagato_info ../Example_PPI/Homo_sapiens_43.gfu ../Example_PPI/Homo_sapiens_43_32V.gfu 100

config.h	-> advanced APPAGATO Configuration

--------------------------------------------------------------------------------
DOCUMENTATION
--------------------------------------------------------------------------------

For other information see Documentation.pdf in Documentation directory
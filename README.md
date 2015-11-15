# hand-pose-estimation
hand pose estimation system based on [Qian et al., 2014]. This system is implemented in Windows; however, it can also be compiled in Linux.

#DEPENDENCIES:
1. MinGW (g++ 4.8.0)
2. OpenCV
3. Armadillo (with OpenBLAS)
4. OpenGL

## MinGW Installation:
Download mingw-64 builds from [here](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/rubenvb/) then install. It is recommended to install mingw at `C:\mingw`. Remember to add `MINGW_INSTALL_DIR\bin` to the `PATH` environment variable on Windows. g++ V4.8.0 is used as our compiler, but later version should also work.

It is also recommended to install git for windows, which can be downloaded from [here](https://git-scm.com/download/win). After the installation, git-bash should appear in the start menu. Link git-bash with mingw by adding `export PATH=MINGW_INSTALL_DIR/bin:$PATH` to the file `GIT_INSTALL_DIR\etc\.bashrc`. Check this is working by opening a new git-bash shell and calling `g++ --v`.

## OpenCV Installation:
OpenCV **must be compiled from source**. Downloading precompiled binaries most likely will not link with our compiler (i.e. g++ V4.8.0). 

To compile OpenCV, CMake is needed. Download binary from [here](https://cmake.org/download/) and install. No need to add `CMAKE_INSTALL_DIR\bin` to `PATH`. 

Then, clone the OpenCV source by running `git clone git://github.com/Itseez/opencv.git` in a git-bash shell. This may take a while depending on internet speed. 

Once this is done, follow [these]() instructions to configure CMake. 

#DATA:
The data set can be found [here]( http://research.microsoft.com/en-us/um/people/yichenw/handtracking/index.html). Subject1 data is used in our system. Place the depth map files in `../handModelling/Release_2014_5_28/Subject1/` 

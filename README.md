# hand-pose-estimation
hand pose estimation system based on [Qian et al., 2014](http://www.google.com.au/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0CCEQFjAAahUKEwiOhKqHhZTJAhVkKqYKHSkVAuE&url=http%3A%2F%2Fresearch.microsoft.com%2Fen-US%2Fpeople%2Fyichenw%2Fcvpr14_handtracking.pdf&usg=AFQjCNH1T1513mdzrW9SM363UUg7ZqnJXg&bvm=bv.107467506,d.dGY). which uses single depth maps. This system is implemented in Windows; however, it can also be compiled in Linux.


#DEPENDENCIES:
1. MinGW (g++ 4.8.0)
2. OpenCV 3.0
3. Armadillo (with OpenBLAS)
4. OpenGL

## MinGW Installation:
Download mingw-64 builds from [here](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/rubenvb/) then install. It is recommended to install mingw at a short directory without spaces in its path, e.g. `C:\mingw`. Remember to add `MINGW_INSTALL_DIR\bin` to the `PATH` environment variable on Windows. Refer to [these notes](http://www.ntu.edu.sg/home/ehchua/programming/howto/Cygwin_HowTo.html) for more detailed instructions. g++ V4.8.0 is used as our compiler, but later version should also work. 

It is also recommended to install git for windows (e.g. in `C:\mingw`), which can be downloaded from [here](https://git-scm.com/download/win). After the installation, git-bash should appear in the start menu. Link git-bash with mingw by adding `export PATH=MINGW_INSTALL_DIR/bin:$PATH` to the file `GIT_INSTALL_DIR\etc\bash.bashrc`. Check this is working by opening a new git-bash shell and calling `g++ --v`.

## OpenCV Installation:
OpenCV **must be compiled from source**. Downloading precompiled binaries most likely will not link with the mingw compiler (i.e. g++ V4.8.0). 

To compile OpenCV, CMake is needed. Download binary from [here](https://cmake.org/download/) and install. No need to add `CMAKE_INSTALL_DIR\bin` to `PATH`. Eigen (a linear algebra library) is also needed. This is a header only library; hence to install, simply [download source](http://eigen.tuxfamily.org/index.php?title=Main_Page) and extract.

Then, clone the OpenCV source by running `git clone git://github.com/Itseez/opencv.git` in a git-bash shell. This may take a while depending on internet speed. 

Once this is done, follow [these instructions](http://perso.uclouvain.be/allan.barrea/opencv/cmake_config.html) to configure CMake. Take note that:

1. Instead of choosing `Visual Studio ...` as generate, use `Eclipse mingw makefiles`. Then, `specify native compilers` by pointing to `gcc`, `g++` and `gfortran`, which are in `MINGW_INSTALL_DIR\bin`.
2. IMPORTANT: do not select `with Qt`
3. do not select `build_opencv_python`
4. TBB is optional; if you want to include TBB, then refer to their instructions
5. IMPORTANT: set `CMAKE_BUILD_TYPE` to `Release`
6. change `CMAKE_INSTALL_PATH`; (I set this to `MINGW_INSTALL_DIR\opencv` so everything is in one parent directory)

After the configuration and the makefiles are generated, then open git-bash at the directory where you specified to build the binaries. Then execute:
```
mingw32-make
```
This will take a while. Once it's finished without throwing any errors, then run:
```
mingw32-make install
```
Which will install OpenCV to `CMAKE_INSTALL_DIR`. Finally, remember to include `OPENCV_INSTALL_DIR\bin` to `PATH`.

## Armadillo Installation:
Armadillo is also a header only linear algebra library that provides *MATLAB*-like syntaxes. Here, I will run through instructions for installing Armadillo with OpenBLAS, which is a fast BLAS library. 

1. Compile OpenBLAS to `MINGW_INSTALL_DIR` by following instructions [here](https://github.com/xianyi/OpenBLAS)
2. Compile Lapack to `MINGW_INSTALL_DIR` by following instructions [here], CMake is needed; and it is better to compile using the same g++ than downloading the precompiled binaries
3. Download and extract aramdillo source from [here](http://arma.sourceforge.net/download.html)
4. Setup CMake using the same process as for OpenCV; make sure that CMake can find LAPACK and OpenBLAS
5. Compile and install using `mingw32-make`

Test that the installation is fine by running `ARMADILLO_SOURCE/examples/example1.cpp`. The instructions to compile this example can be found in the README file.

**IMPORTANT:** our implementation can be parallelised on runtime by enabling OpenMP on runtime. This will conflict with OpenBLAS's multithreading and cause errors. To avoid this, add the environment variable `OPENBLAS_NUM_THREADS` and set to 1. 

## OpenGL Installation:
In mingw we need to install GLUT separately. Detailed instructions can be found [here](http://www.ntu.edu.sg/home/ehchua/programming/opengl/HowTo_OpenGL_C.html).

## IDE Setup:
I used Eclipse for development. To install Eclipse and setup for C++, follow [these instructions](http://www.ntu.edu.sg/home/ehchua/programming/howto/EclipseCpp_HowTo.html). 

#DATA:
The data set can be found [here]( http://research.microsoft.com/en-us/um/people/yichenw/handtracking/index.html). Subject1 data is used in our system. Place the depth map files in `../handModelling/Release_2014_5_28/Subject1/`. Alternatively, modify the search paths in `testmodel.cpp` and `visualiser.cpp`.

The hand geometries (i.e. joint lengths and hand spheres radii) are predefined and they are stored in the .bat files in `\misc`. So, using another subject will likely degrade performance as subjects' hand geometries are different.

# COMPILE:
Firstly, ensure that all dependencies are installed and the test data is in the proper directory.

Then after cloning this repo and importing it to eclipse, setup project properties by:

1. right click on project and go to `Properties`
2. expand `C\C++ Build` and select `Settings`
3. **OPTIONAL**: in `GCC C++ COMPILER` *and* `MinGW C++ Linker` add to the Command box `-fopenmp` to enable parallelism (make sure then to turn off OpenBLAS's multithreading)
4. in `Includes` under `GCC C++ COMPILER`, add the include paths -I for *ALL* installed dependencies (since I installed everything in `MINGW_INSTALL_DIR`, then I only needed to add `MINGW_INSTALL_DIR\include`)
5. **OPTIONAL**: in `Optimisation` under `GCC C++ COMPILER` set the optimisation level
6. in `Libraries` under `MinGW C++ Linker`, add to Libraries -l:
    1. armadillo
    2. openblas
    3. lapack
    4. opencv_coreXYZ (where XYZ is the version of OpenCV you installed)
    5. opencv_imgprocXYZ
    6. opencv_flannXYZ
    7. opencv_features2dXYZ
    8. opencv_highguiXYZ
    9. opencv_imgcodecsXYZ
    10. opengl32
    11. glu32
    12. freeglut
7. in `Libraries` under `MinGW C++ Linker`, add the lib paths of *all* installed dependencies to Library search path -L (again, since I have installed everything under `MINGW_INSTALL_DIR`, I only needed to add `MINGW_INSTALL_DIR\lib`)
8. finally build project and run as local C++ application; a few different tests are available in `testmodel.cpp`


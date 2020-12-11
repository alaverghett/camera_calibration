# Camera Calibration

The program estimates 3D locations from 2D images of two captured camera views, using Tsai's calibration method.

The project requires python 3.x and MatLab to run. The MatLab API for python must also be installed for the python you want to use, see this guide for installation details:

https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

The implementation of Tsai's method comes from the repository below, clone this repo, and copy the code here before running:

https://github.com/simonwan1980/Tsai-Camera-Calibration/tree/master/Tsai

Use this command to run the script: ```python calibrate.py```

Python will call the MatLab engine during execution; it will read its input data from meaursedRaw.txt and pixelsRaw.txt, which contain the world and pixel coordinates of the key points.
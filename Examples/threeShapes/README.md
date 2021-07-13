# Basic shape detection

Data sets are generated using the ``genData.py <n> <m>`` script which generates
``m``  binary images of size ``n x n`` on the standard output.

The binary images are classified in three categories~:

 - straight lines (first output),
 - rectangle aligned with the axes (second output),
 - triangle (third output),
 - random points (no output).

These images may be visualized using the ``displayData.py <filename>`` script.

An example of training data is provided in the ``data`` folder. Neural Networks
have been trianed using this data set and are available in the ``trainedNN``
folder.



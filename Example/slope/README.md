# Basic slope classification

Data sets are generated using the ``genData.py <n> <m>`` script which generates
``m``  binary images of size ``n x n`` on the standard output.

Each binary image contains a sigle line segment. These segments are classified
according to their slope : 

 - less than -1,
 - between -1 and 0,
 - between 0 and 1,
 - more than 1.

These images may be visualized using the ``displayData.py <filename>`` script.

An example of training data is provided in the ``data`` folder. Neural Networks
have been trianed using this data set and are available in the ``trainedNN``
folder.

```
$ ./gui -nn ../Example/slope/trainedNN/900_20_20_20_4
```



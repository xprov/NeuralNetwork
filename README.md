# NeuralNetwork
Simple Back Propagation Neural Network

This code is a fork from [Bobby Anguelov's NeuralNetwork](https://github.com/BobbyAnguelov/NeuralNetwork).

The example code makes use of [Florian Rappl's command parser](https://github.com/FlorianRappl/CmdParser )

# Disclaimer
This code is meant to be a simple implementation of the back-propagation neural network discussed in the tutorial below:

[https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/](https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/)

[https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/](https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/)

It is intended as a reference/example implementation and will not be maintained or supported.


# Compilation
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Alternatively, since one might find useful to use debug/release versions:
```
$ mkdir debug && cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j
$ cd ..
$ mkdir release && cd release && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```



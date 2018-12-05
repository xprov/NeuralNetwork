# NeuralNetwork
Simple Back Propagation Neural Network

This code is a fork from [Bobby Anguelov's NeuralNetwork](https://github.com/BobbyAnguelov/NeuralNetwork).

The example code makes use of [Florian Rappl's command parser](https://github.com/FlorianRappl/CmdParser )

# Disclaimer
This code is meant to be a simple implementation of the back-propagation neural network discussed in the tutorial below:

[https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/](https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/)

[https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/](https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/)

It is intended as a reference/example implementation and will not be maintained or supported.

# Required Packages

This project contains two programs, a BPN trainer and a graphic interface to
play with a trained BPN. While the trainer requires nothing more than a C++11
compiler, the GUI requires :

 - GTK+2.x or higher, including headers (libgtk2.0-dev)
 - GooCanvas

Using Synaptic Manager, the simplest way to install these dependencies is to run:
```
sudo apt-get install libgtk2.0-dev libgoocanvas-dev
```


# Compilation
```
mkdir build
cd build
cmake ..
make
```

Alternatively, since one might find useful to use debug/release versions:
```
mkdir debug && cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j
cd ..
mkdir release && cd release && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

# Typical usage
Let's train a Network using data from the
``Example/threeShapes/data/threeShapes.csv`` file. In this example, the input
are binary pictures of size 40x40. Consequently, there needs to be 1600 input
neurons on the first layer. The output layer must have size 3 since datas
are classified in three categories :

1. Straight line (first output neuron).
2. Rectangle (second output neuron).
3. Triangles (third output neuron).

Here we choose to have 3 hidden layers of size 20, so the "-l" argument is
"1600,20,20,20,3". Finally, we stop the learning once our network has obtain at
least 90% accuracy on the genaralization set.

```
./trainBPN -d ../Example/threeShapes/data/threeShapes.csv -l 1600,20,20,20,3 -a 99 -e myTrainedNN
Input file: ../Example/threeShapes/data/threeShapes.csv
Read complete: 100000 inputs loaded

 Neural Network Training Starting: 
==========================================================================
 LR: 0.01, Momentum: 0.9, Max Epochs: 100
 1600 Input Neurons, 3 Hidden Layers, 3 Output Neurons
==========================================================================

Epoch :0 Training Set Accuracy:25.0967%, MSE: 0.187347 Generalization Set Accuracy:25.62%, MSE: 0.182734
Epoch :1 Training Set Accuracy:26.87%, MSE: 0.159222 Generalization Set Accuracy:25.725%, MSE: 0.143087
Epoch :2 Training Set Accuracy:24.225%, MSE: 0.137596 Generalization Set Accuracy:23.925%, MSE: 0.131987
Epoch :3 Training Set Accuracy:24.0633%, MSE: 0.128191 Generalization Set Accuracy:26.735%, MSE: 0.122087
Epoch :4 Training Set Accuracy:32.175%, MSE: 0.115479 Generalization Set Accuracy:41.26%, MSE: 0.10896
Epoch :5 Training Set Accuracy:44.265%, MSE: 0.100699 Generalization Set Accuracy:57.36%, MSE: 0.0884957
[...]
```

Note that in the above example, the trained network is exported to the file ``myTrainedNN``.

To visualize the Network in action, use the ``gui`` program which reads the
export of a Neural Network on the standard input.
```
./gui -nn myTrainedNN -l Line,Rectangle,Triangle
```
The grid on the left shows the state of the input neurons. Click on a square to
change it's value. The neurons layers are displayed to the right. Last column
is the ouput layer. Labels for the output neurons may be specified using the
``-l`` argument.

Here are some examples : 

 - A line ; the first output neuron is active.
![line detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectLine.png)

 - A rectangle ; the second output neuron is active.
![rectangle detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectRectangle.png)

 - A triangle ; the third output neuron is active.
![triangle detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectTriangle.png)






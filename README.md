
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


# Install Dependencies, Compile and Run Under Linux

Using Synaptic Manager, the simplest way to install these dependencies is to run:
```
sudo apt-get install make cmake libgtk2.0-dev libgoocanvas-dev
```

## Compilation
```
mkdir build
cd build
cmake ..
make
```

# Install Dependencies, Compile and Run Under Windows

I don't personally use Windows, as a consequence the following instructions
might be uselessly complicated. If you know a better way, please let me know.

1. Install Cygwin, go to [https://cygwin.com/install.html](https://cygwin.com/install.html) and download [setup-x86_64.exe](https://cygwin.com/setup-x86_64.exe).

2. Run __setup-x86_64.exe__ and click __next__, __next__, __next__, until you get to the package selection page. In this page, use the __search__ box in order to select the following packages for installation:
     - In __devel__ 
         - gcc-g++
         - make
         - cmake
         - git
     - In __lib__
         - libgtk2.0_0
         - libgtk2.0_devel
         - libgoocanvas_devel
     - In __X11__
         - xinit
         - xorg-server

    Once all packages are selected, click on __next__ the install process should start. You now have enough time to grab a coffee. Once installation is completed, you may exit the installer. 

3. Launch the __Cygwin prompt__, an icon should have been created on your desktop. Congratulation, you know have something that looks like a real terminal !

4. Get the files from Github, enter the following command:
    ```
    git clone https://github.com/xprov/NeuralNetwork
    ```

5. It's now time to compile ! Run the following commands:
    ```
    cd NeuralNetwork
    mkdir build
    cd build
    cmake ..
    make
    ```

# Debug VS Release mode

Since the training of neural network requires a lot of computation power, one might find useful to compile a debug and a release versions (the release version should run significantly faster):
```
mkdir debug && cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j
cd ..
mkdir release && cd release && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

# Typical usage
Let's train a Network using data from the
``Example/threeShapes/data/threeShapes.csv`` file. In this example, the input
are binary pictures of size 40x40. Consequently, there needs to be 1600 input
neurons on the first layer. The output layer must have size 3 since the inputs
are classified in three categories :

1. Straight line (first output neuron).
2. Rectangle (second output neuron).
3. Triangles (third output neuron).

Here we choose to have 3 hidden layers of size 20, so the "-l" argument is
"1600,20,20,20,3". Finally, we stop the learning once our network has obtain at
least 90% accuracy on the generalization set.

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

- Under __Windows__, this command will fail. You must start the X server manually, run the following commands prior to the one above.
     ```
     startxwin &
     export DISPLAY=:0.0
     ```
     
     


The grid on the left shows the state of the input neurons. Click on a square to
change it's value. The neurons layers are displayed to the right. Last column
is the output layer. Labels for the output neurons may be specified using the
``-l`` argument.

Here are some examples : 

 - A line ; the first output neuron is active.
![line detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectLine.png)

 - A rectangle ; the second output neuron is active.
![rectangle detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectRectangle.png)

 - A triangle ; the third output neuron is active.
![triangle detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectTriangle.png)






PROJECT( NeuralNetwork )

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g" ) 

ADD_EXECUTABLE( trainBPN trainBPN.cpp NeuralNetwork.cpp NeuralNetworkTrainer.cpp
    DataReader.cpp Matrix.cpp ActivationFunctions.cpp )

# Add an executable compiled from evalNN.cpp
ADD_EXECUTABLE( evalNN evalNN.cpp NeuralNetwork.cpp DataReader.cpp ActivationFunctions.cpp Matrix.cpp )

# Use the package PkgConfig to detect GTK+ headers/library files
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GTK2 REQUIRED gtk+-2.0)
PKG_CHECK_MODULES(GOOCANVAS REQUIRED goocanvas)

# Setup CMake to use GTK+, tell the compiler where to look for headers
# and to the linker where to look for libraries
INCLUDE_DIRECTORIES(${GTK2_INCLUDE_DIRS} ${GOOCANVAS_INCLUDE_DIRS})
LINK_DIRECTORIES(${GTK2_LIBRARY_DIRS} ${GOOCANVAS_LIBRARY_DIRS})

# Add other flags to the compiler
ADD_DEFINITIONS(${GTK2_CFLAGS_OTHER} ${GOOCANVAS_CFLAGS_OTHER})

# Add an executable compiled from gui.cpp
ADD_EXECUTABLE(gui gui.cpp NeuralNetwork.cpp Matrix.cpp ActivationFunctions.cpp)

# Link the target to the GTK+ libraries
TARGET_LINK_LIBRARIES(gui ${GTK2_LIBRARIES} ${GOOCANVAS_LIBRARIES})





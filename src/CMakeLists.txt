PROJECT( NeuralNetwork )

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g" ) 

ADD_EXECUTABLE( trainBPN trainBPN.cpp NeuralNetwork.cpp NeuralNetworkTrainer.cpp
    TrainingDataReader.cpp Matrix.cpp ActivationFunctions.cpp )


FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)

# Use the package PkgConfig to detect GTK+ headers/library files
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)

# Setup CMake to use GTK+, tell the compiler where to look for headers
# and to the linker where to look for libraries
INCLUDE_DIRECTORIES(${GTK3_INCLUDE_DIRS})
LINK_DIRECTORIES(${GTK3_LIBRARY_DIRS})

# Add other flags to the compiler
ADD_DEFINITIONS(${GTK3_CFLAGS_OTHER})

# Add an executable compiled from gui.cpp
ADD_EXECUTABLE(gui gui.cpp)

# Link the target to the GTK+ libraries
TARGET_LINK_LIBRARIES(gui ${GTK3_LIBRARIES})





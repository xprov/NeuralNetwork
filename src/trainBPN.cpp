//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>

#include "NeuralNetworkTrainer.h"
#include "DataReader.h"
#include "Matrix.h"
#include "vectorstream.h"

#if _MSC_VER
#pragma warning(push, 0)
#pragma warning(disable: 4702)
#endif

#include "cmdParser.h"

#if _MSC_VER
#pragma warning(pop)
#endif

// Operators from "vectorstream.h"
using BPN::operator<<;
using BPN::operator>>;

//-------------------------------------------------------------------------

int main( int argc, char* argv[] )
{

  //
  // Definition of the command line arguments
  //
  cli::Parser cmdParser( argc, argv );
  cmdParser.set_required<std::string>( "d", "dataFile", "Path to training data file." );
  cmdParser.set_optional<std::string>( "f", "format", "numberList", "Format of the training data file.\n"
                                 "     numberList, Comma separated list of numbers. The first ones are the\n"
                                 "                 values of the input nodes and the following ones are the\n"
                                 "                 expected values of the output nodes. The number of values\n"
                                 "                 must fit the number of intput/output nodes.\n"
                                 "     text,       Format is \"<text>\",<expectedOutput> where the expected\n"
                                 "                 output is given a comma separated list of integers. Size of\n"
                                 "                 the inputs may not equal the number of input nodes. If less\n"
                                 "                 then extra zeros are added. If more, extra characters are\n"
                                 "                 ignored.");
  cmdParser.set_optional<std::string>( "l", "layers", "[]", "Comma separated list of the layers sizes (e.g. 16,4,4,3 or 1,1,1).\n     First layer is the input neurons.\n     Last layer is the output layer." );
  cmdParser.set_optional<std::string>( "i", "import", "", "Import neural network from file before training. ( \"-\" stands for stdin)" );
  cmdParser.set_optional<std::string>( "e", "export", "", "Export neural network to file after training. ( \"-\" stands for stdout)" );

  cmdParser.set_optional<std::string>( "s", "activation", "Sigmoid(1)", "The actiation function. Available options are:\n"
                                 "     Sigmoid(k), Logistic function with stepness ``k``.\n"
                                 "     ReLU,       Rectified linear unit.\n"
                                 "     LeakyReLY,  Leaky ReLU, like ReLU but with small gradiant (1/100)\n"
                                 "                 when the unit is not active.");
  cmdParser.set_optional<uint64_t>( "m", "maxEpoch", 100, "Maximum num of iterations (-1 stands for 2^64-1)." );
  cmdParser.set_optional<double>( "r", "learningRate", 0.01, "Multiplicative coefficient on error gradient" );
  cmdParser.set_optional<double>( "mom", "momentum", 0.9, "Multiplicative coefficient applied on previous error delta when non using batch learning." );
  cmdParser.set_optional<bool>( "b", "batchLearning", false, "Use batch learning (1 : yes, 0 : no)." );
  cmdParser.set_optional<double>( "a", "accuracy", 95, "Desired accuracy. Training stops when the desired accuracy is obtained." );
  cmdParser.set_optional<std::string>( "L", "labels", "", "Labels for output nodes. Comma separated list of work without white spaces.\n"
                                     "   Only for new networks and is only used with the GUI visualization tool.");
  cmdParser.set_optional<int32_t>( "v", "verbose", 1, "Verbosity level.\n     Level 0: quiet mode.\n     Level 1: prints training evolution.\n     Level 2: prints NN at initialization and at the end.\n     Level 3: prints NN at every iteration of the learning phase." );

  if ( !cmdParser.run() )
    {
      std::cout << "Invalid command line arguments";
      return 1;
    }

  //
  // Get values from the command line parser
  // 
  std::string const trainingDataPath   = cmdParser.get<std::string>( "d" );
  std::string const format             = cmdParser.get<std::string>( "f" );
  std::string const layers             = cmdParser.get<std::string>( "l" );
  std::string const importFile         = cmdParser.get<std::string>( "i" );
  std::string const exportFile         = cmdParser.get<std::string>( "e" );
  std::string const activationFunction = cmdParser.get<std::string>( "s" );
  uint64_t const    maxEpoch           = cmdParser.get<uint64_t>( "m" );
  double            learningRate       = cmdParser.get<double>( "r" );
  double            momentum           = cmdParser.get<double>( "mom" );
  bool              batchLearning      = cmdParser.get<bool>( "b" );
  double            accuracy           = cmdParser.get<double>( "a" );
  std::string const labels             = cmdParser.get<std::string>( "L" );
  int32_t           verbosity          = cmdParser.get<int32_t>( "v" );

  // If the user wants to create a new neural network then he must specify it's
  // shape (number of layers and their sizes). Otherwise, an existing neural
  // network must be imported. 
  // Verify that the user has specified at least one of the two.
  if ( layers.compare("[]") == 0 && importFile.compare("") == 0 )
    {
      std::cerr << "At least one parameter among -l or -i must be specified. (See help for more impormations)" << std::endl;
      exit(1);
    }

  // Verify that the user has specified not provided both.
  if ( layers.compare("[]") != 0 && importFile.compare("") != 0 )
    {
      std::cerr << "Only one parameter among -l and -i may be specified. (See help for more impormations)" << std::endl;
      exit(1);
    }

  // Validation of the input format : eigher `numberList` or `text`.
  BPN::InputDataFormat inputDataFormat;
  if ( format.compare("numberList") == 0 )
    inputDataFormat = BPN::numberList;
  else if ( format.compare("text") == 0 )
    inputDataFormat = BPN::text;
  else
    throw std::runtime_error("Invalid format for input data. For more help use --help or -h.");

  // Load activation function
  BPN::ActivationFunction* sigma = BPN::ActivationFunction::deserialize( activationFunction );

  // Create neural network
  BPN::Network* nn = NULL;

  if ( layers.compare("[]") != 0 ) // new network
    {
      // Read layers sizes. The first layer is the input layer, the last layer
      // is the output. All other layers are hidden.
      std::vector<int> layerSizes;
      std::stringstream ss(layers);
      ss >> layerSizes;

      nn = new BPN::Network( layerSizes, sigma, labels );
    }
  else if ( importFile.compare("") != 0 ) // existing network
    {
      if ( importFile.compare("-") == 0 ) // read on stdin
        {
          nn = new BPN::Network( std::cin );
        }
      else
        {
          // import network from text file
          std::fstream fs;
          fs.open( importFile, std::fstream::in );
          nn = new BPN::Network( fs );
        }
    }

  assert( nn != NULL );

  if (verbosity >= 2)
    {
      std::cout << *nn << std::endl;
    }

  // 
  // Data from the data file is loaded into memory
  BPN::DataReader dataReader( trainingDataPath, 
                             nn->getNumInputs(), 
                             nn->getNumOutputs(), 
                             inputDataFormat, 
                             verbosity );

  BPN::TrainingData data;
  if ( !dataReader.readTraningData( data ) )
    {
      return 1;
    }
  if ( verbosity >= 1 )
    {
      int nbTraining = data.m_trainingSet.size();
      int nbGeneralization = data.m_generalizationSet.size();
      int nbValidation = data.m_validationSet.size();
      std::cout << " Training data read successfully:\n";
      std::cout << "==========================================================================\n"
        << " Input data file: " << trainingDataPath << "\n"
        << " Read complete: " << nbTraining + nbGeneralization + nbValidation << " inputs loaded" 
        << " (" << nbTraining << " for training, "
        << nbGeneralization << " for generalization and " 
        << nbValidation << " for validation)\n"
        << "==========================================================================" 
        << std::endl;
    }


  //
  // Create neural network trainer
  // 
  BPN::NetworkTrainer::Settings trainerSettings;
  trainerSettings.m_learningRate     = learningRate;
  trainerSettings.m_momentum         = momentum;
  trainerSettings.m_useBatchLearning = batchLearning;
  trainerSettings.m_maxEpochs        = maxEpoch;
  trainerSettings.m_desiredAccuracy  = accuracy;
  trainerSettings.m_verbosity        = verbosity;

  BPN::NetworkTrainer trainer( trainerSettings, nn );

  //
  // All the real work is done here
  // 
  trainer.Train( data );
  // It's all over now

  if ( verbosity >= 2) 
    {
      std::cout << *nn << std::endl;
    }

  // 
  // If required, the network is exported
  if ( exportFile.compare("") != 0 )
    {
      if ( exportFile.compare("-") == 0 ) 
        {
          // export to stdout
          std::cout << nn->serialize() << std::endl;
        }
      else
        {
          // export to a text file
          std::fstream fs;
          fs.open( exportFile, std::fstream::out );
          fs << nn->serialize() << std::endl;
        }
    }

  // Useless but hey, best practices are best practices. 
  delete nn;
  return 0;
}


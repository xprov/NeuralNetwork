//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "NeuralNetwork.h"
#include "displayUtils.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <random>
#include <iostream>
#include <sstream>
#include <iomanip>

//-------------------------------------------------------------------------

namespace BPN
{
  Network::Network(const std::vector<int>& layerSizes, const ActivationFunction* sigma) 
    : m_layerSizes(layerSizes)
      , m_sigma(sigma)
    {
      assert(layerSizes.size() >= 3);
      m_numLayers = layerSizes.size();
      m_numInputs = layerSizes[0];
      m_numOutputs = layerSizes[m_numLayers-1];
      m_numOnLastHidden = layerSizes[m_numLayers-2];
      InitializeNetwork();
      InitializeWeights();
    }

  //Network::Network( Settings const& settings, const ActivationFunction* sigma )
  //  : m_numInputs( settings.m_numInputs ), 
  //    m_numHidden( settings.m_numHidden ), 
  //    m_numOutputs( settings.m_numOutputs ), 
  //    m_sigma(sigma)
  //{
  //  assert( settings.m_numInputs > 0 && settings.m_numOutputs > 0 && settings.m_numHidden > 0 );
  //  InitializeNetwork();
  //  InitializeWeights();
  //}

  //Network::Network( Settings const& settings, std::vector<double> const& weights, const ActivationFunction* sigma )
  //  : m_numInputs( settings.m_numInputs )
  //    , m_numHidden( settings.m_numHidden )
  //    , m_numOutputs( settings.m_numOutputs )
  //    , m_sigma( sigma )
  //{
  //  assert( settings.m_numInputs > 0 && settings.m_numOutputs > 0 && settings.m_numHidden > 0 );
  //  InitializeNetwork();
  //  //LoadWeights( weights );
  //}

  Network::Network(const char* filename)
    {
      loadFromFile(filename);
    }


  void Network::InitializeNetwork()
    {
      // Create storage and initialize the neurons and the outputs
      //-------------------------------------------------------------------------

      for (auto layerSize : m_layerSizes) 
        {
          m_neurons.push_back(Layer(layerSize, Neuron(0,0)));
        }
      m_inputNeurons = &m_neurons[0];
      m_lastHiddenNeurons = &m_neurons[m_numLayers-2];
      m_outputNeurons = &m_neurons[m_numLayers-1];


      // Add bias values
      for (int i=0; i<m_numLayers-1; ++i)
        {
          m_neurons[i].push_back(Neuron(1.0, 1.0));
        }

      // Set the size of clamped output 
      m_clampedOutputs.resize( m_numOutputs, 0 );

      // Create storage and initialize the weights
      //-------------------------------------------------------------------------
      for (int i=0; i<m_numLayers-1; ++i)
        {
          // add one the the input size for th bias
          m_weightsByLayer.push_back(Matrix(m_layerSizes[i]+1, m_layerSizes[i+1], 0.0));
        }

    }

  void Network::InitializeWeights()
    {
      std::random_device rd;
      //std::mt19937 generator( rd() );
      std::mt19937 generator( 0 );

      double const distributionRangeHalfWidth = ( 2.4 / m_numInputs );
      double const standardDeviation = distributionRangeHalfWidth * 2 / 6;
      std::normal_distribution<> normalDistribution( 0, standardDeviation );

      // Set weights to normally distributed random values between [-2.4 / numInputs, 2.4 / numInputs]
      for ( int32_t i=0; i<m_numLayers-1; ++i) 
        {
          int32_t currentLayerSize = m_layerSizes[i];
          int32_t nextLayerSize = m_layerSizes[i+1];
          for (int32_t currentLayerIdx=0; currentLayerIdx<=currentLayerSize; ++currentLayerIdx)
            {
              for (int32_t nextLayerIdx=0; nextLayerIdx<nextLayerSize; ++nextLayerIdx)
                {
                  double const weight = normalDistribution( generator );
                  m_weightsByLayer[i](currentLayerIdx, nextLayerIdx) = weight;
                }
            }
        }
      //for ( int32_t inputIdx = 0; inputIdx <= m_numInputs; inputIdx++ )
      //  {
      //    for ( int32_t hiddenIdx = 0; hiddenIdx < m_numHidden; hiddenIdx++ )
      //      {
      //        //int32_t const weightIdx = GetInputHiddenWeightIndex( inputIdx, hiddenIdx );
      //        double const weight = normalDistribution( generator );
      //        m_weightsInputHidden(inputIdx, hiddenIdx) = weight;
      //      }
      //  }

      //// Set weights to normally distributed random values between [-2.4 / numInputs, 2.4 / numInputs]
      //for ( int32_t hiddenIdx = 0; hiddenIdx <= m_numHidden; hiddenIdx++ )
      //  {
      //    for ( int32_t outputIdx = 0; outputIdx < m_numOutputs; outputIdx++ )
      //      {
      //        double const weight = normalDistribution( generator );
      //        //double const weight = rand();
      //        m_weightsHiddenOutput(hiddenIdx, outputIdx) = weight;
      //      }
      //  }
    }

  //void Network::LoadWeights( std::vector<double> const& weights )
  //  {
  //    int32_t const numInputHiddenWeights = m_numInputs * m_numHidden;
  //    int32_t const numHiddenOutputWeights = m_numHidden * m_numOutputs;
  //    assert( weights.size() == (unsigned int) numInputHiddenWeights + numHiddenOutputWeights );

  //    int32_t weightIdx = 0;
  //    for ( auto InputHiddenIdx = 0; InputHiddenIdx < numInputHiddenWeights; InputHiddenIdx++ )
  //      {
  //        m_weightsInputHidden[InputHiddenIdx] = weights[weightIdx];
  //        weightIdx++;
  //      }

  //    for ( auto HiddenOutputIdx = 0; HiddenOutputIdx < numHiddenOutputWeights; HiddenOutputIdx++ )
  //      {
  //        m_weightsHiddenOutput[HiddenOutputIdx] = weights[weightIdx];
  //        weightIdx++;
  //      }
  //  }

  // TODO GEN TO MULTIPLE LAYERS
  std::vector<int32_t> const& Network::Evaluate( std::vector<double> const& input )
    {
      assert( input.size() == (unsigned int) m_numInputs );
      //assert( m_inputNeurons.back() == -1.0 && m_hiddenNeurons.back() == -1.0 );
      // Local variables
      Layer& inputNeurons  = *m_inputNeurons;

      // Set input values
      //-------------------------------------------------------------------------

      // Activation function is not applied on the value of input neurons
      for ( int i=0; i<m_numInputs; ++i ) 
        {
          inputNeurons[i] = Neuron(input[i], input[i]);
        }

      // Update neurons one layer at the time starting from the first hidden
      // la yer to the output layer.
      //-------------------------------------------------------------------------
      
      for ( int32_t i=1; i<m_numLayers; ++i )
        {

          for ( int32_t actualIdx=0; actualIdx < m_layerSizes[i]; ++actualIdx )
            {

              double activation = 0.0;

              // Get weighted sum of pattern and bias neuron
              for ( int32_t prevIdx = 0; prevIdx <= m_layerSizes[i-1]; ++prevIdx )
                {
                  activation += m_neurons[i-1][prevIdx].value * m_weightsByLayer[i-1](prevIdx, actualIdx);
                }

              // Apply activation function
              m_neurons[i][actualIdx].activation = activation;
              m_neurons[i][actualIdx].value = m_sigma->evaluate( activation );

              // If this is the output layer (the last layer), then update
              // clamped outputs
              if (i == m_numLayers-1)
                {
                  m_clampedOutputs[actualIdx] = ClampOutputValue( activation );
                }
            }
        }

      return m_clampedOutputs;
    }

  std::string Network::selfDisplay() const
    {
      std::stringstream ss;
      ss << "+----------------------------------------------------+\n"
         << "| Number of input  nodes: " << m_numInputs << '\n'
         << "| Number of hidden nodes: " << DisplayUtils::vectorToString(m_layerSizes) << '\n'
         << "| Number of output nodes: " << m_numOutputs << "\n| \n"
         << "| Input  (line)(last is bias) -> Hidden #1 (column) weights:\n\n"
         << m_weightsByLayer[0]
         << "\n|\n";
      for (int32_t i=1; i<m_numLayers-2; ++i) 
        {
          ss << "| Hidden #" << i << " (line)(last is bias) to Hidden #" << i+1 << " (column) weights:\n\n" 
             << m_weightsByLayer[i]
             << "\n|\n";
        }
      ss << "| Hidden #" << m_numLayers-2 << " (line)(last is bias) -> Output (column) weights:\n\n"
         << m_weightsByLayer[m_numLayers-2]
         << "\n|\n"
         << "| Input  neurons    : " << DisplayUtils::vectorToString(*m_inputNeurons) << "\n"
         //<< "| Hidden neurons    : " << DisplayUtils::vectorToString(*m_hiddenNeurons) << "\n"
         << "| Output neurons    : " << DisplayUtils::vectorToString(*m_outputNeurons) << "\n"
         << "| Clamp o/p neurons : " << DisplayUtils::vectorToString(m_clampedOutputs) << "\n"
         << "+----------------------------------------------------+\n";
      return ss.str();
    }

  std::string Network::serialize() const
    {
    }

  void Network::saveToFile(const char* filename) const
    {
      (void) filename;
    }

  void Network::loadFromFile(const char* filename)
    {
      (void) filename;
    }


std::ostream& operator<<( std::ostream& os, const BPN::Network& n )
  {
    //return os;
    os << n.selfDisplay();
    return os;
  }

std::ostream& operator<<( std::ostream& os, const BPN::Neuron& n) 
  {
    os << "(" << n.activation << ", " << n.value << ")";
    return os;
  }

}

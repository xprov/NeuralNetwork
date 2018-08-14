//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// A simple neural network supporting only a single hidden layer

#pragma once
#include <stdint.h>
#include <iostream>
#include <vector>
#include "SafeVector.h"
#include "Matrix.h"
#include "ActivationFunctions.h"
#include "vectorstream.h"

//-------------------------------------------------------------------------

namespace BPN 
{

  //-------------------------------------------------------------------------
  struct Neuron
    {
      Neuron() : activation(0.0), value(0.0) {} // empty constructor
      Neuron(double a, double v) : activation(a), value(v) 
      {}
      double activation; 
      double value; // = Sigma(activation)
      friend std::ostream& operator<<( std::ostream& os, const BPN::Neuron& n );
    };

  class Network
    {
      friend class NetworkTrainer;

      //-------------------------------------------------------------------------

      inline static int32_t ClampOutputValue( double x )
        {
          if ( x < 0.1 ) return 0;
          else if ( x > 0.9 ) return 1;
          else return -1;
        }

  public:

      Network( const std::vector<int>& layerSizes, const ActivationFunction* sigma );
      Network( std::istream& is );

      std::vector<int32_t> const& Evaluate( std::vector<double> const& input );

      void saveToFile(const char* filename) const;

      std::string serialize() const;
      void deserialize(std::istream& is);
  private:
      void loadFromFile(const char* filename);
      void InitializeNetwork();
      void InitializeWeights();

  private:

      int32_t                     m_numLayers;       // number of layers including input and output (min 3)
      int32_t                     m_numInputs;       // number of neurons on the input layer
      int32_t                     m_numOutputs;      // number of neurons on the output layer
      int32_t                     m_numOnLastHidden; // number of neurons on the last hidden layer
      std::vector<int>            m_layerSizes;      // m_layerSizes[i] is the number of neurons on the i-th layer.


      typedef std::vector<Neuron> Layer;
      std::vector<Layer>          m_neurons;           // m_neurons[i] is the i-th layer
      Layer*                      m_inputNeurons;      // &m_neurons[0]
      Layer*                      m_lastHiddenNeurons; // &m_neurons[-2] (python notations)
      Layer*                      m_outputNeurons;     // &m_neurons[-1]

      std::vector<int32_t>        m_clampedOutputs;

      // m_wrigntsByLayer[i] is the matrix of weights from layer i to layer i+1
      std::vector<Matrix>         m_weightsByLayer;

      const ActivationFunction*   m_sigma;

  public:

      std::string selfDisplay() const;
      friend std::ostream& operator<<( std::ostream& os, const BPN::Network& n );
    };

}



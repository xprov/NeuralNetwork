//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "NeuralNetworkTrainer.h"
#include "displayUtils.h"
#include <string.h>
#include <assert.h>
#include <iostream>

//-------------------------------------------------------------------------

namespace BPN
{
    NetworkTrainer::NetworkTrainer( Settings const& settings, Network* pNetwork )
        : m_pNetwork( pNetwork )
        , m_learningRate( settings.m_learningRate )
        , m_momentum( settings.m_momentum )
        , m_desiredAccuracy( settings.m_desiredAccuracy )
        , m_maxEpochs( settings.m_maxEpochs )
        , m_useBatchLearning( settings.m_useBatchLearning )
        , m_currentEpoch( 0 )
        , m_trainingSetAccuracy( 0 )
        , m_validationSetAccuracy( 0 )
        , m_generalizationSetAccuracy( 0 )
        , m_trainingSetMSE( 0 )
        , m_validationSetMSE( 0 )
        , m_generalizationSetMSE( 0 )
    {
        assert( pNetwork != nullptr );
        for (int32_t i=0; i < m_pNetwork->m_numLayers-1; ++i) 
          {
            // Generate the delta matrix from later i to layer i+1
            // add one to actualLayerSize for bias
            int32_t actualLayerSize = m_pNetwork->m_layerSizes[i];
            int32_t nextLayerSize = m_pNetwork->m_layerSizes[i+1];
            m_deltas.push_back( Matrix( actualLayerSize+1, nextLayerSize, 0.0 ) );
          }

        // m_errorGradients[0] is not used... dummy value to fill the spot
        m_errorGradients.push_back(SafeVector<double>());
        for (int32_t i=1; i < m_pNetwork->m_numLayers; ++i) 
          {
            int layerSize = m_pNetwork->m_layerSizes[i];
            if ( i < m_pNetwork->m_numLayers-1 ) 
              {
                layerSize += 1; // add one for bias
              }
            m_errorGradients.push_back( SafeVector<double>() );
            m_errorGradients[i].resize( layerSize, 0.0 );
          }

        // to remove
        //m_deltaInputHidden = &m_deltas[0];
        //m_deltaHiddenOutput = &m_deltas[1];
        //m_errorGradientsHidden = &m_errorGradients[1];
        //m_errorGradientsOutput = &m_errorGradients[2];
    }

    void NetworkTrainer::Train( TrainingData const& trainingData )
    {
        // Reset training state
        m_currentEpoch = 0;
        m_trainingSetAccuracy = 0;
        m_validationSetAccuracy = 0;
        m_generalizationSetAccuracy = 0;
        m_trainingSetMSE = 0;
        m_validationSetMSE = 0;
        m_generalizationSetMSE = 0;

        // Print header
        //-------------------------------------------------------------------------

        std::cout	<< std::endl << " Neural Network Training Starting: " << std::endl
                    << "==========================================================================" 
                    << std::endl
                    << " LR: " << m_learningRate 
                    << ", Momentum: " << m_momentum 
                    << ", Max Epochs: " << m_maxEpochs << std::endl
                    << " " << m_pNetwork->m_numInputs 
                    << " Input Neurons, " << m_pNetwork->m_numLayers-2
                    << " Hidden Layers, " << m_pNetwork->m_numOutputs
                    << " Output Neurons" << std::endl
                    << "==========================================================================" 
                    << std::endl << std::endl;

        // Train network using training dataset for training and generalization dataset for testing
        //--------------------------------------------------------------------------------------------------------

        while ( ( m_trainingSetAccuracy < m_desiredAccuracy 
                 || m_generalizationSetAccuracy < m_desiredAccuracy ) 
               && m_currentEpoch < m_maxEpochs )
        {
            // Use training set to train network
            RunEpoch( trainingData.m_trainingSet );

            // Get generalization set accuracy and MSE
            GetSetAccuracyAndMSE( trainingData.m_generalizationSet, 
                                 m_generalizationSetAccuracy, 
                                 m_generalizationSetMSE );

            std::cout << "Epoch :" << m_currentEpoch;
            std::cout << " Training Set Accuracy:" << m_trainingSetAccuracy 
              << "%, MSE: " << m_trainingSetMSE;
            std::cout << " Generalization Set Accuracy:" << m_generalizationSetAccuracy 
              << "%, MSE: " << m_generalizationSetMSE << std::endl;

            m_currentEpoch++;
        }

        // Get validation set accuracy and MSE
        GetSetAccuracyAndMSE( trainingData.m_validationSet, m_validationSetAccuracy, m_validationSetMSE );

        // Print validation accuracy and MSE
        std::cout << std::endl << "Training Complete!!! - > Elapsed Epochs: " << m_currentEpoch << std::endl;
        std::cout << " Validation Set Accuracy: " << m_validationSetAccuracy << std::endl;
        std::cout << " Validation Set MSE: " << m_validationSetMSE << std::endl << std::endl;
    }

    double NetworkTrainer::getErrorGradient( int32_t layer, int32_t index ) const
      {
        assert( layer >= 1 ); // no error on input
        assert( layer <= m_pNetwork->m_numLayers-2 ); // output layer is computed differently

        // Get sum of ``layer[i] --> layer[i+1] weights`` * layer[i+1] error dradients
        double weightedSum = 0;
        int32_t numOnNextLayer   = m_pNetwork->m_layerSizes[layer+1];
        for ( auto nextLayerIdx=0; nextLayerIdx < numOnNextLayer; ++nextLayerIdx )
          {
            //int32_t const weightIdx = m_pNetwork->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );
            weightedSum += m_pNetwork->m_weightsByLayer[layer](index, nextLayerIdx) 
              * m_errorGradients[layer+1][nextLayerIdx];
          }

        // Return error gradient
        const Neuron& n = m_pNetwork->m_neurons[layer][index];
        double derivative = m_pNetwork->m_sigma->evalDerivative( n.activation, n.value );
        return derivative * weightedSum;
        //return m_pNetwork->m_hiddenNeurons[hiddenIdx].value
        //  * ( 1.0 - m_pNetwork->m_hiddenNeurons[hiddenIdx].value ) * weightedSum;
      }

    //double NetworkTrainer::GetHiddenErrorGradient( int32_t hiddenIdx ) const
    //{
    //    // Get sum of hidden->output weights * output error gradients
    //    double weightedSum = 0;
    //    for ( auto outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
    //    {
    //        //int32_t const weightIdx = m_pNetwork->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );
    //        weightedSum += m_pNetwork->m_weightsByLayer[m_pNetwork->m_numLayers-2](hiddenIdx, outputIdx) 
    //                       * (*m_errorGradientsOutput)[outputIdx];
    //    }
    //    
    //    // Return error gradient
    //    const Neuron& n = (*m_pNetwork->m_hiddenNeurons)[hiddenIdx];
    //    double derivative = m_pNetwork->m_sigma->evalDerivative( n.activation, n.value );
    //    return derivative * weightedSum;
    //    //return m_pNetwork->m_hiddenNeurons[hiddenIdx].value
    //    //  * ( 1.0 - m_pNetwork->m_hiddenNeurons[hiddenIdx].value ) * weightedSum;
    //}

    void NetworkTrainer::RunEpoch( TrainingSet const& trainingSet )
    {
        double incorrectEntries = 0;
        double MSE = 0;

        for ( auto const& trainingEntry : trainingSet )
        {
            // Feed inputs through network and back propagate errors
            m_pNetwork->Evaluate( trainingEntry.m_inputs );

            Backpropagate( trainingEntry.m_expectedOutputs );

            // Check all outputs from neural network against desired values
            bool resultCorrect = true;
            for ( int outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
            {
                if ( m_pNetwork->m_clampedOutputs[outputIdx] != trainingEntry.m_expectedOutputs[outputIdx] )
                {
                    resultCorrect = false;
                }

                // Calculate MSE
                MSE += pow( ( (*m_pNetwork->m_outputNeurons)[outputIdx].value 
                             - trainingEntry.m_expectedOutputs[outputIdx] ), 2);
            }

            if ( !resultCorrect )
            {
                incorrectEntries++;
            }
        }
        
        // If using batch learning - update the weights
        if ( m_useBatchLearning )
        {
            UpdateWeights();
        }

        // Update training accuracy and MSE
        m_trainingSetAccuracy = 100.0 - ( incorrectEntries / trainingSet.size() * 100.0 );
        m_trainingSetMSE = MSE / ( m_pNetwork->m_numOutputs * trainingSet.size() );

       // std::cout << *m_pNetwork << std::endl;
    }

    void NetworkTrainer::Backpropagate( std::vector<int32_t> const& expectedOutputs )
      {
        // Modify deltas between the last hidden layer and output layers
        //--------------------------------------------------------------------------------------------------------
        int32_t numLayers = m_pNetwork->m_numLayers;
        BPN::Network::Layer& lastHiddenNeurons = *m_pNetwork->m_lastHiddenNeurons;
        BPN::Network::Layer& outputNeurons = *m_pNetwork->m_outputNeurons;

        for ( auto outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; ++outputIdx )
          {
            // Get error gradient for every output node
            m_errorGradients[numLayers-1][outputIdx] = getOutputErrorGradient( 
                                         (double) expectedOutputs[outputIdx], 
                                         outputNeurons[outputIdx] );

            // For all nodes in the last hidden layer and bias neuron
            for ( auto hiddenIdx = 0; hiddenIdx <= m_pNetwork->m_numOnLastHidden; ++hiddenIdx )
              {
                //int32_t const weightIdx = m_pNetwork->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );

                // Calculate change in weight
                if ( m_useBatchLearning )
                  {
                    m_deltas[numLayers-2](hiddenIdx, outputIdx) +=
                      m_learningRate 
                      * lastHiddenNeurons[hiddenIdx].value 
                      * m_errorGradients[numLayers-1][outputIdx];
                    //(*m_deltaHiddenOutput)(hiddenIdx, outputIdx) += 
                    //  m_learningRate 
                    //  * lastHiddenNeurons[hiddenIdx].value 
                    //  * (*m_errorGradientsOutput)[outputIdx];
                  }
                else
                  {
                    m_deltas[numLayers-2](hiddenIdx, outputIdx) = 
                      m_learningRate 
                      * lastHiddenNeurons[hiddenIdx].value 
                      * m_errorGradients[numLayers-1][outputIdx] 
                      + m_momentum * m_deltas[numLayers-2](hiddenIdx, outputIdx);
                    //(*m_deltaHiddenOutput)(hiddenIdx, outputIdx) = 
                    //  m_learningRate * lastHiddenNeurons[hiddenIdx].value 
                    //  * (*m_errorGradientsOutput)[outputIdx] 
                    //  + m_momentum * (*m_deltaHiddenOutput)(hiddenIdx, outputIdx);
                  }
              }
          }

        //// //// Modify deltas between all other layers
        //// ////--------------------------------------------------------------------------------------------------------
        // deltas[numLaters-2] have been computed, lets compute all others.
        for ( int32_t layer = numLayers-3; layer >= 0; --layer )
          {
            // ``next layer`` is (layer+1)-th layer
            // ``actual layer`` is layer-th layer
            for ( auto nextIdx = 0; nextIdx < m_pNetwork->m_layerSizes[layer+1]; nextIdx++ )
              {
                // Get error gradient for every hidden node
                m_errorGradients[layer+1][nextIdx] = getErrorGradient( layer+1, nextIdx );
                //(*m_errorGradientsHidden)[hiddenIdx] = getErrorGradient( numLayers-2, hiddenIdx );
                //(*m_errorGradientsHidden)[hiddenIdx] = GetHiddenErrorGradient( hiddenIdx );

                // For all nodes in actual layer and bias neuron
                for ( auto actualIdx = 0; actualIdx <= m_pNetwork->m_layerSizes[layer]; actualIdx++ )
                  {
                    // Calculate change in weight 
                    if ( m_useBatchLearning )
                      {
                        m_deltas[layer](actualIdx, nextIdx) +=
                          m_learningRate 
                          * m_pNetwork->m_neurons[layer][actualIdx].value 
                          * m_errorGradients[layer+1][nextIdx];

                        //(*m_deltaInputHidden)(inputIdx, hiddenIdx) += 
                        //  m_learningRate 
                        //  * inputNeurons[inputIdx].value 
                        //  * (*m_errorGradientsHidden)[hiddenIdx];
                      }
                    else
                      {
                        m_deltas[layer](actualIdx, nextIdx) = 
                          m_learningRate 
                          * m_pNetwork->m_neurons[layer][actualIdx].value
                          * m_errorGradients[layer+1][nextIdx]
                          + m_momentum * m_deltas[layer](actualIdx, nextIdx);
                        //(*m_deltaInputHidden)(inputIdx, hiddenIdx) = 
                        //  m_learningRate 
                        //  * inputNeurons[inputIdx].value 
                        //  * (*m_errorGradientsHidden)[hiddenIdx] 
                        //  + m_momentum * (*m_deltaInputHidden)(inputIdx, hiddenIdx);
                      }
                  }
              }

          }

        //for ( auto hiddenIdx = 0; hiddenIdx < m_pNetwork->m_numOnLastHidden; hiddenIdx++ )
        //  {
        //    // Get error gradient for every hidden node
        //    //(*m_errorGradientsHidden)[hiddenIdx] = GetHiddenErrorGradient( hiddenIdx );
        //    (*m_errorGradientsHidden)[hiddenIdx] = getErrorGradient( numLayers-2, hiddenIdx );

        //    // For all nodes in input layer and bias neuron
        //    for ( auto inputIdx = 0; inputIdx <= m_pNetwork->m_numInputs; inputIdx++ )
        //      {
        //        //int32_t const weightIdx = m_pNetwork->GetInputHiddenWeightIndex( inputIdx, hiddenIdx );

        //        // Calculate change in weight 
        //        if ( m_useBatchLearning )
        //          {
        //            (*m_deltaInputHidden)(inputIdx, hiddenIdx) += 
        //              m_learningRate * inputNeurons[inputIdx].value * (*m_errorGradientsHidden)[hiddenIdx];
        //          }
        //        else
        //          {
        //            (*m_deltaInputHidden)(inputIdx, hiddenIdx) = 
        //              m_learningRate * inputNeurons[inputIdx].value * (*m_errorGradientsHidden)[hiddenIdx] 
        //              + m_momentum * (*m_deltaInputHidden)(inputIdx, hiddenIdx);
        //          }
        //      }
        //  }



        // If using stochastic learning update the weights immediately
        if ( !m_useBatchLearning )
          {
            UpdateWeights();
          }
        //std::cout << "################################################################\n";
        //std::cout << m_deltas[0] << std::endl;
        //std::cout << "\n";
        //std::cout << m_deltas[1] << std::endl;
        //std::cout << "--------------------------------------------------\n";
        //std::cout << "errG hid : " << DisplayUtils::vectorToString(m_errorGradients[1]) << std::endl;
        //std::cout << "errG out : " << DisplayUtils::vectorToString(m_errorGradients[2]) << std::endl;
        //std::cout << "################################################################\n";

      }

    void NetworkTrainer::UpdateWeights()
    {
      for ( int32_t layer = 0; layer < m_pNetwork->m_numLayers-1; ++layer )
        {
          for ( int32_t actualIdx = 0; actualIdx <= m_pNetwork->m_layerSizes[layer]; ++actualIdx )
            {
              for ( int32_t nextIdx = 0; nextIdx < m_pNetwork->m_layerSizes[layer+1]; ++nextIdx )
                {
                  m_pNetwork->m_weightsByLayer[layer](actualIdx, nextIdx) += 
                    m_deltas[layer](actualIdx, nextIdx);

                  // Clear delta only if using batch (previous delta is needed for momentum
                  if ( m_useBatchLearning )
                    {
                      m_deltas[layer](actualIdx, nextIdx) = 0.0;
                    }

                }
            }

        }
        //// Input -> hidden weights
        ////--------------------------------------------------------------------------------------------------------

        //for ( auto inputIdx = 0; inputIdx <= m_pNetwork->m_numInputs; inputIdx++ )
        //{
        //    for ( auto hiddenIdx = 0; hiddenIdx < m_pNetwork->m_numOnLastHidden; hiddenIdx++ )
        //    {
        //        //int32_t const weightIdx = m_pNetwork->GetInputHiddenWeightIndex( inputIdx, hiddenIdx );
        //        m_pNetwork->m_weightsByLayer[0](inputIdx, hiddenIdx) += (*m_deltaInputHidden)(inputIdx, hiddenIdx);

        //        // Clear delta only if using batch (previous delta is needed for momentum
        //        if ( m_useBatchLearning )
        //        {
        //            (*m_deltaInputHidden)(inputIdx, hiddenIdx) = 0;
        //        }
        //    }
        //}

        //// Hidden -> output weights
        ////--------------------------------------------------------------------------------------------------------

        //for ( auto hiddenIdx = 0; hiddenIdx <= m_pNetwork->m_numOnLastHidden; hiddenIdx++ )
        //{
        //    for ( auto outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
        //    {
        //        //int32_t const weightIdx = m_pNetwork->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );
        //        m_pNetwork->m_weightsByLayer[m_pNetwork->m_numLayers-2](hiddenIdx, outputIdx) += (*m_deltaHiddenOutput)(hiddenIdx, outputIdx);

        //        // Clear delta only if using batch (previous delta is needed for momentum)
        //        if ( m_useBatchLearning )
        //        {
        //            (*m_deltaInputHidden)(hiddenIdx, outputIdx) = 0;
        //        }
        //    }
        //}
    }

    void NetworkTrainer::GetSetAccuracyAndMSE( TrainingSet const& trainingSet, double& accuracy, double& MSE ) const
    {
        accuracy = 0;
        MSE = 0;

        double numIncorrectResults = 0;
        for ( auto const& trainingEntry : trainingSet )
        {
            m_pNetwork->Evaluate( trainingEntry.m_inputs );

            // Check if the network outputs match the expected outputs
            bool correctResult = true;
            for ( int32_t outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
            {
                if ( (double) m_pNetwork->m_clampedOutputs[outputIdx] != trainingEntry.m_expectedOutputs[outputIdx] )
                {
                    correctResult = false;
                }

                MSE += pow( ( (*m_pNetwork->m_outputNeurons)[outputIdx].value - trainingEntry.m_expectedOutputs[outputIdx] ), 2 );
            }

            if ( !correctResult )
            {
                numIncorrectResults++;
            }
        }

        accuracy = 100.0f - ( numIncorrectResults / trainingSet.size() * 100.0 );
        MSE = MSE / ( m_pNetwork->m_numOutputs * trainingSet.size() );
    }

}

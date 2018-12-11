//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "DataReader.h"
#include <assert.h>
#include <iosfwd>
#include <algorithm>
#include <iostream>

//-------------------------------------------------------------------------


//std::vector<std::string> split(const std::string& t, const std::string& m)
//{
//  std::vector<std::string> splitted;
//  std::size_t first = 0;
//  std::size_t last = t.find( m );
//  while ( last != std::string::npos )
//    {
//      splitted.push_back( t.substr( first, last-first ) );
//      first = last + m.size();
//      last = t.find( m, first );
//    }
//  splitted.push_back( t.substr( first, t.size() - first ) );
//  return splitted;
//}

void insertListInStream( std::stringstream& ss, const std::string& text, const std::string& delimitor)
{
  std::size_t first = 0;
  std::size_t last = text.find( delimitor );
  while ( last != std::string::npos )
    {
      ss << text.substr( first, last-first ) << " ";
      first = last + delimitor.size();
      last = text.find( delimitor, first );
    }
  ss << text.substr( first, text.size() - first );
}

void textToListOfDouble( std::vector<double>& l, const std::string& s)
{
  for ( uint32_t i=0; i<s.size(); ++i )
    {
      l.push_back( ((double)s[i]) / 256.0);
    }
}


namespace BPN
{
  DataReader::DataReader( std::string const& filename, 
                          int32_t numInputs, 
                          int32_t numOutputs,
                          BPN::InputDataFormat format,
                          int32_t verbosity )
    : m_filename( filename ), 
    m_numInputs( numInputs ), 
    m_numOutputs( numOutputs ), 
    m_dataFormat( format ), 
    m_verbosity( verbosity )
  {
    assert( !filename.empty() && m_numInputs > 0 && m_numOutputs > 0 );
  }

  bool DataReader::readTraningData( TrainingData& data )
    {
      std::vector<TrainingEntry> entries;
      assert( !m_filename.empty() );

      std::fstream inputFile;
      inputFile.open( m_filename, std::ios::in );

      if ( inputFile.is_open() )
        {
          std::string line;

          while ( !inputFile.eof() )
            {
              std::getline( inputFile, line );

              // line that starts with # are comments and thus ignored.
              if (line[0] == '#' || line.size() == 0)
                continue;

              entries.push_back( TrainingEntry() );
              TrainingEntry& entry = entries.back();

              if ( m_dataFormat == BPN::numberList )
                {
                  std::stringstream ss;
                  insertListInStream( ss, line, "," );
                  for ( int i=0; i < m_numInputs; ++i )
                    {
                      double d;
                      ss >> d;
                      entry.m_inputs.push_back( d );
                    }
                  for ( int i=0; i < m_numInputs; ++i )
                    {
                      int32_t x;
                      ss >> x;
                      entry.m_expectedOutputs.push_back( x );
                    }
                }
              else // ( m_dataType == BPN::text )
                {
                  if (line[0] != '"')
                    {
                      throw std::runtime_error("Bad training data file (must start by a text between \")");
                    }
                  std::size_t last = line.find_last_of( '"' );
                  if (line[last+1] != ',')
                    {
                      throw std::runtime_error("Bad training data file (should be \"<text>\",<outputs>");
                    }
                  std::vector<double> inputs;
                  textToListOfDouble( inputs, line.substr(1, last-1) );
                  std::stringstream outputStream;
                  insertListInStream( outputStream, line.substr(last+2, std::string::npos), ",");
                  int nbInputs = inputs.size();
                  for ( int i=0; i < std::min( nbInputs, m_numInputs ); ++i )
                    {
                      entry.m_inputs.push_back(inputs[i]);
                    }
                  for ( int i=0; i < m_numInputs - nbInputs; ++i )
                    {
                      entry.m_inputs.push_back(0);
                    }
                  for ( int i=0; i < m_numOutputs; ++i )
                    {
                      int32_t x;
                      outputStream >> x;
                      entry.m_expectedOutputs.push_back( x );
                    }
                }

              if ( m_verbosity >= 2 )
                {
                  std::cout << "  Input : " << entry.m_inputs << std::endl;
                  std::cout << "  Output : " << entry.m_expectedOutputs << "\n" << std::endl;

                }

              assert( entry.m_inputs.size() == (uint32_t) m_numInputs );
              assert( entry.m_expectedOutputs.size() == (uint32_t) m_numOutputs );
            }

          inputFile.close();

          if ( !entries.empty() )
            {
              CreateTrainingData(data, entries);
            }
          return true;
        }
      else
        {
          throw std::runtime_error("Error opening input file");
        }
    }



  void DataReader::CreateTrainingData( TrainingData& data, std::vector<TrainingEntry>& entries )
    {
      assert( !entries.empty() );

      std::random_shuffle( entries.begin(), entries.end() );

      // Training set
      int32_t const numEntries = (int32_t) entries.size();
      int32_t const numTrainingEntries  = (int32_t) ( 0.6 * numEntries );
      int32_t const numGeneralizationEntries = (int32_t) ( ceil( 0.2 * numEntries ) );

      int32_t entryIdx = 0;
      for ( ; entryIdx < numTrainingEntries; entryIdx++ )
        {
          data.m_trainingSet.push_back( entries[entryIdx] );
        }

      // Generalization set
      for ( ; entryIdx < numTrainingEntries + numGeneralizationEntries; entryIdx++ )
        {
          data.m_generalizationSet.push_back( entries[entryIdx] );
        }

      // Validation set
      for ( ; entryIdx < numEntries; entryIdx++ )
        {
          data.m_validationSet.push_back( entries[entryIdx] );
        }
    }
}


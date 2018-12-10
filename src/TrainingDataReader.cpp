//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "TrainingDataReader.h"
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

std::stringstream listToStream(const std::string& t, const std::string& m)
{
  std::stringstream ss;
  std::size_t first = 0;
  std::size_t last = t.find( m );
  while ( last != std::string::npos )
    {
      ss << t.substr( first, last-first ) << " ";
      first = last + m.size();
      last = t.find( m, first );
    }
  ss << t.substr( first, t.size() - first );
  return ss;
}


namespace BPN
{
  TrainingDataReader::TrainingDataReader( std::string const& filename, int32_t numInputs, int32_t numOutputs )
    : m_filename( filename ), m_numInputs( numInputs ), m_numOutputs( numOutputs )
  {
    assert( !filename.empty() && m_numInputs > 0 && m_numOutputs > 0 );
  }

  bool TrainingDataReader::ReadData()
    {
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
              if (line[0] == '#')
                continue;

              m_entries.push_back( TrainingEntry() );
              TrainingEntry& entry = m_entries.back();

              std::stringstream ss = listToStream(line, ",");;
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

          inputFile.close();

          if ( !m_entries.empty() )
            {
              CreateTrainingData();
            }

          return true;
        }
      else
        {
          throw std::runtime_error("Error opening input file");
        }
    }

  void TrainingDataReader::CreateTrainingData()
    {
      assert( !m_entries.empty() );

      std::random_shuffle( m_entries.begin(), m_entries.end() );

      // Training set
      int32_t const numEntries = (int32_t) m_entries.size();
      int32_t const numTrainingEntries  = (int32_t) ( 0.6 * numEntries );
      int32_t const numGeneralizationEntries = (int32_t) ( ceil( 0.2 * numEntries ) );

      int32_t entryIdx = 0;
      for ( ; entryIdx < numTrainingEntries; entryIdx++ )
        {
          m_data.m_trainingSet.push_back( m_entries[entryIdx] );
        }

      // Generalization set
      for ( ; entryIdx < numTrainingEntries + numGeneralizationEntries; entryIdx++ )
        {
          m_data.m_generalizationSet.push_back( m_entries[entryIdx] );
        }

      // Validation set
      for ( ; entryIdx < numEntries; entryIdx++ )
        {
          m_data.m_validationSet.push_back( m_entries[entryIdx] );
        }
    }
}


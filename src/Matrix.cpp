#include <iostream>
#include <sstream>
#include "Matrix.h"

namespace BPN
{

  std::ostream& operator<<(std::ostream& os, const Matrix& m)
    {
      for ( int i=0; i<m.nRows; ++i ) 
        {
          os << "[" << m(i, 0);
          for ( int j=1; j<m.nCols; ++j ) 
            {
              os << ",\t" << m(i, j);
            }
          os << "]\n";
        }
      return os;
    }

}

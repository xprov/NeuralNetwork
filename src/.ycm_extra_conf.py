def FlagsForFile( filename ):
  return { 'flags': [
    '-Wall',
    '-Wextra',
    '-Werror',
    '-std=c++11',
    '-x', 'c++',    
    '-isystem', '/usr/include/c++/9',
    '-isystem', '/usr/include/c++/9/backward',
    '-isystem', '/usr/local/include',
    '-isystem', '/usr/include',
  ] }

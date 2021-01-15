#ifndef SIZE_CMD_H
#define SIZE_CMD_H

#include <iostream>
#include <stdlib.h>

namespace sc {
  int processArgs(int argc, char * argv[], int &result, bool &check) {
    if (argc < 2) {
      return -1;
    } 

    std::string size = argv[1];
    int factor = 1;

    if (argv[1][0] == '-' and argv[1][1] == 's') {
      size = size.substr(2);
      factor = -1;
    }

    if (argc > 2) {
      std::string checkflag = argv[2];
      if(checkflag.compare("--no-check") == 0)
        check = false;
    }
    
    result = factor * atoi(size.c_str());

    return 0;
  }
}



#endif

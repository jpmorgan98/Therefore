# the compiler: gcc for C program, define as g++ for C++
  CC = g++
 
  # compiler flags:
  #  -g     - this flag adds debugging information to the executable file
  #  -Wall  - this flag is used to turn on most compiler warnings
  CFLAGS  = -lhdf5 -lhdf5_cpp -std=c++14 -llapack
 
  # The build target 
  TARGET = main
 
  all: $(TARGET)
 
  $(TARGET): $(TARGET).cpp
            $(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp
 
  clean:
              $(RM) $(TARGET)
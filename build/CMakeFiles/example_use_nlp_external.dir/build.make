# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/hakim/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/hakim/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hakim/Desktop/Phd/projects/MPC_casadi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hakim/Desktop/Phd/projects/MPC_casadi/build

# Include any dependencies generated for this target.
include CMakeFiles/example_use_nlp_external.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/example_use_nlp_external.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/example_use_nlp_external.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example_use_nlp_external.dir/flags.make

CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o: CMakeFiles/example_use_nlp_external.dir/flags.make
CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o: /home/hakim/Desktop/Phd/projects/MPC_casadi/example/other/example_use_nlp_external.cpp
CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o: CMakeFiles/example_use_nlp_external.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hakim/Desktop/Phd/projects/MPC_casadi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o -MF CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o.d -o CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o -c /home/hakim/Desktop/Phd/projects/MPC_casadi/example/other/example_use_nlp_external.cpp

CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hakim/Desktop/Phd/projects/MPC_casadi/example/other/example_use_nlp_external.cpp > CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.i

CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hakim/Desktop/Phd/projects/MPC_casadi/example/other/example_use_nlp_external.cpp -o CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.s

# Object files for target example_use_nlp_external
example_use_nlp_external_OBJECTS = \
"CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o"

# External object files for target example_use_nlp_external
example_use_nlp_external_EXTERNAL_OBJECTS =

example_use_nlp_external: CMakeFiles/example_use_nlp_external.dir/example/other/example_use_nlp_external.cpp.o
example_use_nlp_external: CMakeFiles/example_use_nlp_external.dir/build.make
example_use_nlp_external: /usr/local/lib/libcasadi.so
example_use_nlp_external: CMakeFiles/example_use_nlp_external.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hakim/Desktop/Phd/projects/MPC_casadi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_use_nlp_external"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_use_nlp_external.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example_use_nlp_external.dir/build: example_use_nlp_external
.PHONY : CMakeFiles/example_use_nlp_external.dir/build

CMakeFiles/example_use_nlp_external.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example_use_nlp_external.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example_use_nlp_external.dir/clean

CMakeFiles/example_use_nlp_external.dir/depend:
	cd /home/hakim/Desktop/Phd/projects/MPC_casadi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hakim/Desktop/Phd/projects/MPC_casadi /home/hakim/Desktop/Phd/projects/MPC_casadi /home/hakim/Desktop/Phd/projects/MPC_casadi/build /home/hakim/Desktop/Phd/projects/MPC_casadi/build /home/hakim/Desktop/Phd/projects/MPC_casadi/build/CMakeFiles/example_use_nlp_external.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/example_use_nlp_external.dir/depend


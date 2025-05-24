cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(OpenDP)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_CXX_FLAGS "-O3")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb -O0 -Wall -Wextra -Wpedantic")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")


set(THREADS_PREFER_PTHREAD_FLAG ON)

# 使Gurobi成为可选依赖
option(USE_GUROBI "Use Gurobi optimizer" OFF)

if(USE_GUROBI)
    # 允许用户指定Gurobi路径
    set(GUROBI_HOME "" CACHE PATH "Path to Gurobi installation")
    
    if(GUROBI_HOME)
        include_directories(${GUROBI_HOME}/include)
        link_directories(${GUROBI_HOME}/lib)
        find_library(GUROBI gurobi_c++ PATHS ${GUROBI_HOME}/lib)
        if(GUROBI)
            add_definitions(-DUSE_GUROBI)
            message(STATUS "Gurobi found at ${GUROBI_HOME}")
        else()
            message(WARNING "Gurobi library not found at ${GUROBI_HOME}")
        endif()
    else()
        message(STATUS "Gurobi support disabled - GUROBI_HOME not set")
    endif()
endif()

set(OPENDP_HOME ${PROJECT_SOURCE_DIR})
set(DEFLIB_HOME
  ${PROJECT_SOURCE_DIR}/module/def/5.8-p027
)
set(LEFLIB_HOME
  ${PROJECT_SOURCE_DIR}/module/lef/5.8-p027
)

add_subdirectory(${DEFLIB_HOME})
add_subdirectory(${LEFLIB_HOME})


set( OPENDP_SRC
  src/assign.cpp
  src/check_legal.cpp
  src/main.cpp
  src/circuit.cpp
  src/mymeasure.cpp
  src/parser.cpp
  src/parser_helper.cpp
  src/place.cpp
  src/utility.cpp
  src/poabacus.cpp

  src/defParser.cpp
  src/lefParser.cpp

  src/circuitParser.cpp

  src/th/CTSTree.cpp
  src/th/ns.cpp
  src/th/Router.cpp

)

set ( OPENDP_HEADER
  src/circuit.h
  src/mymeasure.h
)

############################################################
# Include directories
############################################################

include_directories (
  ${LEFLIB_HOME}/lef
  ${LEFLIB_HOME}/lefzlib
  ${LEFLIB_HOME}/clef
  ${LEFLIB_HOME}/clefzlib

  ${DEFLIB_HOME}/def
  ${DEFLIB_HOME}/defzlib
  ${DEFLIB_HOME}/cdef
  ${DEFLIB_HOME}/cdefzlib
)

############################################################
# Libraries setting 
############################################################

add_library( opendplib STATIC
  ${OPENDP_SRC}
)

set_target_properties( opendplib 
  PROPERTIES OUTPUT_NAME "opendp"  
)

############################################################
# Executable setting 
############################################################

add_executable( opendp
  ${OPENDP_SRC}
)

find_package(Threads REQUIRED)

target_link_libraries( opendp
  PUBLIC
  Threads::Threads
  def
  defzlib
  cdef
  cdefzlib
  lef
  lefzlib
  clef
  clefzlib
)

# 仅在找到Gurobi时添加链接
if(USE_GUROBI AND GUROBI)
    target_link_libraries(opendp PUBLIC ${GUROBI})
endif()

add_dependencies( opendp def )
add_dependencies( opendp lef )

add_dependencies( opendplib def )
add_dependencies( opendplib lef )
############################################################
# Install  
############################################################

install(FILES ${OPENDP_HEADER} DESTINATION include)
install(TARGETS opendp DESTINATION bin)
install(TARGETS opendplib DESTINATION lib)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(opendp PUBLIC OpenMP::OpenMP_CXX)
endif()
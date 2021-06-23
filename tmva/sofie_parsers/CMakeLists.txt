# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

#sofie is built only if protobuf is found

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx_proto3")
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS_Development_Main} ${NUMPY_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTMVASofieParser
  HEADERS
   TMVA/RModelParser_ONNX.hxx
   TMVA/RModelParser_Keras.hxx
   ${PROTO_HDRS}
  SOURCES
    src/RModelParser_ONNX.cxx
    src/RModelParser_Keras.cxx
    ${PROTO_SRCS}
  LIBRARIES
    ${Protobuf_LIBRARIES}
    ${PYTHON_LIBRARIES_Development_Main}
  DEPENDENCIES
    TMVA
    ROOTTMVASofie
)

target_include_directories(ROOTTMVASofieParser PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_include_directories(ROOTTMVASofieParser PUBLIC
  ${Protobuf_INCLUDE_DIRS})
set_target_properties(ROOTTMVASofieParser PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

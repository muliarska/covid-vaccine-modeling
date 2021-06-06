cmake_minimum_required(VERSION 3.15)
project(covid_model)

set(CMAKE_CXX_STANDARD 14)

# Project source compilation
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/inc HEADERS)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src SOURCES)

add_executable(covid_model ${SOURCES} ${HEADERS} inc/base_header.hpp src/main.cpp src/config.cpp
        inc/constants.hpp inc/config.hpp src/covid_model_class.cpp inc/utils.hpp)

target_link_libraries(covid_model)

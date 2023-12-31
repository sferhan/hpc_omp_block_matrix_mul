cmake_minimum_required(VERSION 3.16)
project(hw2 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Release")

find_package(BLAS REQUIRED)
find_package(LIKWID QUIET)

include(FindOpenMP)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
    # You can set compiler flags or other options here
else()
    message(STATUS "OpenMP not found")
    # Handle the case where OpenMP is not available
endif()

# use cmake --log-level=debug to see DEBUG messages
if ( LIKWID_FOUND ) 
   message ( STATUS "Likwid: found")
else()
   message( STATUS  "Likwid: NOT found")
endif()

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

# Common library target for benchmarking.
add_library(benchmark OBJECT benchmark.cpp)
#target_compile_features(benchmark PRIVATE cxx_std_11)
target_compile_options(benchmark PRIVATE -fopenmp -march=native)
if ( LIKWID_FOUND )
    target_include_directories(benchmark PRIVATE ${LIKWID_INCLUDE_DIR})
    target_compile_definitions(benchmark PRIVATE -DLIKWID_PERFMON)
endif ()

add_executable(benchmark-basic-omp dgemm-basic-omp.cpp)
target_link_libraries(benchmark-basic-omp PRIVATE benchmark ${BLAS_LIBRARIES})
target_link_options(benchmark-basic-omp PRIVATE -fopenmp)
if ( LIKWID_FOUND )
   target_include_directories(benchmark-basic-omp PRIVATE ${LIKWID_INCLUDE_DIR})
   target_compile_definitions(benchmark-basic-omp PRIVATE -DLIKWID_PERFMON)
   target_link_libraries(benchmark-basic-omp PRIVATE ${LIKWID_LIBRARIES})
endif ()
#target_compile_features(benchmark-basic-omp PRIVATE c_std_11 c_restrict)
target_compile_options(benchmark-basic-omp PRIVATE -fopenmp -Wall -pedantic -march=native)
set(BENCHMARK "basic-omp")
configure_file(job.in job-${BENCHMARK})

add_executable(benchmark-blas dgemm-blas.cpp)
target_link_libraries(benchmark-blas PRIVATE benchmark ${BLAS_LIBRARIES})
target_link_options(benchmark-blas PRIVATE -fopenmp)
if ( LIKWID_FOUND )
   target_include_directories(benchmark-blas PRIVATE ${LIKWID_INCLUDE_DIR})
   target_compile_definitions(benchmark-blas PRIVATE -DLIKWID_PERFMON)
   target_link_libraries(benchmark-blas PRIVATE ${LIKWID_LIBRARIES})
endif ()
target_compile_features(benchmark-blas PRIVATE c_std_11 c_restrict)
target_compile_options(benchmark-blas PRIVATE -Wall -pedantic -march=native)
set(BENCHMARK "blas")
configure_file(job.in job-${BENCHMARK})

add_executable(benchmark-blocked-omp dgemm-blocked-omp.cpp)
add_library(bench-blocked-omp OBJECT benchmark.cpp)
target_compile_definitions(bench-blocked-omp PRIVATE -DBLOCKED)
target_compile_options(bench-blocked-omp PRIVATE -fopenmp)
target_link_libraries(benchmark-blocked-omp PRIVATE bench-blocked-omp ${BLAS_LIBRARIES})
target_link_options(benchmark-blocked-omp PRIVATE -fopenmp)
if ( LIKWID_FOUND )
   target_include_directories(benchmark-blocked-omp PRIVATE ${LIKWID_INCLUDE_DIR})
   target_compile_definitions(benchmark-blocked-omp PRIVATE -DLIKWID_PERFMON)
   target_include_directories(bench-blocked-omp PRIVATE ${LIKWID_INCLUDE_DIR})
   target_compile_definitions(bench-blocked-omp PRIVATE -DLIKWID_PERFMON)
   target_link_libraries(benchmark-blocked-omp PRIVATE ${LIKWID_LIBRARIES})
endif ()
target_compile_options(benchmark-blocked-omp PRIVATE -fopenmp -Wall -pedantic -march=native)
set(BENCHMARK "blocked-omp")
configure_file(job.in job-${BENCHMARK})

# eof

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
add_executable(tester tester.cpp)
target_link_libraries(tester studentlib -fopenmp)
target_compile_options(tester PRIVATE -Ofast -march=native -fopenmp)

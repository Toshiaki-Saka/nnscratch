# SPDX-License-Identifier: Apache-2.0
# Each test is a standalone executable registered with CTest. The harness
# signals failure through the process exit code, so no test framework is needed.

foreach(test test_tensor test_gradcheck test_optimizer)
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE nnscratch nnscratch_warnings)
    target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test} COMMAND ${test})
endforeach()

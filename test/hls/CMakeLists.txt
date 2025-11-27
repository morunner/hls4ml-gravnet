set(HLS4ML_GRAVNET_DIR ${CMAKE_SOURCE_DIR}/hls4ml_gravnet/hls)
set(HLS4ML_VIVADO_AP_TYPES_DIR ${CMAKE_SOURCE_DIR}/hls4ml/hls4ml/templates/vivado/ap_types)

add_executable(gravnet_test_runner src/test_gravnet.cc)

target_include_directories(gravnet_test_runner PRIVATE include ${HLS4ML_GRAVNET_DIR} ${HLS4ML_VIVADO_AP_TYPES_DIR})

target_link_libraries(gravnet_test_runner PRIVATE gtest_main gtest gmock pthread)

include(GoogleTest)
gtest_discover_tests(gravnet_test_runner)


find_package(Catch2 CONFIG REQUIRED)

add_executable(test_libtorch20 CPUTensor.cpp)
target_compile_options(test_libtorch20 PUBLIC -fconcepts -std=c++2a -fmax-errors=5)

target_include_directories(
  test_libtorch20 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(test_libtorch20 Catch2::Catch2)

install(TARGETS test_libtorch20
        COMPONENT test
        # EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
	    RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

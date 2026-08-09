cmake_minimum_required(VERSION 3.5)
project(g1_logging)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/cbf_csv_logger_node.py
  RENAME cbf_csv_logger_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/cbf_csv_plotter_node.py
  RENAME cbf_csv_plotter_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(g1_logging_test_env
    "JAX_PLATFORMS=cpu"
    "JAX_ENABLE_X64=true"
  )
  ament_add_pytest_test(test_csv_logging
    test/test_csv_logging.py
    ENV ${g1_logging_test_env}
  )
  ament_add_pytest_test(test_geometry
    test/test_geometry.py
    ENV ${g1_logging_test_env}
  )
  ament_add_pytest_test(test_plotter
    test/test_plotter.py
    ENV ${g1_logging_test_env}
  )
endif()

ament_package()

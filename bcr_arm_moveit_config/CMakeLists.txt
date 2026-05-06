cmake_minimum_required(VERSION 3.22)
project(bcr_arm_moveit_config)

find_package(ament_cmake REQUIRED)

# Install launch files
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE
  )
endif()

# Install MoveIt config files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(
  PROGRAMS
    scripts/add_world_collision_objects.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install MoveIt Setup Assistant metadata
install(
  FILES .setup_assistant
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

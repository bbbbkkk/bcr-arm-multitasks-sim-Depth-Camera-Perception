cmake_minimum_required(VERSION 3.8)
project(bcr_arm_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# find_package(urdf_tutorial REQUIRED) # not needed in production, just for testing urdf and all

# Install directories
install(DIRECTORY
  urdf
  meshes
  launch
  rviz
  # config
  DESTINATION share/${PROJECT_NAME}
)


ament_package()

cmake_minimum_required(VERSION 3.21)
project(helloworld)
find_package(Eigen3 REQUIRED)
# Find the Ceres Solver library
find_package(Ceres REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common io search visualization)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")


include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

add_executable(helloworld
        main.cpp)
add_executable(unorderedmap
unorderedmap.cpp)
add_executable(multimap
multimap.cpp)
add_executable(ellipDist
ellipDist.cpp)

add_executable(ellipDistCMU
ellipDistCMU.cpp)  
add_executable(wassersteinDistance
wassersteinDistance.cpp        
)

add_executable(temp
temp.cpp        
)
add_executable(isPtsInEllipsoid isPtsInEllipsoid.cpp)
add_executable(bgkgradtest bgkgradtest.cpp)
add_executable(arrayMat3d arrayMat3d.cpp)
add_executable(pclRsearch pclRsearch.cpp)  
add_executable(pclPc2Extract pclPc2Extract.cpp)
add_executable(unordered_set unordered_set.cpp)
add_executable(templateAstar templateAstar.cpp)
# add_executable(ellipDistCeres ellipDistCeres.cpp)
# target_link_libraries(ellipDistCeres ${CERES_LIBRARIES} Eigen3::Eigen)

target_link_libraries(pclRsearch ${PCL_LIBRARIES})
target_link_libraries(pclPc2Extract ${PCL_LIBRARIES})



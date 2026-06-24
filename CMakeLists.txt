cmake_minimum_required(VERSION 3.5)

project(chess LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets Sql REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets Sql REQUIRED)

# src
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB SRC_FILES
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.hpp
    ${SRC_DIR}/*.cpp
    ${SRC_DIR}/*.ui)
include_directories(${SRC_DIR})

# rl
file(GLOB RL_FILES
    ${SRC_DIR}/rl/*.h
    ${SRC_DIR}/rl/*.hpp
    ${SRC_DIR}/rl/*.cpp)
list(APPEND SRC_FILES ${RL_FILES})

include_directories(${SRC_DIR}/rl)


add_executable(chess ${SRC_FILES})

target_link_libraries(chess PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Sql
)

# Test target
add_executable(test_ab ${CMAKE_CURRENT_SOURCE_DIR}/test/test_main.cpp
    ${SRC_DIR}/chess.cpp
    ${SRC_DIR}/gamedb.cpp
    ${SRC_DIR}/pos.cpp
    ${SRC_DIR}/stone.cpp
    ${SRC_DIR}/abagent.cpp)

target_link_libraries(test_ab PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Sql
)

# MCTS Test target
add_executable(test_mcts ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mcts_main.cpp
    ${SRC_DIR}/chess.cpp
    ${SRC_DIR}/gamedb.cpp
    ${SRC_DIR}/mcts.cpp
    ${SRC_DIR}/pos.cpp
    ${SRC_DIR}/stone.cpp
    ${SRC_DIR}/abagent.cpp)

target_link_libraries(test_mcts PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Sql
)

# Policy Gradient Agent Test target
add_executable(test_pg ${CMAKE_CURRENT_SOURCE_DIR}/test/test_pg_main.cpp
    ${SRC_DIR}/chess.cpp
    ${SRC_DIR}/gamedb.cpp
    ${SRC_DIR}/pos.cpp
    ${SRC_DIR}/stone.cpp
    ${SRC_DIR}/pgagent.cpp
    ${RL_FILES})

target_link_libraries(test_pg PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Sql
)

# DQN Agent Test target
add_executable(test_dqn ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dqn_main.cpp
    ${SRC_DIR}/chess.cpp
    ${SRC_DIR}/gamedb.cpp
    ${SRC_DIR}/pos.cpp
    ${SRC_DIR}/stone.cpp
    ${SRC_DIR}/dqnagent.cpp
    ${RL_FILES})

target_link_libraries(test_dqn PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Sql
)

# PPO+MCTS (AlphaZero-style) Agent Test target
add_executable(test_ppomcts ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ppomcts_main.cpp
    ${SRC_DIR}/chess.cpp
    ${SRC_DIR}/gamedb.cpp
    ${SRC_DIR}/pos.cpp
    ${SRC_DIR}/stone.cpp
    ${SRC_DIR}/mcts.cpp
    ${SRC_DIR}/ppomcts_agent.cpp
    ${RL_FILES})

target_link_libraries(test_ppomcts PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Sql
)

# DQN+MCTS Agent Test target
add_executable(test_dqnmcts ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dqnmcts_main.cpp
    ${SRC_DIR}/chess.cpp
    ${SRC_DIR}/gamedb.cpp
    ${SRC_DIR}/pos.cpp
    ${SRC_DIR}/stone.cpp
    ${SRC_DIR}/mcts.cpp
    ${SRC_DIR}/dqnmcts_agent.cpp
    ${RL_FILES})

target_link_libraries(test_dqnmcts PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Sql
)

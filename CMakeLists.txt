cmake_minimum_required(VERSION 3.16)

project(chess LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Single-config generators (Ninja / Makefiles) default to Release so a plain
# `cmake -S . -B build` does not silently produce an unoptimised build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

option(CHESS_BUILD_TESTS "Build the test executables" ON)
option(CHESS_USE_PCH
       "Precompile rl/tensor.hpp + rl/layer.h for the RL library (CMake >= 3.16)"
       OFF)
option(CHESS_ENABLE_AVX2
       "Compile the RL core with AVX2 so rl/simd_ops.hpp uses the AVX2 kernels"
       ON)

# Lets `ctest` see the test executables declared at the bottom of this file.
# Harmless when CHESS_BUILD_TESTS is OFF.
enable_testing()

# ============================================================================
# Build organisation (mirrors the snakeAI project layout)
# ----------------------------------------------------------------------------
# The RL layer (src/rl/) is pure C++ and never touches Qt. It is built ONCE as
# the static library RL_CORE. The previous build file globbed
# src/rl/*.{h,hpp,cpp} into every executable and had CMAKE_AUTOMOC ON globally,
# so all 6 targets recompiled the same translation units (and AUTOMOC scanned
# the RL headers in each of them) — roughly 6x the compiler work for code no
# target needs more than one copy of.
# ============================================================================

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets Sql REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets Sql REQUIRED)

# src
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================================================ RL core (no Qt)
# Explicit source lists instead of file(GLOB): deterministic, no re-glob on
# configure, and headers are not passed to the compiler as translation units.
# Add new rl/*.cpp here when you add a file.
set(RL_SOURCES
    ${SRC_DIR}/rl/bcq.cpp
    ${SRC_DIR}/rl/convdqn.cpp
    ${SRC_DIR}/rl/convpg.cpp
    ${SRC_DIR}/rl/ddpg.cpp
    ${SRC_DIR}/rl/dpg.cpp
    ${SRC_DIR}/rl/dqn.cpp
    ${SRC_DIR}/rl/drpg.cpp
    ${SRC_DIR}/rl/gru.cpp
    ${SRC_DIR}/rl/lstm.cpp
    ${SRC_DIR}/rl/mamba.cpp
    ${SRC_DIR}/rl/mpg.cpp
    ${SRC_DIR}/rl/ppo.cpp
    ${SRC_DIR}/rl/qlstm.cpp
    ${SRC_DIR}/rl/sac.cpp
    ${SRC_DIR}/rl/ssm.cpp
    ${SRC_DIR}/rl/trpo.cpp
    ${SRC_DIR}/rl/util.cpp
)
add_library(RL_CORE STATIC ${RL_SOURCES})

# PUBLIC so consumers may `#include "rl/xxx.h"` as well as `#include "xxx.h"`.
target_include_directories(RL_CORE PUBLIC ${SRC_DIR})
set_target_properties(RL_CORE PROPERTIES
    AUTOMOC OFF
    AUTOUIC OFF
    AUTORCC OFF)

# ---------------------------------------------------------------------------
# Workaround: make header changes actually trigger rebuilds.
#
# On this machine MSVC emits its /showIncludes note in the localized form
# ("注意: 包含文件:") while the prefix CMake recorded during compiler detection
# is a different (mis-decoded) byte sequence, so ninja matches nothing and
# stores NO header dependencies. Consequence: editing rl/tensor.hpp or
# rl/layer.h rebuilds nothing and incremental builds silently produce STALE
# binaries. Declaring the headers as explicit object dependencies of every
# source keeps incremental builds correct regardless of locale.
# ---------------------------------------------------------------------------
file(GLOB RL_HEADER_DEPS
    ${SRC_DIR}/rl/*.h
    ${SRC_DIR}/rl/*.hpp)
set_property(SOURCE ${RL_SOURCES} APPEND PROPERTY OBJECT_DEPENDS ${RL_HEADER_DEPS})

file(GLOB APP_HEADER_DEPS
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.hpp)

# ---------------------------------------------------------------------------
# 同一个 workaround 必须应用到**每一个**目标, 不只是 RL_CORE 和 chess。
# 漏掉测试目标时症状不明显但很恶劣: 改了 src/chess.h / src/stone.h 之后
# test_ab / test_mcts / test_pg ... 的目标文件不会重编, 于是跑的是新旧混合的
# 二进制 —— 既不报错也看不出来, 只是慢或者行为对不上
# (实测: 同一份源码, 单独用 cl 编出来的 test_ab 跑 5 s, CMake 增量构建出来的
#  跑 23 s, 搜索基准 1910 ms vs 3133 ms, 就是这个原因)。
# ---------------------------------------------------------------------------
function(chess_add_header_deps target)
    get_target_property(_srcs ${target} SOURCES)
    set_property(SOURCE ${_srcs} APPEND PROPERTY OBJECT_DEPENDS
                 ${APP_HEADER_DEPS} ${RL_HEADER_DEPS})
endfunction()

# /bigobj: the RL translation units instantiate a lot of templates
# (tensor.hpp / layer.h / attention.hpp / moe.hpp), which overflows the default
# section limit of an MSVC object file.
target_compile_options(RL_CORE PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

# ---------------------------------------------------------------------------
# SIMD: rl/simd_ops.hpp dispatches the hot Tensor_ operations to the kernels in
# rl/simd/{sse2func,avx2func}.hpp (copied from the N-spirits project, where
# tensorsi.hpp drives them).  Those kernels select their instruction set from
# the __AVX2__ / __SSE2__ macros, and MSVC's x64 baseline only defines __SSE2__,
# so /arch:AVX2 is what actually unlocks them: measured on rl/tensor.hpp's
# kernels the AVX2 ones are ~10.5x faster on GEMV (w*x, the hot path in
# Layer/attention), ~8.2x on the reductions and 1.4-1.8x on the large matmuls.
#
# The catch: /arch:AVX2 makes the resulting binary REQUIRE an AVX2-capable CPU
# (it will fault with an illegal instruction otherwise), so the option must not
# be turned on blindly for a system that cannot run it.  The probe below asks the
# build host once, at configure time, and reports the decision.
#
# PUBLIC because tensor.hpp is a header: every target that instantiates its
# inline operators (the chess app and the agent tests as well) must agree on the
# instruction set, otherwise those TUs would compile the scalar fallback.
#
# Set -DCHESS_ENABLE_AVX2=OFF to build a binary that runs on CPUs without AVX2
# (correct either way, just slower).
# ---------------------------------------------------------------------------
if(CHESS_ENABLE_AVX2)
    # CPUID probe, same order as RL::cpuinfo: AVX2 bit first, then confirm the OS
    # has enabled YMM state (OSXSAVE + AVX + XCR0 bits 1..2).
    file(WRITE ${CMAKE_BINARY_DIR}/chess_avx2_probe.cpp
"#include <cstdio>
#if defined(_MSC_VER)
#include <intrin.h>
static unsigned long long xgetbv0()
{
    return _xgetbv(0);
}
#else
#include <cpuid.h>
static void cpuidEx(int leaf, int sub, int r[4])
{
    unsigned int a, b, c, d;
    __cpuid_count(leaf, sub, a, b, c, d);
    r[0] = (int)a; r[1] = (int)b; r[2] = (int)c; r[3] = (int)d;
}
static void cpuid1(int leaf, int r[4])
{
    unsigned int a, b, c, d;
    __cpuid(leaf, a, b, c, d);
    r[0] = (int)a; r[1] = (int)b; r[2] = (int)c; r[3] = (int)d;
}
static unsigned long long xgetbv0()
{
    unsigned int lo, hi;
    __asm__ __volatile__(\"xgetbv\" : \"=a\"(lo), \"=d\"(hi) : \"c\"(0));
    return ((unsigned long long)hi << 32) | lo;
}
#endif
int main()
{
    int r[4] = {0,0,0,0};
#if defined(_MSC_VER)
    __cpuid(r, 0);
#else
    cpuid1(0, r);
#endif
    if (r[0] < 7) { std::printf(\"0\"); return 0; }
#if defined(_MSC_VER)
    __cpuid(r, 1);
#else
    cpuid1(1, r);
#endif
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    const bool avx     = (r[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) { std::printf(\"0\"); return 0; }
    if ((xgetbv0() & 0x6ULL) != 0x6ULL) { std::printf(\"0\"); return 0; }
#if defined(_MSC_VER)
    __cpuidex(r, 7, 0);
#else
    cpuidEx(7, 0, r);
#endif
    std::printf(\"%d\", (r[1] & (1 << 5)) != 0 ? 1 : 0);
    return 0;
}
")
    try_run(CHESS_AVX2_PROBE_RUN CHESS_AVX2_PROBE_COMPILED
            ${CMAKE_BINARY_DIR}/chess_avx2_probe_bin
            ${CMAKE_BINARY_DIR}/chess_avx2_probe.cpp
            RUN_OUTPUT_VARIABLE CHESS_AVX2_PROBE_OUTPUT)
    if(NOT CHESS_AVX2_PROBE_COMPILED)
        message(STATUS "SIMD: CPU probe did not build - keeping /arch:AVX2 as requested")
    elseif(CHESS_AVX2_PROBE_OUTPUT STREQUAL "1")
        message(STATUS "SIMD: this system reports AVX2 -> enabling /arch:AVX2 (AVX2 kernels)")
        target_compile_options(RL_CORE PUBLIC
            $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>)
    else()
        message(WARNING
            "SIMD: this system does NOT report AVX2 (or the OS has not enabled YMM state).\n"
            "      /arch:AVX2 is therefore NOT applied: the binary would fault with an\n"
            "      illegal instruction on this machine. The SSE2 kernels are used instead.\n"
            "      Pass -DCHESS_ENABLE_AVX2=ON explicitly to override, or build on a\n"
            "      host that has AVX2 to get the AVX2 kernels.")
    endif()
endif()

# Detect Ninja once, explicitly, so the MSVC condition below cannot be misread
# by CMake's if() operator precedence.
set(CHESS_GENERATOR_IS_NINJA FALSE)
if(CMAKE_GENERATOR MATCHES "Ninja")
    set(CHESS_GENERATOR_IS_NINJA TRUE)
endif()

if(MSVC AND NOT CHESS_GENERATOR_IS_NINJA)
    # /MP parallelises the TUs *within* one target. With Ninja that is
    # counterproductive (Ninja already runs many compilers in parallel), so it
    # is only added for MSBuild-style generators.
    target_compile_options(RL_CORE PUBLIC /MP)
endif()

if(CHESS_USE_PCH)
    # Opt-in: must be re-validated against the toolchain before becoming the
    # default. rl/q*.cpp are the only TUs that do not include these two headers
    # at all; MSVC skips the PCH for them automatically.
    target_precompile_headers(RL_CORE PRIVATE
        ${SRC_DIR}/rl/tensor.hpp
        ${SRC_DIR}/rl/layer.h)
endif()

# =============================================================== GUI program
set(APP_SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/abagent.cpp
    ${SRC_DIR}/chess.cpp
    ${SRC_DIR}/chessboard.cpp
    ${SRC_DIR}/dqnagent.cpp
    ${SRC_DIR}/dqnmcts_agent.cpp
    ${SRC_DIR}/evagent.cpp
    ${SRC_DIR}/gamedb.cpp
    ${SRC_DIR}/mainwindow.cpp
    ${SRC_DIR}/mcts.cpp
    ${SRC_DIR}/pgagent.cpp
    ${SRC_DIR}/pos.cpp
    ${SRC_DIR}/ppomcts_agent.cpp
    ${SRC_DIR}/sacazagent.cpp
    ${SRC_DIR}/stone.cpp
    ${SRC_DIR}/thinkingindicator.cpp
    ${SRC_DIR}/metricsview.cpp
    ${SRC_DIR}/busydialog.cpp
    ${SRC_DIR}/mainwindow.ui
    ${SRC_DIR}/res.qrc
)
# ---------------------------------------------------------------------------
# Windows 可执行文件图标 (PE 资源里的 RT_GROUP_ICON)。
# 注意它和"运行时窗口图标"是两件事:
#   * src/app.rc / app.ico  -> 资源管理器/任务栏固定时看到的 exe 图标 (这里)
#   * src/res.qrc / app.png -> main.cpp 里 QApplication::setWindowIcon (窗口/任务栏)
# 两个都加上才算完整 (只做一个会出现"窗口有图标、文件没有")。
# 图标由 tools/make_app_icon.ps1 生成, 不是手搓的二进制。
if (WIN32)
    list(APPEND APP_SOURCES ${SRC_DIR}/app.rc)
endif()
add_executable(chess ${APP_SOURCES})

chess_add_header_deps(chess)

# Qt's code generators are needed by this target only.
set_target_properties(chess PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON)

target_link_libraries(chess PRIVATE
    RL_CORE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Sql)

target_compile_options(chess PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

if(CHESS_USE_PCH)
    target_precompile_headers(chess PRIVATE
        ${SRC_DIR}/rl/tensor.hpp
        ${SRC_DIR}/rl/layer.h)
endif()

# ===================================================================== tests
if(CHESS_BUILD_TESTS)
    # Alpha-Beta and MCTS tests only use the chess engine, so they do not link
    # RL_CORE. The remaining four drive an RL agent and do.
    foreach(t IN ITEMS test_ab test_mcts)
        if(t STREQUAL "test_ab")
            set(_src ${CMAKE_CURRENT_SOURCE_DIR}/test/test_main.cpp)
            set(_extra ${SRC_DIR}/abagent.cpp)
        else()
            set(_src ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mcts_main.cpp)
            set(_extra ${SRC_DIR}/mcts.cpp ${SRC_DIR}/abagent.cpp)
        endif()
        add_executable(${t} ${_src}
            ${SRC_DIR}/chess.cpp
            ${SRC_DIR}/gamedb.cpp
            ${SRC_DIR}/pos.cpp
            ${SRC_DIR}/stone.cpp
            ${_extra})
        target_include_directories(${t} PRIVATE ${SRC_DIR})
        target_link_libraries(${t} PRIVATE
            Qt${QT_VERSION_MAJOR}::Core
            Qt${QT_VERSION_MAJOR}::Sql)
        target_compile_options(${t} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
        set_target_properties(${t} PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF)
        chess_add_header_deps(${t})
        add_test(NAME ${t} COMMAND ${t})
        set_tests_properties(${t} PROPERTIES TIMEOUT 900)
    endforeach()

    # ---------------------------------------------------------------- rules test
    # 规则引擎回归测试: 只依赖 src/chess.cpp / pos.cpp / stone.cpp, 完全不碰 Qt,
    # 所以它既不链 RL_CORE 也不链 Qt —— 秒级完成, 并且对每条棋规都有断言
    # (初始 44 个合法走法、应将、飞将/照面、将杀、三次重复、60 回合限着、
    #  落子/回退往返一致性)。仓库里其它 test_* 只有 printf, 抓不到这些。
    add_executable(test_rules
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_rules_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp)
    chess_add_header_deps(test_rules)
    target_include_directories(test_rules PRIVATE ${SRC_DIR})
    set_target_properties(test_rules PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    add_test(NAME test_rules COMMAND test_rules)
    set_tests_properties(test_rules PROPERTIES TIMEOUT 120)

    # The four RL-agent tests. These are plain C++ programs that only include
    # src/ + src/rl/ headers, so they link RL_CORE and skip Qt's code
    # generators; Qt::Core/Qt::Sql are still needed by gamedb.cpp.
    foreach(t IN ITEMS test_pg test_dqn test_ppomcts test_dqnmcts)
        if(t STREQUAL "test_pg")
            set(_src ${CMAKE_CURRENT_SOURCE_DIR}/test/test_pg_main.cpp)
            set(_agents ${SRC_DIR}/pgagent.cpp)
            set(_extra "")
        elseif(t STREQUAL "test_dqn")
            set(_src ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dqn_main.cpp)
            set(_agents ${SRC_DIR}/dqnagent.cpp)
            set(_extra "")
        elseif(t STREQUAL "test_ppomcts")
            set(_src ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ppomcts_main.cpp)
            set(_agents ${SRC_DIR}/ppomcts_agent.cpp)
            set(_extra ${SRC_DIR}/mcts.cpp)
        else()
            set(_src ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dqnmcts_main.cpp)
            set(_agents ${SRC_DIR}/dqnmcts_agent.cpp)
            set(_extra ${SRC_DIR}/mcts.cpp)
        endif()
        add_executable(${t} ${_src}
            ${SRC_DIR}/chess.cpp
            ${SRC_DIR}/gamedb.cpp
            ${SRC_DIR}/pos.cpp
            ${SRC_DIR}/stone.cpp
            ${_agents}
            ${_extra})
        target_include_directories(${t} PRIVATE ${SRC_DIR})
        target_link_libraries(${t} PRIVATE
            RL_CORE
            Qt${QT_VERSION_MAJOR}::Core
            Qt${QT_VERSION_MAJOR}::Sql)
        target_compile_options(${t} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
        set_target_properties(${t} PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF)
        chess_add_header_deps(${t})
        # Deliberately NOT registered with CTest, exactly as snakeAI treats its
        # `bench_agent_game`: these are stochastic self-play TRAINING benchmarks
        # that print a learning curve rather than a pass/fail, and they run for
        # minutes. Measured on this machine (Release, MSVC 2022, Qt 6.9.2),
        # with / without the Tensor::MM optimization (docs/issues_review.md B19):
        #     test_pg        346 s  ->   26 s
        #     test_dqn      > 900 s ->  536 s   (timed out before)
        #     test_ppomcts  > 900 s ->  233 s   (timed out before)
        #     test_dqnmcts  > 900 s -> 1299 s   (timed out before)
        # The old numbers came from Tensor::MM::* measuring 0.11 GFLOP/s: its
        # inner loop indexed through operator()/posOf() twice per
        # multiply-accumulate. Rewriting the kernels with flat pointers and
        # hoisted strides made them ~190x faster (20 GFLOP/s).
        #
        # Build them and run them by hand when you want the numbers; putting them
        # in the default suite only produces timeouts that look like failures.
    endforeach()

    # ------------------------------------------------------------------ EVAB test
    # EVABAgent (学会评估的 Alpha-Beta, 见 docs/agent_evab_design.md) 的训练与对抗
    # 测试。它既要 alpha-beta/MCTS/DQN/PG 这些对手, 也要 RL_CORE 的价值网络,
    # 所以链接面最宽。
    # 默认参数下几分钟内跑完并打出胜/和/负, 因此不注册 ctest (同 test_pg 等)。
    add_executable(test_evab
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_evab_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/evagent.cpp
        ${SRC_DIR}/abagent.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/dqnagent.cpp
        ${SRC_DIR}/pgagent.cpp)
    target_include_directories(test_evab PRIVATE ${SRC_DIR})
    target_link_libraries(test_evab PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(test_evab PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_evab PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_evab)

    # ------------------------------------------------- 走子前探索+预训练 的验证
    # 盯住"探索不能改动真棋局"这条性质 (象棋没法像贪吃蛇那样在局部坐标上模拟,
    # 试走只能落在真棋盘上, 所以必须原样回退), 并检查探索后仍能给出合法走法。
    add_executable(test_pretrain
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_pretrain_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/evagent.cpp
        ${SRC_DIR}/abagent.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/dqnagent.cpp
        ${SRC_DIR}/pgagent.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/dqnmcts_agent.cpp)
    target_include_directories(test_pretrain PRIVATE ${SRC_DIR})
    target_link_libraries(test_pretrain PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(test_pretrain PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_pretrain PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_pretrain)
    add_test(NAME test_pretrain COMMAND test_pretrain)
    set_tests_properties(test_pretrain PROPERTIES TIMEOUT 600)

    # ------------------------------------------------- Agent 对 Agent 对弈 (arena)
    # chessboard.cpp 是 QWidget, 所以这个测试要链 Qt Widgets 并起一个 QApplication
    # (用 offscreen 平台插件, 无显示器也能跑)。它验证的是对弈统计本身:
    # 每局交换先后手、比分按参赛者而非红黑归属、到上限判和、中止生效。
    # 通过 setMaxPliesPerGame(4) 把每局压到 4 手, 于是几秒内就能跑完整场。
    add_executable(test_match
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_match_main.cpp
        ${SRC_DIR}/chessboard.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/evagent.cpp
        ${SRC_DIR}/abagent.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/dqnagent.cpp
        ${SRC_DIR}/pgagent.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/dqnmcts_agent.cpp
        ${SRC_DIR}/sacazagent.cpp)
    target_include_directories(test_match PRIVATE ${SRC_DIR})
    target_link_libraries(test_match PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(test_match PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_match PROPERTIES
        AUTOMOC ON
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_match)
    add_test(NAME test_match COMMAND test_match)
    # ctest 从普通 shell 启动时 PATH 里没有 Qt 的 bin, test_match 会以
    # 0xc0000135 (STATUS_DLL_NOT_FOUND) 立刻退出 —— 看起来像测试挂了, 其实
    # 是环境问题。这里把 Qt 的 bin 目录显式前置到测试的 PATH 里。
    # 用 ENVIRONMENT_MODIFICATION 而不是 ENVIRONMENT, 因为 PATH 自己就是用
    # ';' 分隔的, 直接塞进 ENVIRONMENT 会被当成多条环境设置。
    set(_qt_bin_dir "${Qt6_DIR}/../../../bin")
    cmake_path(NORMAL_PATH _qt_bin_dir)
    set_tests_properties(test_match PROPERTIES TIMEOUT 600
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_qt_bin_dir}")

    # ------------------------------------------- 梯度传播正确性 (SIMD 之后)
    # 用中心差分核对 Net::backward 填出来的解析梯度, 覆盖普通 MLP 与 DQN 的
    # 真实骨干 (MOE<16,16> + TransformerBlock<16> + TanhNorm + Sigmoid),
    # 另外直接把 MM 内核"累加 vs 覆盖"的语义差异量出来。
    # 只用 RL_CORE, 不需要 Qt / 象棋代码。
    add_executable(test_grad
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_grad_main.cpp)
    target_include_directories(test_grad PRIVATE ${SRC_DIR})
    target_link_libraries(test_grad PRIVATE RL_CORE)
    target_compile_options(test_grad PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_grad PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_grad)
    add_test(NAME test_grad COMMAND test_grad)
    set_tests_properties(test_grad PROPERTIES TIMEOUT 600)

    # ------------------------------------------- 权重文件格式
    # 权重文件是训练成果的唯一载体: 老格式是十进制文本 (有损! 存一次读回来权重就
    # 漂移 ~1e-6)、没有校验、写一半崩掉就毁文件。这里逐条验证新格式: 逐比特无损、
    # 结构指纹 + 每张量 CRC32、原子写入、以及**老文件仍然能读**。
    add_executable(test_weights
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_weights_main.cpp)
    target_include_directories(test_weights PRIVATE ${SRC_DIR})
    target_link_libraries(test_weights PRIVATE RL_CORE)
    target_compile_options(test_weights PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_weights PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_weights)
    add_test(NAME test_weights COMMAND test_weights)
    set_tests_properties(test_weights PROPERTIES TIMEOUT 300)

    # ------------------------------------------- 稀疏路由 MoE    # 上游 moe.hpp 是稠密的 (全算 E 个专家), 在 d_model=1260 上不可用;
    # sparse_moe.hpp 只算门控选中的 top-k 个。这个测试盯四件事:
    #   * 前向真的跳过了未选中的专家 (改它们的权重输出要逐位不变)
    #   * TopK==E 时与上游 MOE 的前向/反向完全一致 (证明门控约定没变)
    #   * 反向的解析梯度 vs 有限差分 (含"未选中专家的梯度必须恰好为 0")
    #   * 负载均衡辅助损失的梯度 vs 有限差分, 以及它确实把坍缩的路由拉回来
    add_executable(test_sparse_moe
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sparse_moe_main.cpp)
    target_include_directories(test_sparse_moe PRIVATE ${SRC_DIR})
    target_link_libraries(test_sparse_moe PRIVATE RL_CORE)
    target_compile_options(test_sparse_moe PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_sparse_moe PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_sparse_moe)
    add_test(NAME test_sparse_moe COMMAND test_sparse_moe)
    set_tests_properties(test_sparse_moe PROPERTIES TIMEOUT 600)

    # ------------------------------------------- 骨干 A/B/C/D 等时对弈基准
    # **不进 ctest**: 它跑真实对局, 依赖随机开局, 而且 TB 专家骨干一步几十毫秒,
    # 放进 ctest 既慢又会偶发失败。手动跑:
    #   cmake --build <build> --target bench_moe && <build>/bench_moe --games=4
    add_executable(bench_moe
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_moe_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/sacazagent.cpp)
    target_include_directories(bench_moe PRIVATE ${SRC_DIR})
    target_link_libraries(bench_moe PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_moe PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_moe PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_moe)

    # ------------------------------------------- PPO+MCTS vs Alpha-Beta 静默对弈
    # **不进 ctest**: 同 bench_moe —— 跑真实对局、依赖随机开局, 而且默认双方用的是
    # 随机初始权重, 几局之内翻转结论是常事。手动跑:
    #   cmake --build <build> --target bench_ppo_vs_ab
    #   <build>/bench_ppo_vs_ab --games=6 --budget=90      # 等时间
    #   <build>/bench_ppo_vs_ab --load=weights/ppomcts_agent.dat   # 用 GUI 存下的权重
    add_executable(bench_ppo_vs_ab
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_ppo_vs_ab_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/abagent.cpp)
    target_include_directories(bench_ppo_vs_ab PRIVATE ${SRC_DIR})
    target_link_libraries(bench_ppo_vs_ab PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_ppo_vs_ab PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_ppo_vs_ab PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_ppo_vs_ab)

    # ------------------------------------------- PPO+MCTS 多线程分身训练吞吐
    # **不进 ctest**: 同 bench_ppo_vs_ab —— 跑真实自对弈、依赖线程调度与随机开局,
    # 耗时以分钟计。手动跑:
    #   cmake --build <build> --target bench_ppo_mt
    #   <build>/bench_ppo_mt --games=8 --sweep=1,2,4,6,8 --sims=80
    # 它量的不是棋力, 而是"单位时间产出/消化多少样本", 以及 worker 数与加速比、
    # 并行效率的关系 (串行基线用同样的学习配置, 见 test/bench_ppo_mt_main.cpp)。
    add_executable(bench_ppo_mt
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_ppo_mt_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    target_include_directories(bench_ppo_mt PRIVATE ${SRC_DIR})
    target_link_libraries(bench_ppo_mt PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_ppo_mt PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_ppo_mt PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_ppo_mt)

    # ------------------------------------------- SAC + MCTS + AlphaZero agent
    # 盯四件事: 掩码 softmax 的雅可比(有限差分)、搜索给出的走法合法性、
    # 软价值里 α 熵项的形式、critic 是否真的把 Q 拉向目标。
    add_executable(test_sacaz
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sacaz_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/sacazagent.cpp)
    target_include_directories(test_sacaz PRIVATE ${SRC_DIR})
    target_link_libraries(test_sacaz PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(test_sacaz PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_sacaz PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_sacaz)
    add_test(NAME test_sacaz COMMAND test_sacaz)
    set_tests_properties(test_sacaz PROPERTIES TIMEOUT 600)

    # 想用 AddressSanitizer 查这类问题 (arena 里"agent 返回非法走法"/堆损坏),
    # 必须**整个构建**都带 /fsanitize=address —— 只给一个目标加会在链接时报
    # LNK2038 (annotate_string/annotate_vector 不匹配, RL_CORE 没带 ASan)。
    # 可行的做法是单独开一个构建目录:
    #
    #   cmake -S . -B build/asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
    #         -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64 ^
    #         -DCMAKE_CXX_FLAGS="/fsanitize=address /Zi" ^
    #         -DCMAKE_EXE_LINKER_FLAGS="/fsanitize=address"
    #   cmake --build build/asan --target <目标>
    #
    # 运行前把 clang_rt.asan_dynamic-x86_64.dll 所在目录加进 PATH
    # (VC\Tools\MSVC\<ver>\bin\Hostx64\x64), 并设 ASAN_OPTIONS=detect_leaks=0
    # (Qt 自己的分配会被报成泄漏)。
endif()

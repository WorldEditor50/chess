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
    ${SRC_DIR}/sacazlegacyagent.cpp
    ${SRC_DIR}/dqnabagent.cpp
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

    # ------------------------------------------------------- diagnostics matrix
    # 诊断指标矩阵 (rl/diag.h): 棋力是滞后指标, 这一层量的是"信号健康度"。
    # 它做两件事:
    #   1. test_diag   —— **解析验证**每个指标公式 (熵/CE/KL/EV/CV/校准/qForParent),
    #      外加一个端到端接线检查 (rootDiag 的不变量 + evalRootNoise 开关真的接上了)。
    #      指标算错比没有指标更贵, 所以这一层必须进 ctest。
    #   2. bench_diag  —— **不进 ctest** (跑真实对局/战术题, 依赖随机开局)。
    #   test_diag 用小网络 (hidden=16), 所以只链 RL_CORE, 秒级完成。
    add_executable(test_diag
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_diag_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    chess_add_header_deps(test_diag)
    target_include_directories(test_diag PRIVATE ${SRC_DIR})
    target_link_libraries(test_diag PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(test_diag PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_diag PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    add_test(NAME test_diag COMMAND test_diag)
    set_tests_properties(test_diag PROPERTIES TIMEOUT 300)

    # ------------------------------------------- 诊断仪表盘 (自对弈 + 自动战术题库)
    # **不进 ctest**: 跑真实自对弈与战术题, 依赖随机开局, 分钟级。
    #   cmake --build <build> --target bench_diag
    #   <build>/bench_diag --games=2 --plies=40 --sims=40 --hidden=16 --expert=16 --tactics=20
    #   <build>/bench_diag --sims=400 --tactics=60 --load=D:\...\weights\ppo_bc_d4 --csv=diag
    add_executable(bench_diag
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_diag_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    chess_add_header_deps(bench_diag)
    target_include_directories(bench_diag PRIVATE ${SRC_DIR})
    target_link_libraries(bench_diag PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_diag PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_diag PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)

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
        #
        # 2026-09 更新: test_dqnmcts **现在有断言**了 (第 [6] 节 16 条: 界面"模型自检"
        # 面板所依赖的那些量 —— 报告非空/含关键字段、只读 (调用前后棋盘哈希不变)、
        # 可重复、终局通道计数一局恰好一条), 并且 main() 按失败数返回非零退出码。
        # 本机实测 339.7 s (Release, MSVC 2022, Qt 6.8.0) —— 仍然明显超过随手跑的成本,
        # 所以它**依旧不注册进 ctest**; 但手动跑时 `echo %ERRORLEVEL%` 是有意义的:
        # 0 = 全过。若要注册, 请先把上面那张耗时表更新到实测值。
        # 另: 表示能力探针是独立目标 probe_dqnmcts_aliasing (秒级, 见文件末尾),
        # 需要"不改代码就能复现结构结论"时用它, 不必跑这个 340 s 的基准。
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
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/dqnabagent.cpp
        # [2.9] 节要直接构造 CurveChart, 钉住"换一批线"的语义 (clearData 只清点,
        # removeAllSeries 才清线) —— 曲线控件只依赖 Qt Widgets, 不需要主窗口。
        ${SRC_DIR}/metricsview.cpp
        ${SRC_DIR}/metricsview.h)
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
    # TIMEOUT 600 -> 900 (2026-09): test_match 现在还会逐个 agent 跑一轮**后台训练**
    # ([2.12] EVAB + [2.13] SAC+AZ / SAC+AZ-MoE / DQN+AB), 而其中两个 agent 的一轮
    # 本身是分钟级 (稀疏 MoE 的前向 + ~276 MB 权重的临时文件往返)。实测本机 250 s,
    # 原来的 600 s 在慢一点的机器上会擦边。
    set_tests_properties(test_match PROPERTIES TIMEOUT 900
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_qt_bin_dir}")

    # ------------------------------------------- "对弈 x 后台训练"并发崩溃的最小复现
    # 2026-09 用户报的 0xC0000374 (堆损坏): 只在对弈的某一方 == 后台训练的目标时出现,
    # 而且必须"对弈在另一个线程 + 面板在刷自检"才稳定复现 (见该文件的顶部注释)。
    # **故意不进 ctest**: 它压的是时序相关的堆损坏, 属于"改完并发代码后手工压一下"的
    # 工具; 计时与调度变了它可能一时不复现, 那就不要拿它当门禁。
    add_executable(repro_concurrency
        ${CMAKE_CURRENT_SOURCE_DIR}/test/repro_concurrency_main.cpp
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
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/dqnabagent.cpp)
    target_include_directories(repro_concurrency PRIVATE ${SRC_DIR})
    target_link_libraries(repro_concurrency PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(repro_concurrency PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(repro_concurrency PROPERTIES
        AUTOMOC ON
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(repro_concurrency)

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

    # ------------------------------------------- ScaledConcat (结构不变量)
    # 这个层在这之前**一个测试都没有** (它只活在 dqn.cpp 的一棵 #elif 0 里)。
    # 2026-09 重写之后, 三类东西必须钉住:
    #   * 旧实现的结构性缺陷真的修掉了 —— 旧门控 softmax(sigmoid(·)) 的
    #     gate_max/gate_min 被 e^1 卡死 (实测有效路数 58.6~61.3/64), 新门控无界;
    #   * 门控与特征真的解耦 —— 改专家权重门控必须逐位不变, 反之亦然;
    #   * 反向 vs 中心差分, 含**输入梯度的三条累加通路** (W2ᵀdz / Wgᵀdlogit / 专家)
    #     以及专家模板参数 (MlpExpert / TransformerBlock / Layer<Fn>)。
    add_executable(test_scaledconcat
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_scaledconcat_main.cpp)
    target_include_directories(test_scaledconcat PRIVATE ${SRC_DIR})
    target_link_libraries(test_scaledconcat PRIVATE RL_CORE)
    target_compile_options(test_scaledconcat PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_scaledconcat PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_scaledconcat)
    add_test(NAME test_scaledconcat COMMAND test_scaledconcat)
    set_tests_properties(test_scaledconcat PROPERTIES TIMEOUT 600)

    # ------------------------------------------- RL::SAC (库里的离散 SAC)
    # 这个类在这之前**没有任何调用方**, 也从没被跑过。2026-09 把 PPO 的训练侧优化
    # 方法 (P3 梯度累积 / P4 回放池多 epoch / MoE 批统计与负载均衡辅助损失 /
    # 批平均损失 / R2 只算合法列) 搬过来时顺手撞出两处静默缺陷 (critic 的
    # [state;π] 拼接被 MM 的形状契约静默截断; learn() 的 learningRate 参数没接线),
    # 所以必须有测试盯着。它只依赖 RL_CORE (不碰棋盘/Qt), 因此也进 ctest。
    # 盯三件事:
    #   * R2 的口径: 合法集 Σπ ≡ 1、非法列 π 恰好为 0、非法列的**头权重梯度恰好为 0**
    #   * critic 真的在学 (固定经验, TD 目标恰好 0.5, 断言 Q 朝它走)
    #   * P3/P4 的机制: 多 epoch 的梯度成比例、优化器只调一次、
    #     resetMoeBatchStats 真的划出了批边界 (带正对照)
    add_executable(test_sac
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sac_main.cpp)
    target_include_directories(test_sac PRIVATE ${SRC_DIR})
    target_link_libraries(test_sac PRIVATE RL_CORE)
    target_compile_options(test_sac PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_sac PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_sac)
    add_test(NAME test_sac COMMAND test_sac)
    set_tests_properties(test_sac PROPERTIES TIMEOUT 600)

    # ------------------------------------------- DQN+AB (AB 当 DQN 的 planning head)
    # 新 agent: 稀疏 MoE + TransformerBlock 专家骨干 + Dueling 双头 (V/A), AB 用 Q 排序、
    # 用 V 当叶子, 并把"展开若干层后的值"当 TD 目标 (planning head)。
    # 测试盯 [3] Dueling 双头的解析梯度 vs 有限差分、[4] 一步杀命中率 (AB vs 纯 Q-argmax)、
    # [1][2] 规范视角的镜像对称与动作双射、[5] 固定目标收敛、[6][7] 自对弈/路由/代价/棋盘复原。
    # 只依赖 RL_CORE + 棋盘 (不碰 Qt)。
    add_executable(test_dqnab
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dqnab_main.cpp
        ${SRC_DIR}/dqnabagent.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp)
    target_include_directories(test_dqnab PRIVATE ${SRC_DIR})
    target_link_libraries(test_dqnab PRIVATE RL_CORE)
    target_compile_options(test_dqnab PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_dqnab PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_dqnab)
    add_test(NAME test_dqnab COMMAND test_dqnab)
    set_tests_properties(test_dqnab PROPERTIES TIMEOUT 900)

    # ------------------------------------------- DQN+AB vs AB 静默对弈验证
    # **不进 ctest** (与 bench_sacaz_vs_ab / bench_ppo_vs_ab 同一理由): 跑真实对局,
    # 依赖随机开局, TB 骨干建网要 ~1 s、每步要 1~2 s。
    # 手动跑:
    #   cmake --build <build> --target bench_dqnab_vs_ab
    #   <build>/bench_dqnab_vs_ab --games=4 --budget=512 --quiet
    #   对照(关掉搜索, 只按 Q 取最大): <build>/bench_dqnab_vs_ab --pure-q
    add_executable(bench_dqnab_vs_ab
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_dqnab_vs_ab_main.cpp
        ${SRC_DIR}/dqnabagent.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/abagent.cpp)
    target_include_directories(bench_dqnab_vs_ab PRIVATE ${SRC_DIR})
    target_link_libraries(bench_dqnab_vs_ab PRIVATE RL_CORE)
    target_compile_options(bench_dqnab_vs_ab PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_dqnab_vs_ab PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_dqnab_vs_ab)

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
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp)
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

    # ------------------------------------------- PPO+MCTS 同权重、只改模拟次数的头对头
    # **不进 ctest**: 同 bench_ppo_vs_ab —— 跑真实对局、依赖随机开局。
    # 它回答的是 "把模拟次数从 80 提到 800, 同一个策略会不会下得更好": 两方权重逐位
    # 相同、温度 0、开局由局号决定, 唯一差别是 --sims-a / --sims-b。手动跑:
    #   cmake --build <build> --target bench_ppo_sims
    #   <build>/bench_ppo_sims --games=20 --load=weights/<prefix> --sims-a=80 --sims-b=800
    add_executable(bench_ppo_sims
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_ppo_sims_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/abagent.cpp)
    target_include_directories(bench_ppo_sims PRIVATE ${SRC_DIR})
    target_link_libraries(bench_ppo_sims PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_ppo_sims PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_ppo_sims PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_ppo_sims)

    # ------------------------------------------- 纯 MCTS (随机回放) 的模拟次数标度探针
    # **不进 ctest**: 同 bench_ppo_sims —— 依赖随机回放与随机开局。
    #   cmake --build <build> --target bench_mcts_sims
    #   <build>/bench_mcts_sims --positions=20 --sims-a=800 --sims-b=1600
    add_executable(bench_mcts_sims
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_mcts_sims_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/mcts.cpp)
    target_include_directories(bench_mcts_sims PRIVATE ${SRC_DIR})
    target_link_libraries(bench_mcts_sims PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_mcts_sims PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_mcts_sims PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_mcts_sims)

    # ------------------------------------------- SAC+AZ(TB 专家) vs AB 静默对弈验证
    # GUI 里的 "SAC+AZ-MoE" 那一档 (`AGENT_SACAZ_MOE`) 的无界面对应物: 骨干 =
    # SparseMoeTb, 16 次模拟/步, AB 深度 4 —— 与本文件里 SACAZ_MOE_* 常量一致。
    #
    # 它验的是**机制**并把结果放进退出码: 逐手校验走法合法性、不许有无效走法、
    # 每局必须在手数上限内结束、对局后策略与 Q 必须全有限。比分**不影响退出码**
    # (默认是随机初始化权重, 输给固定深度的 alpha-beta 是预期内的, 不是缺陷)。
    # --quiet 只打一行结论 (静默验证)。
    #
    # **不进 ctest**: 与 bench_ppo_vs_ab / bench_moe 同一理由 —— 它跑真实对局,
    # 依赖随机开局, 一个 TB 专家 agent 还要 ~1.6 GB 内存与秒级的建网时间。
    # 手动跑:
    #   cmake --build <build> --target bench_sacaz_vs_ab
    #   <build>/bench_sacaz_vs_ab --games=4 --sims=16 --quiet
    add_executable(bench_sacaz_vs_ab
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_sacaz_vs_ab_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/abagent.cpp)
    target_include_directories(bench_sacaz_vs_ab PRIVATE ${SRC_DIR})
    target_link_libraries(bench_sacaz_vs_ab PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_sacaz_vs_ab PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_sacaz_vs_ab PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_sacaz_vs_ab)

    # ------------------------------------------- PPO 稀疏输出口径的核对与代价拆解
    # 用户要求: PPO 保持 8100 双射, 但输出走"只对合法动作动态打分"的稀疏口径。
    # 这条路径 PPO 已有 (`sparsePolicyHead` -> `actionMasked` -> `forwardTrunk` +
    # `sparseLogits` + 合法集 softmax), 这个工具把它**钉住**并拆开代价:
    #   1) 稀疏 vs 全量在合法集上逐元素一致 (证明没有静默回退);
    #   2) 头权重的读取字节数 (确定性事实, 不随机器抖动);
    #   3) 骨干 / 稀疏头 / 全量头 / 值头各自的单价与在一次决策里的占比。
    add_executable(bench_ppo_sparse
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_ppo_sparse_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    target_include_directories(bench_ppo_sparse PRIVATE ${SRC_DIR})
    target_link_libraries(bench_ppo_sparse PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_ppo_sparse PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_ppo_sparse PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_ppo_sparse)

    # ------------------------------------------- 跨版本对照的最小工具
    # 与 bench_agent_arena 的区别: 只用**两个 commit 都有**的最小接口, 于是同一份源码
    # 能编到改前那棵树 (worktree) 上, 用来回答"现在的 SAC 是不是比改前弱"。
    # 协议与 arena 一致: 交换先后手 / 同一随机开局 / 同一模拟预算 / 逐手合法性校验 /
    # 得分率 + Wilson 区间。
    add_executable(bench_sac_mcts_min
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_sac_mcts_min_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp)
    target_include_directories(bench_sac_mcts_min PRIVATE ${SRC_DIR})
    target_link_libraries(bench_sac_mcts_min PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_sac_mcts_min PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_sac_mcts_min PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_sac_mcts_min)

    # ---- bench_sac_learn: "SAC 边下边学" 的受控复现 + 奖励塑形 A/B (2026-09) ----
    # 与上面那个的区别: 这个工具**每手都调 exploreAndTrain** (界面 preTrainThenDecide 的
    # 那一步) 并统计胜负和的成因/奖励累计/损失曲线/critic 尺度 —— 也就是用户报的
    # "奖励两倍但 300 手和棋" 那组现象。协议逐条对齐 ChessBoard::playMatchGame。
    add_executable(bench_sac_learn
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_sac_learn_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp)
    target_include_directories(bench_sac_learn PRIVATE ${SRC_DIR})
    target_link_libraries(bench_sac_learn PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_sac_learn PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_sac_learn PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_sac_learn)

    # 同一份源码 + `SACAZ_ALIGNED_REPR=1` -> 对齐表示 (19 平面 / 8100 双射) 的副本。
    # 表示是**编译期**开关 (STATE_DIM / ACTION_DIM 决定网络形状), 所以要两个二进制才
    # 能在同一次会话里对跑。默认那个是改前表示 (实测更强, 见 sacazagent.h 的说明)。
    add_executable(bench_sac_mcts_min_aligned
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_sac_mcts_min_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp)
    target_compile_definitions(bench_sac_mcts_min_aligned PRIVATE SACAZ_ALIGNED_REPR=1)
    target_include_directories(bench_sac_mcts_min_aligned PRIVATE ${SRC_DIR})
    target_link_libraries(bench_sac_mcts_min_aligned PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_sac_mcts_min_aligned PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_sac_mcts_min_aligned PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_sac_mcts_min_aligned)

    # ------------------------------------------- 策略头梯度不变量探针
    # 量"非法动作的头部权重是否逐位不变" —— 掩码 softmax 的契约 (dz 在非法列恒为 0)。
    # test_sacaz [11] 钉的是同一条; 这个探针用来**定位**它为什么坏 (熵项/裁剪/动作空间)。
    add_executable(probe_sac_head_grad
        ${CMAKE_CURRENT_SOURCE_DIR}/test/probe_sac_head_grad_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    target_include_directories(probe_sac_head_grad PRIVATE ${SRC_DIR})
    target_link_libraries(probe_sac_head_grad PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(probe_sac_head_grad PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(probe_sac_head_grad PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(probe_sac_head_grad)

    # ------------------------------------------- 受控对局竞技场 (三方对照)
    # 为什么需要它: "哪个算法更强"是**三方**问题 (两个学习 agent 互打 + 各自对同一个
    # 无学习基线), 而 bench_sacaz_vs_ab / bench_ppo_vs_ab 各自只测"一个 agent 对 AB",
    # 两次运行之间的随机开局/种子/模拟预算无法保证一致。这里把三方装进一个二进制,
    # 共用同一套协议 (交换先后手 + 同一随机开局 + 同一模拟预算 + 逐手合法性校验),
    # 并输出得分率的 Wilson 95% 区间 —— 小样本下"5 胜 0 负"必须配区间才读得对。
    # **不进 ctest**: 跑真实对局/自对弈, 耗时以分钟计。
    #   cmake --build <build> --target bench_agent_arena
    #   <build>/bench_agent_arena --mode=train --a=sac --b=ppo --games=200 --sims=64 \
    #                             --save-prefix=weights/arena
    #   <build>/bench_agent_arena --mode=match --a=sac --b=ppo --games=60 --sims=64 \
    #                             --load-a=weights/arena_sac --load-b=weights/arena_ppo
    add_executable(bench_agent_arena
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_agent_arena_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/abagent.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    target_include_directories(bench_agent_arena PRIVATE ${SRC_DIR})
    target_link_libraries(bench_agent_arena PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_agent_arena PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    # PDB: 生成符号以便把崩溃地址翻译成函数名 (`cdb -c "sxe av; g; kb"`)。
    # 这个目标存在的意义就是**定位缺陷**, 没有符号时一次访问违例只能看到偏移量
    # (2026-09 实际吃过这个亏)。只影响产物大小, 不影响任何行为。
    target_link_options(bench_agent_arena PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/DEBUG:FULL>)
    set_target_properties(bench_agent_arena PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_agent_arena)

    # ------------------------------------------- 同源副本 (数值对照用)
    # 与 bench_agent_arena 是**同一份源码**, 只换产物名。
    # 为什么需要它: 长时训练/对局进程会一直占住 bench_agent_arena.exe (Windows 不允许
    # 覆盖正在运行的可执行文件), 于是"想改一个旋钮再跑一轮对照"会被链接失败挡住。
    # 有了这个副本名, 迭代时链接 `bench_agent_arena_ab`, 正在进行的长跑不受影响。
    # **临时目标**: 只服务于 2026-09 的 SAC/PPO 对照实验, 实验归档后可删。
    add_executable(bench_agent_arena_ab
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_agent_arena_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/mcts.cpp
        ${SRC_DIR}/abagent.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    target_include_directories(bench_agent_arena_ab PRIVATE ${SRC_DIR})
    target_link_libraries(bench_agent_arena_ab PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_agent_arena_ab PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_agent_arena_ab PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_agent_arena_ab)

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

    # ------------------------------------------- 奖励/学习目标 诊断 (Phase 0)
    # **不进 ctest**: 它跑真实自对弈并打印统计量, 数字本身会随奖励公式一起变,
    # 是"改动前后对比"的尺子, 不是"通过/失败"的断言。手动跑:
    #   cmake --build <build> --target test_reward_diag
    #   <build>/test_reward_diag --games=1 --sims=8 --moves=20
    # 它量五件事: ①跨 agent 的 computeReward 一致性 ②奖励尺度表与"终局是否压倒材质"
    # 这条不变量 ③一局随机棋的材质/终局奖励比 ④PPO 的价值目标分布 (有多少样本真的有
    # 信号) ⑤"最后一条样本"与"整条轨迹批平均"两种损失口径的差。
    add_executable(test_reward_diag
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_reward_diag_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/pgagent.cpp
        ${SRC_DIR}/dqnagent.cpp
        ${SRC_DIR}/dqnmcts_agent.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/evagent.cpp)
    target_include_directories(test_reward_diag PRIVATE ${SRC_DIR})
    target_link_libraries(test_reward_diag PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(test_reward_diag PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_reward_diag PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_reward_diag)

    # ------------------------------------------- 选点一致率探针 (灵敏的代理指标)
    # **不进 ctest**: 它跑 200 个局面的双方决策, 数字随权重变化 —— 是"改动前后对比"的
    # 尺子, 不是通过/失败断言。手动跑:
    #   cmake --build <build> --target bench_policy_agreement
    #   <build>/bench_policy_agreement --positions=200 --sims=80 --depth=4
    #   <build>/bench_policy_agreement --positions=200 --load=weights/verify_on
    # 为什么需要它: 20 局等时间对局的噪声带实测是 ±5%, 测不出小改动; 而 200 个局面的
    # 选点一致率每个局面一个二值样本, 标准误约 3.5%, 且局面集由种子固定、两个权重文件
    # 跑的是同一批局面 (配对比较)。
    add_executable(bench_policy_agreement
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_policy_agreement_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/abagent.cpp)
    target_include_directories(bench_policy_agreement PRIVATE ${SRC_DIR})
    target_link_libraries(bench_policy_agreement PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_policy_agreement PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_policy_agreement PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_policy_agreement)

    # ------------------------------------------- 价值头蒸馏 (Step 2)
    # **不进 ctest**: 它跑 AB 搜索打分 + 训练 critic, 分钟级。手动跑:
    #   cmake --build <build> --target bench_ppo_distill
    #   <build>/bench_ppo_distill --positions=1500 --depth=3 --save=weights/distill
    # 用 ABAgent 的**根搜索分**(含三四层战术) 当监督目标, 单独预训练 critic ——
    # 不碰奖励、不依赖外部引擎 (全仓无 FEN 接口, 接 Pikafish 还得先写局面交换桥)。
    add_executable(bench_ppo_distill
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_ppo_distill_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/abagent.cpp)
    target_include_directories(bench_ppo_distill PRIVATE ${SRC_DIR})
    target_link_libraries(bench_ppo_distill PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_ppo_distill PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_ppo_distill PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_ppo_distill)

    # ------------------------------------------- P0.1: 无界面常驻训练器
    # **不进 ctest**: 跑真实自对弈训练, 分钟级, 且打印的是"生态读数"而不是通过/失败。
    #
    # 为什么需要它: GUI 的训练线程是**每局** 3 趟整模型磁盘往返 (seed 写盘 -> clone
    # 读入 -> clone 写回 -> 主 agent 读入; actor+critic 一对约 555 MB), 换 3.2 分钟
    # 的自对弈。那个架构是为"界面上看得见进度"服务的, 不是吞吐基线。本程序对一个
    # 常驻 agent 反复调 trainSelfPlay, 每局零磁盘 I/O。
    # 同时是 P0.2 (和棋原因分桶) / P0.4 (一次性 CSV 报告) / P1.1 (PBRS A/B) 的落点。
    #   cmake --build <build> --target train_ppo
    #   <build>/train_ppo --games=20 --sims=400 --moves=60 --opening=8 --csv=run1
    #   <build>/train_ppo --games=20 --no-shaping --csv=shaping_off   # P1.1 A/B
    add_executable(train_ppo
        ${CMAKE_CURRENT_SOURCE_DIR}/test/train_ppo_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    target_include_directories(train_ppo PRIVATE ${SRC_DIR})
    target_link_libraries(train_ppo PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(train_ppo PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(train_ppo PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(train_ppo)

    # ------------------------------------------- P0.3: 固定锚点评测
    # **不进 ctest**: 跑真实对局, 分钟级起。它给出"棋力有没有涨"的可证伪尺子:
    # 固定开局集 (确定性生成 + FNV 指纹) + 换先手成对计分 + 得分率/Elo 的 95% 区间。
    #   cmake --build <build> --target bench_anchor
    #   <build>/bench_anchor --a=weights/run_10k --b=weights/run_5k --openings=20 --plies=8
    #   <build>/bench_anchor --a=weights/run_10k --ab-depth=4 --budget=100 --openings=20
    add_executable(bench_anchor
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_anchor_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/ppomcts_agent.cpp
        ${SRC_DIR}/abagent.cpp)
    target_include_directories(bench_anchor PRIVATE ${SRC_DIR})
    target_link_libraries(bench_anchor PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_anchor PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_anchor PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(bench_anchor)

    # ------------------------------- 对弈模式 (P0-a) 的成本/语义读数 (不进 ctest)
    # "评估模式到底快多少"决定它能不能当日常工具用: 模式同时关掉两条学习路径
    # (rollout + SAC 的 learnFromSearch), 省下的时间与 agent 组合有关
    # (SAC+AZ 每手一次 learnBatch 约 2.3 s)。回归断言在 test_match [2.7d], 这里只是读数。
    #   <build>/bench_match_modes --games=2 --plies=40 --sims=32 --a=ppo --b=sac
    add_executable(bench_match_modes
        ${CMAKE_CURRENT_SOURCE_DIR}/test/bench_match_modes_main.cpp
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
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/dqnabagent.cpp)
    target_include_directories(bench_match_modes PRIVATE ${SRC_DIR})
    target_link_libraries(bench_match_modes PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(bench_match_modes PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_match_modes PROPERTIES
        AUTOMOC ON
        AUTOUIC OFF)
    chess_add_header_deps(bench_match_modes)

    # ------------------- PG "lastLoss 变非有限值且不复位" 的定位探针 (不进 ctest)
    # test_match [2.7b] 曾经时真时假 (一次绿一次红), 根因就是 dpg.lastLoss 会变成
    # NaN/Inf 而曲线控件丢弃非有限值 ⇒ 损失曲线静默停止上报。这个探针把"哪一手开始坏"
    # 打出来。修好后应当打印 "0/N 局出现过非有限 lastLoss"。
    add_executable(probe_pg_loss_nan
        ${CMAKE_CURRENT_SOURCE_DIR}/test/probe_pg_loss_nan_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/pgagent.cpp)
    target_include_directories(probe_pg_loss_nan PRIVATE ${SRC_DIR})
    target_link_libraries(probe_pg_loss_nan PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(probe_pg_loss_nan PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(probe_pg_loss_nan PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF)
    chess_add_header_deps(probe_pg_loss_nan)

    # ------------------------------------------- SAC + MCTS + AlphaZero agent
    # 盯四件事: 掩码 softmax 的雅可比(有限差分)、搜索给出的走法合法性、
    # 软价值里 α 熵项的形式、critic 是否真的把 Q 拉向目标。
    # 对齐表示那一支的编译+自洽验证: 同一份源码 + SACAZ_ALIGNED_REPR=1。
    # 两种表示都必须在断言下自洽 (STATE_DIM/ACTION_DIM/上下文槽位置都不同),
    # 否则"开关"就成了"两个会漂移的实现"。
    add_executable(test_sacaz_aligned
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sacaz_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp
        ${SRC_DIR}/ppomcts_agent.cpp)
    target_compile_definitions(test_sacaz_aligned PRIVATE SACAZ_ALIGNED_REPR=1)
    target_include_directories(test_sacaz_aligned PRIVATE ${SRC_DIR})
    target_link_libraries(test_sacaz_aligned PRIVATE
        RL_CORE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql)
    target_compile_options(test_sacaz_aligned PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(test_sacaz_aligned PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(test_sacaz_aligned)
    add_executable(test_sacaz
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sacaz_main.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/gamedb.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp
        ${SRC_DIR}/sacazagent.cpp
        ${SRC_DIR}/sacazlegacyagent.cpp)
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

    # ------------------------------------------- DQN+MCTS 表示能力探针 (2026-09)
    # 为什么需要: DQNMCTSAgent 的训练数字 (50 胜 0 负 50 和 / 损失 22) 无法回答
    # "它学没学会下棋"。这个探针量的是**表示层**的两条硬约束, 与训练量无关:
    #   [1] 动作别名 —— STEP_TO_ACTION_IDX 把 (棋子 id, 目标格) 哈希进 128 个槽位,
    #       而真实走法空间是 16x90 (静态) / 90x90 (全盘): 多少走法被迫共用一个下标。
    #   [2] 状态不可分 —— 90 维只存"每格有什么子", 于是走子方 / 重复进度 /
    #       自然限着逐字节不可分; 且 evaluateLeaf 用 isGameOver() 判终局 (只认将帅
    #       是否还在场), 判和与将杀都不是终局。
    # **不进 ctest**: 它是"值不值得继续训这个 agent"的前置闸门, 不是回归测试。
    #   cmake --build <build> --target probe_dqnmcts_aliasing
    #   <build>/probe_dqnmcts_aliasing --positions=32 --plies=20
    add_executable(probe_dqnmcts_aliasing
        ${CMAKE_CURRENT_SOURCE_DIR}/test/probe_dqnmcts_aliasing_main.cpp
        ${SRC_DIR}/dqnmcts_agent.cpp
        ${SRC_DIR}/chess.cpp
        ${SRC_DIR}/pos.cpp
        ${SRC_DIR}/stone.cpp)
    target_include_directories(probe_dqnmcts_aliasing PRIVATE ${SRC_DIR})
    target_link_libraries(probe_dqnmcts_aliasing PRIVATE RL_CORE)
    target_compile_options(probe_dqnmcts_aliasing PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(probe_dqnmcts_aliasing PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF)
    chess_add_header_deps(probe_dqnmcts_aliasing)

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

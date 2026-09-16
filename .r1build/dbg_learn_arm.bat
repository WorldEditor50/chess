@echo off
REM 编译并运行一个临时诊断程序 (只依赖 RL_CORE + 源文件, 不进 CMake)
REM 用法: dbg_learn_arm.bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0..
cl /nologo /std:c++17 /EHsc /O2 /MD /utf-8 /bigobj /arch:AVX2 ^
   /I src /I src\rl /I . ^
   /Fe:.r1build\dbg_learn_arm.exe /Fo:.r1build\ ^
   .r1build\dbg_learn_arm.cpp src\dqnabagent.cpp src\chess.cpp src\pos.cpp src\stone.cpp ^
   /link build\Desktop_Qt_6_8_0_MSVC2022_64bit-Release\RL_CORE.lib
if errorlevel 1 exit /b 1
.r1build\dbg_learn_arm.exe

@echo off
rem P0 verification: compile-only check of the two new test programs, to surface warnings.
rem (The CMake build succeeded; this is for warning visibility without the piped build log.)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\home\lab\chess\.r1build
cl /nologo /c /std:c++17 /O2 /DNDEBUG /EHsc /GR /bigobj /arch:AVX2 /utf-8 /I..\src /I..\src\rl ..\test\train_ppo_main.cpp
echo TRAIN_EXITCODE=%ERRORLEVEL%
cl /nologo /c /std:c++17 /O2 /DNDEBUG /EHsc /GR /bigobj /arch:AVX2 /utf-8 /I..\src /I..\src\rl ..\test\bench_anchor_main.cpp
echo ANCHOR_EXITCODE=%ERRORLEVEL%

@echo off
rem Full build check: chess.h / diag.h are cross-cutting headers, so every target that
rem includes them must be recompiled. This builds ALL targets (no --target filter).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\home\lab\chess
cmake --build build\Desktop_Qt_6_8_0_MSVC2022_64bit-Release
echo FULL_BUILD_EXITCODE=%ERRORLEVEL%

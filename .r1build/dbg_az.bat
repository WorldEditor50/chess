@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\home\lab\chess\.r1build
cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /GR /bigobj /arch:AVX2 /utf-8 /I..\src /I..\src\rl dbg_az.cpp ..\src\sacazagent.cpp ..\src\chess.cpp ..\src\pos.cpp ..\src\stone.cpp ..\src\rl\util.cpp ..\build\Desktop_Qt_6_8_0_MSVC2022_64bit-Release\RL_CORE.lib /Fe:dbg_az.exe
echo EXITCODE=%ERRORLEVEL%

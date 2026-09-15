@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\home\lab\chess\.r1build
cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /GR /bigobj /arch:AVX2 /utf-8 /I..\src /I..\src\rl dbg_gate.cpp ..\src\rl\util.cpp /Fe:dbg_gate.exe
echo EXITCODE=%ERRORLEVEL%

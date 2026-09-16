@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\home\lab\chess\.r1build
cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /GR /bigobj /arch:AVX2 /utf-8 /I..\src /I..\src\rl bench_ppo_expert.cpp ..\src\rl\util.cpp /Fe:bench_ppo_expert.exe
echo EXITCODE=%ERRORLEVEL%

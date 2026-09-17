@echo off
rem P0 verification: compile the rules regression test directly with cl (no CMake needed).
rem test_rules only needs chess.cpp / pos.cpp / stone.cpp / rl/util.cpp.
rem NOTE: keep this file ASCII-only. A cmd.exe batch file with non-ASCII comments gets
rem mis-parsed under the OEM code page ("'xxx' is not recognized as a command").
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\home\lab\chess\.r1build
cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /GR /bigobj /arch:AVX2 /utf-8 ^
   /I..\src /I..\src\rl ^
   ..\test\test_rules_main.cpp ..\src\chess.cpp ..\src\pos.cpp ..\src\stone.cpp ..\src\rl\util.cpp ^
   /Fe:test_rules_p0.exe
echo EXITCODE=%ERRORLEVEL%

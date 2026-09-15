@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\home\lab\chess
cmake --build build\Desktop_Qt_6_8_0_MSVC2022_64bit-Release --target %*
echo EXITCODE=%ERRORLEVEL%

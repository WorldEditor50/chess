call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d e:\home\code\chess
C:\Qt\Tools\CMake_64\bin\cmake.exe -S . -B build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release -G Ninja -DCMAKE_PREFIX_PATH=C:/Qt/6.9.2/msvc2022_64 -DCMAKE_BUILD_TYPE=Release 2>&1
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
C:\Qt\Tools\CMake_64\bin\cmake.exe --build build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release 2>&1
echo Build exit code: %ERRORLEVEL%

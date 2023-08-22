taskkill /IM nw.exe
cd vraddon\
call node-gyp build
cd ..
timeout 1
copy /y vraddon\build\Release\vrviewer.node node_modules\vrviewer\build\Release
start /B nw\nw.exe .
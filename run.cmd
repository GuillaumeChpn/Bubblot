@echo off
taskkill /IM nw.exe
::taskkill /IM bubblot.exe
taskkill /IM VirtualHub.exe
start /MIN virtualhub\VirtualHub.exe 
start /B nw-release\nw.exe .
::start /B bubblot.exe
cd bubblot_display
start /B bubblot_display.exe
cd ..
timeout 5
wmic process where name="bubblot.exe" CALL setpriority "high priority"
::wmic process where name="bubblot_display\bubblot_display.exe" CALL setpriority "high priority"
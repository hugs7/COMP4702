@echo off
color 0a
mode 1000
title Matrix

:matrix
setlocal enabledelayedexpansion

set "chars=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%%^&*()_+[]\{}|;:'"",.<>?/"
set "width=100"

for /l %%a in (1,1,%width%) do (
    set /a "rand=!random! %% 81 + 2"
    set "line="
    for /l %%b in (1,1,!rand!) do (
        set /a "randchar=!random! %% 95"
        set "line=!line!!chars:~!randchar!,1!"
    )
    echo !line!
)

timeout /t 0.05 >nul

goto :matrix

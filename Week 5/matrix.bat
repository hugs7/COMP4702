@echo off

mode 1000
title Matrix

:matrix
setlocal enabledelayedexpansion

set "chars=日月金木水火土竹戈大中一弓人心手口尸廿山女田難卜Ｚ!@#$&*()_+[]\{}|;:'"",.<>?/"
set "width=100"

for /l %%a in (1,1,%width%) do (
    set /a "rand=!random! %% 81 + 2"
    set "line="
    for /l %%b in (1,1,!rand!) do (
        set /a "randchar=!random! %% 95"
        set "line=!line!!chars:~!!,1!"
    )
    echo !line!
)

timeout /t 0.05 >nul

goto :matrix

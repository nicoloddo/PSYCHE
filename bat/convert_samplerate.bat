@ECHO OFF
rem cd %~dp0
cd %cd%
IF NOT EXIST converted MKDIR converted
FOR %%A IN (%*) DO MKDIR "%%~dpAconverted"  & sox -V4 -S %%A "%%~dpAconverted\%%~nA" rate -v 22050
::FOR %%A IN (%*) DO MKDIR "%%~dpAconverted"  & sox -V4 -S %%A "%%~dpAconverted\%%~nA.converted%%~xA.wav" rate -v 22050 This was the old command but with .converted.wav.wav as ending of files
PAUSE
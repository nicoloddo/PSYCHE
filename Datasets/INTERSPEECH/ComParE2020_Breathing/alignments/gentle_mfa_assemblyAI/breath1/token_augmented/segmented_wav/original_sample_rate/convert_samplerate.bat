@ECHO OFF
IF NOT EXIST converted MKDIR converted
FOR %%A IN (*.wav) DO sox -V4 -S "%%A" "converted\%%~nA.wav" rate -v 22050
PAUSE
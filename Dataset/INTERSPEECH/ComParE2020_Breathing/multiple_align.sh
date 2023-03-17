#!/bin/bash

for wav_filename in ./wav/*.wav
do
	python align.py "${wav_filename}" "./transcriptions_ibm/${wav_filename::-4}_transcript.txt" > "./aligned_ibm/${wav_filename::-4}_aligned.txt"
done
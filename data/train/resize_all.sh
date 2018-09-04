#!/bin/bash

for f in $(ls *.mp4) 
do
    time ffmpeg -i $f -vf scale=240:120 -strict -2 -r 1 resized_$f
done

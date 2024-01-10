#!/bin/bash

for f in $1*.mov
do
    # echo "$f"
    vid_name=$f
    num_frames=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 ${vid_name})
    num_frames=$(echo $num_frames | tr "," "\n")

    ext_num=$2
    frame_array=()
    for i in $(seq 0 `expr $ext_num - 1`)
    do
        frame=`expr 1 + $i \* $num_frames \/ $ext_num`
        if [ $i -eq `expr $ext_num - 1` ]
        then
            frame_array+="eq(n\,$frame)"
        else
            frame_array+="eq(n\,$frame)+"
        fi
    done

    filename=$(basename "$vid_name")
    filename=${filename%.mov}
    echo $filename
    ffmpeg -i $vid_name -vf select=$frame_array -vsync 0 $3${filename}-%02d.png
done


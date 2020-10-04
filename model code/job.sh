#!/bin/bash

for c in $1
do
    for f in 50 100 200 300
    do
        for a in 0.175 0.25 0.5
        do
            echo "Cycles = $c, Frequency = $f, Amplitude = $a" >>c${c}_f${f}_a${a}.txt
            for i in 1 2 3 4 5 6 7 8 9 10
            do
                echo "" >>c${c}_f${f}_a${a}.txt
                python model.py $c $f $a >>c${c}_f${f}_a${a}.txt
                echo "" >>c${c}_f${f}_a${a}.txt
                echo "Run $i of 10 complete" >>c${c}_f${f}_a${a}.txt
                echo "" >>c${c}_f${f}_a${a}.txt
            done
        done
    done
done


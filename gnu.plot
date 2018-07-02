#!/bin/bash

echo '

set term x11
set out

set style data linespoints

plot   "out.now" using 2:6 title "train loss" with lines lw 8
replot "out.now" using 2:7 title "test loss" with lines lw 8
replot "out.now" using 2:9 title "train error" with points pt 7 ps 3
replot "out.now" using 2:10 title "test error" with points pt 5 ps 3


' |gnuplot -persist





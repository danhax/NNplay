#!/bin/bash

echo '

set term x11
set out

set logscale y

set style data linespoints

set ytics 2
set yrange [*:0.1]

plot "out.now" using 2:9 every 10 title "train error" with points pt 7 ps 3
replot "out.now" using 2:10 every 10 title "test error" with points pt 5 ps 3
replot   "out.now" using 2:6 title "train loss" with lines lw 8
replot "out.now" using 2:7 title "test loss" with lines lw 8


' |gnuplot -persist





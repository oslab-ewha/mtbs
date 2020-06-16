#!/bin/bash

function usage() {
    cat <<EOF
Usage: run.sh [options] <benchmark code:bench args only>...
 -c <# of tests>, default: 1
 -s <schedules>: hw,sd
 -n <tbs per SM>, default: 1,2,3,4
 -S <# of SM>, default: 1
 -m <start>,<end>: multirun mode
 -t <threads per TB>, default: 32,64,128,256,512,1024
 -T <#TB of start>,<#TB of end>: TB incresing mode
 -x: benchmark direct mode(benchmark TB, threads should be specified) 
 -H: display header
EOF
}

MTBS=${MTBS:-src/mtbs/mtbs}
MTBS_ARGS=${MTBS_ARGS}

n_tests=1
n_sms=1
schedules="hw,sd"
n_tbs_per_sm="1,2,3,4,5,6,7,8"
n_ths_per_tb="32,64,128,256,512,1024"
multirun_range=
tb_range=
while getopts "c:s:n:S:m:t:T:xH" arg
do
    case $arg in
	c)
	    n_tests=$OPTARG
	    ;;
	s)
	    schedules=$OPTARG
	    ;;
	n)
	    n_tbs_per_sm=$OPTARG
	    ;;
	S)
	    n_sms=$OPTARG
	    ;;
	m)
	    multirun_range=$OPTARG
	    ;;
	t)
	    n_ths_per_tb=$OPTARG
	    ;;
	T)
	    tb_range=$OPTARG
	    ;;
	x)
	    benchdirect_mode=true
	    ;;
	H)
	    display_header=true
	    ;;
	*)
	    usage
	    exit 1
    esac
done

shift `expr $OPTIND - 1`

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

function get_elapsed_time() {
    sum=0
    for i in `seq 1 $n_tests`
    do
	str=`$MTBS $MTBS_ARGS $* 2> /dev/null | grep "^elapsed time:" | grep -Ewo '[[:digit:]]*\.[[:digit:]]*'`
	if [ $? -ne 0 ]; then
	    echo -n "-"
	    return
	else
	    sum=`echo "scale=6;$sum + $str" | bc`
	fi
    done
    value=`echo "scale=6;$sum / $n_tests" | bc`
    echo -n $value
}

function run_mtbs() {
    for s in $(echo $schedules | tr "," "\n")
    do
	get_elapsed_time -s $s $*
	echo -n ' '
    done
    echo
}

function show_header() {
    echo -n "#"
    for s in $(echo $schedules | tr "," "\n")
    do
	echo -n "$p "
    done
    echo
}

function run_mtbs_tbs_threads() {
    for ths in $(echo $n_ths_per_tb | tr "," "\n")
    do
	cmd_args=
	for arg in $*
	do
	    bench=`echo $arg | cut -d':' -f1`
	    bencharg=`echo $arg | cut -d':' -f2`
	    cmd_args="$cmd_args $bench:$tbs,1,$ths,1,$bencharg"
	done
	run_mtbs $cmd_args
    done
}

function run_mtbs_tb_per_sm() {
    for m in $(echo $n_tbs_per_sm | tr "," "\n")
    do
	tbs=$(($m * $n_sms))
	run_mtbs_tbs_threads $*
    done
}

function run_mtbs_multirun_range() {
    for multirun in $(seq $(echo $multirun_range | tr "," " "))
    do
	run_mtbs $multirun\*$*
    done
}

function run_mtbs_tbs_range() {
    for tbs in $(seq $(echo $tb_range | tr "," " "))
    do
	run_mtbs_tbs_threads $*
    done
}

if [[ -n $display_header ]]; then
    show_header
fi

if [[ -n $benchdirect_mode ]]; then
    run_mtbs $*
elif [[ -n $multirun_range ]]; then
    run_mtbs_multirun_range $*
elif [[ -n $tb_range ]]; then
    run_mtbs_tbs_range $*
else
    run_mtbs_tb_per_sm $*
fi

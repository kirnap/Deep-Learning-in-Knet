#!/bin/bash
#$ -N bilstm_experiment
#$ -S /bin/bash
#$ -q biyofiz.q
#$ -pe smp 1
#$ -cwd
#$ -o wordLM_largLSTM_exp.out
#$ -e wordLM_lrgLSTM.err
#$ -M okirnap@ku.edu.tr
#$ -m bea
#$ -l gpu=1
julia main.jl > bilstm.txt
#!/bin/bash

mkdir -p slurm 

# declare -a NSP=( 12 16 20 25 30 )  #6 8 10
declare -a NSP=( 2 4 6 8 10 12 16 20 ) #25 30 )  #6 8 10
declare -a LRS=( 0.0001 ) #0.001, 0.0003
declare -a MTD=("deepset" "regular" ) #"doubledeep" 

length1=${#NSP[@]}
length2=${#LRS[@]}
length3=${#MTD[@]}

for (( m=0; m<${length1}; m++ ))
do
for (( i=0; i<${length2}; i++ ))
do
for (( j=0; j<${length3}; j++ ))
do
    ID=nsp${NSP[m]}_lr${LRS[i]}_${MTD[j]}
    sed "s/ID/$ID/" sub.sub|sed s/NSP/${NSP[m]}/|sed s/LRS/${LRS[i]}/|sed s/MTD/${MTD[j]}/|sbatch
    echo $ID
done
done
done

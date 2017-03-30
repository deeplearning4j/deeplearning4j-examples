#!/bin/sh
i=1
load_100=0
load_95=0
load_80=0
load_10=0
echo "Displaying GPU utilization in percent of time:"
echo "100% util / 95% / 80% / less than 10%"
while true
do
   LOAD=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
   if [ "$LOAD" -eq 100 ]
   then
     load_100=$(( load_100 + 1 ))
   fi

   if [ "$LOAD" -gt 94 ]
   then
     load_95=$(( load_95 + 1 ))
   fi

   if [ "$LOAD" -gt 79 ]
   then
     load_80=$(( load_80 + 1 ))
   fi

   if [ "$LOAD" -lt 10 ]
   then
     load_10=$(( load_10 + 1 ))
   fi

   echo -n "$(( load_100 * 100 / i ))% / $(( load_95 * 100 / i ))% / $(( load_80 * 100 / i ))% / $(( load_10 * 100 / i ))%                  \r"
   i=$(( i + 1 ))
   sleep 0.1
done

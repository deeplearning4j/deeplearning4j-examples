#!/usr/bin/env bash

FROM=/mnt/resource/spark_testing/xyz
TO=/mnt/resource/spark_testing/xyz

for i in `seq 5 12`;
do
  #ssh -t 10.0.2.${i} "sudo mkdir -p /mnt/resource/spark/ && sudo chmod 777 /mnt/resource/spark && /opt/spark/sbin/start-slave.sh spark://10.0.2.4:7077"
  sudo mkdir ${TO}/10.02.${i}
  scp -r 10.0.2.${i}:${FROM} ${TO}/10.02.${i}
done


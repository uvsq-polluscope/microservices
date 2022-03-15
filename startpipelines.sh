#!/bin/bash

echo "Starting containers"
ssh root@192.168.33.124 'cd projet-M2-Datascale/microservices && docker-compose up -d'


sleep 5

echo "Starting microservices"
echo "Starting time series pre processing"
curl 192.168.33.124:8001/preprocessing 
sleep 5

echo "Starting GPS data pre processing"
curl 192.168.33.124:8002/preprocessing_gps 
sleep 5

echo "Starting time series classfication"
curl 192.168.33.124:8000/classification 
sleep 5

echo "Enabling controllers"
cd ../Infrastructure/connector && ./start.sh

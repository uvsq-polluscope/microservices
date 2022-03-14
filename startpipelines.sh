#!/bin/bash

echo "Starting containers"
docker-compose up -d 

sleep 30 

echo "Starting microservices"
echo "Starting time series pre processing"
curl +X GET 192.168.33.124:8001/classification

echo "Starting GPS data pre processing"
curl +X GET 192.168.33.124:8002/classification


echo "Starting time series classfication"
curl +X GET 192.168.33.124:8000/classification

#!/bin/bash

# MPAir.csv
if [ ! -f "MPAir.csv" ]; then
    gdown 1lklbnCBydhcs7y1wu9wK3o5aTXC3_EGO
fi

# Matching data 2022
if [ ! -f "MatchingData2022.xlsx" ]; then
    gdown 1aKzff4vq3VULWpQJzVk3Ibl4B0MEDwFw
fi
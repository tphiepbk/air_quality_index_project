#!/bin/bash

# MPAir.csv
if [ ! -f "MPair.csv" ]; then
    gdown 1lklbnCBydhcs7y1wu9wK3o5aTXC3_EGO
fi

# AOD data 2022
if [ ! -f "MatchingData2022.xlsx" ]; then
    gdown 1aKzff4vq3VULWpQJzVk3Ibl4B0MEDwFw
fi

# AOD data 2021
if [ ! -f "aod_data_daily.csv" ]; then
    gdown 1wncw0PQVhB2xwJoAR2wsUbmQi0k4yIRp
fi

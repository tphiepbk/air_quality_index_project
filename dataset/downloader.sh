#!/bin/bash

# MPair.csv
if [ ! -f "MPair.csv" ]; then
    gdown 11B67jA2x8l3JG6DCJK2s0BkDllEngM78
fi

# station2022.csv (used for MPair data)
if [ ! -f "station2022.csv" ]; then
    gdown 1QuHV4c9TrdvDSybSNFmno-6lgxoPcK8t
fi

# station2018-2021.csv (used for MPair data)
if [ ! -f "station2018-2021.csv" ]; then
    gdown 11ibVcOJ76ieY5qV7q7f_c2kSM516h1Ue
fi

# AOD data 2022
if [ ! -f "MatchingData2022.xlsx" ]; then
    gdown 1aKzff4vq3VULWpQJzVk3Ibl4B0MEDwFw
fi

# AOD data 2021
if [ ! -f "aod_data_daily.csv" ]; then
    gdown 1wncw0PQVhB2xwJoAR2wsUbmQi0k4yIRp
fi

# CMAQ dataset
if [ ! -d "concentration_station" ]; then
    gdown 1-yRpPuRlvDSEhR-PZftiprAMAckk1eGh
    tar xzf concentration_station.tar.gz
fi

#!/bin/bash

# MPair.csv
if [ ! -f "MPair.csv" ]; then
    gdown 1lklbnCBydhcs7y1wu9wK3o5aTXC3_EGO
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

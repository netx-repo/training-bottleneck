#!/bin/bash
AUTO_DIR="~/autorun"
LOG_DIR="~/autorun/chaokun_logs"

if [ ! -d "$LOG_DIR" ]; then
    cd $AUTO_DIR
    wget https://dt-training.s3.amazonaws.com/chaokun_logs.tar.gz
    tar -zxvf chaokun_logs.tar.gz
    rm chaokun_logs.tar.gz
fi
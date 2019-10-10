#!/usr/bin/env bash
python3 -m sres.eval \
--model srgan \
--model-path /storage/gan-model/SRGAN.pt \
--dataset div2k_val \
--root-dir /storage/DIV2K \
--num-workers 4
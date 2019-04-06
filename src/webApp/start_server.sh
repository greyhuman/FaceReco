#!/bin/bash
./open_browser.sh &
PYTHONPATH=$PYTHONPATH:../ /usr/bin/python ./app.py

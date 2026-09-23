#!/usr/bin/env bash
# Runs the whole pipeline end to end.
set -e
python -m src.collect_data --synthetic        # skip this line if data/midi/real/ already has your MIDI files
python -m src.preprocess
python -m src.train --epochs 40
python -m src.generate --notes 160 --count 3 --wav

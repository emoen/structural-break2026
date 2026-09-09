#!/bin/bash
cd /home/endrem/project/ADIA-Lab-Structural-Break/structural-break2026
export PYTHONWARNINGS=ignore
echo "===== BEFORE (backup, base79) ====="
.venv/bin/python _calib_nbrun.py streaming_detector_stack4.BEFORE_CALIB.ipynb 2>&1 | grep -E "pointwise matrix|Local TS-AUC|trained|Error|Traceback" 
echo "===== AFTER (calib, 158 cols) ====="
.venv/bin/python _calib_nbrun.py streaming_detector_stack4.ipynb 2>&1 | grep -E "pointwise matrix|Local TS-AUC|trained|Error|Traceback"
echo "===== DONE ====="

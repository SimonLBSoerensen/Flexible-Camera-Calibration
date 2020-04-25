
@echo off

echo Running without debugging.

python bundle_adjustment.py seed=0 cm_fit_control_points=False cm_order=3

pause
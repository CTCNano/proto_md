#!/bin/sh

python -m dms2 analyze \
       -struct /N/dc2/scratch/anabiman/ARGON/src.gro \
       -traj /N/dc2/scratch/anabiman/ARGON/md.trr \
       -subsystem_args "{'NumNodes_x': 80, 'NumNodes_y': 15, 'NumNodes_z': 15, 'Resolution': 10.0, 'Threshold':500, 'Scaling':20.0, 'Tol':0.1, 'NewIters':10}" \
       -subsystem_factory dms2.subsystems.ContinuumSubsystemFactory \
       -nframes 10 \
       -subsystem_selects 'all' \
       -var velocities \
       -ofname of_tmp \
       --plot

#!/bin/sh

python -m dms2_gro analyze \
       -struct /N/dc2/scratch/anabiman/HPV_16/Slowness/neutral_HPV_truncated_dry.gro \
       -traj /N/dc2/scratch/anabiman/HPV_16/Slowness/MD.trr \
       -subsystem_args "{'kmax':1}" \
       -subsystem_selects 'all' \
       -subsystem_factory dms2_gro.subsystems.LegendreSubsystemFactory \
       -nframes 1000 \
       -var velocities \
       -ofname of_tmp \
       --plot

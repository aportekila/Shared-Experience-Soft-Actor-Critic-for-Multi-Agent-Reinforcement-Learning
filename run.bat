@echo off

setlocal enabledelayedexpansion

set "seeds=0 1 2 3 4"
set "agent_types=SEAC SENAC"

for %%a in (%seeds%) do (
    for %%b in (%agent_types%) do (
        python train.py --agent_type %%b --seed %%a --env "Foraging-10x10-3p-3f-v2"
    )
)
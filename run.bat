@echo off

setlocal enabledelayedexpansion

set "seeds=0 1 2"
set "agent_types=ISAC"

for %%a in (%seeds%) do (
    for %%b in (%agent_types%) do (
        python train_off_policy.py --agent_type %%b --seed %%a --env "Foraging-10x10-3p-3f-v2" --total_env_steps 1000000
    )
)
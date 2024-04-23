seeds=(0 1 2 3 4)
agent_types=("IAC" "SNAC")
for seed in ${seeds[@]}; do
    for agent_type in ${agent_types[@]}; do
        python train.py --agent_type ${agent_type} --seed ${seed} --env "Foraging-10x10-3p-3f-v2"
    done
done


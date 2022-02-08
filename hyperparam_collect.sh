envs=("door-open-v2-goal-observable" "drawer-open-v2-goal-observable")
policy_lrs=("1e-3" "3e-4")
seeds=("2" "4") # only use even seeds, train_seed=seed-1, test_seed=seed
dts=("1" "5" "10")
discounts=(".99")

for env in ${envs[@]} ; do
    for policy_lr in ${policy_lrs[@]} ; do
        for seed in ${seeds[@]} ; do
            for dt in ${dts[@]} ; do
                for discount in ${discounts[@]} ; do
                    sbatch run.sh ${env} ${policy_lr} ${seed} ${dt} ${discount}
                done
            done
        done
    done
done

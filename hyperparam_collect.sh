envs=("door-open-v2-goal-observable" "drawer-open-v2-goal-observable")
policy_lrs=("1e-3" "3e-4")
seeds=("4" "42") # only use even seeds, train_seed=seed-1, test_seed=seed
dts=("1" "2" "5" "10")
discounts=(".99799195166" ".99598793558" ".99" ".9801")
mtls=("2500" "1250" "500" "250")

for env in ${envs[@]} ; do
    for policy_lr in ${policy_lrs[@]} ; do
        for seed in ${seeds[@]} ; do
            for i in {0..3} ; do
                sbatch run.sh ${env} ${policy_lr} ${seed} ${dts[i]} ${discounts[i]} ${mtls[i]}
            done
        done
    done
done

# for env in ${envs[@]} ; do
#     for policy_lr in ${policy_lrs[@]} ; do
#         for seed in ${seeds[@]} ; do
#             for dt in ${dts[@]} ; do
#                 for discount in ${discounts[@]} ; do
#                     for mtl in ${mtls[@]} ; do
#                         sbatch run.sh ${env} ${policy_lr} ${seed} ${dt} ${discount} ${mtl}
#                     done
#                 done
#             done
#         done
#     done
# done

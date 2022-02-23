cql_min_q_weights=("5")
policy_lrs=("3e-5")
qf_lrs=("3e-4")
# dts=(".01" ".02" ".04")
# discounts=(".99") # .994987 .99, .9801
# dts=("1" "2" "5" "10")
discounts=(".99")
seeds=("102")
# sparse_rewards=("True")
envs=("door-open-v2-goal-observable")

for cql_min_q_weight in ${cql_min_q_weights[@]} ; do
    for policy_lr in ${policy_lrs[@]} ; do
        for qf_lr in ${qf_lrs[@]} ; do
            # for dt in ${dts[@]} ; do
            for discount in ${discounts[@]} ; do
                for seed in ${seeds[@]} ; do
                    for env in ${envs[@]} ; do   
                        sbatch run.sh ${cql_min_q_weight} ${policy_lr} ${qf_lr} 0 ${discount} 500 ${seed} ${env}
                    done
                done
            done
            # done
        done
    done
done

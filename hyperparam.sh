cql_min_q_weights=("5")
# policy_lrs=("3e-5" "1e-4")
policy_lrs=("3e-4" "1e-3") # pendulum
qf_lrs=("3e-4") #"1e-4")
# discounts=(".99") # .994987 .99, .9801
discounts=(".99")
# seeds=("102") # manip
seeds=("42") # pend
# sparse_rewards=("True")
# envs=("door-open-v2-goal-observable")
# envs=("drawer-open-v2-goal-observable")
envs=("pendulum_.02")

for cql_min_q_weight in ${cql_min_q_weights[@]} ; do
    for policy_lr in ${policy_lrs[@]} ; do
        for qf_lr in ${qf_lrs[@]} ; do
            for discount in ${discounts[@]} ; do
                for seed in ${seeds[@]} ; do
                    for env in ${envs[@]} ; do   
                        sbatch run.sh ${cql_min_q_weight} ${policy_lr} ${qf_lr} 0 ${discount} 1000 ${seed} ${env}
                    done
                done
            done
        done
    done
done
                        # sbatch run.sh ${cql_min_q_weight} ${policy_lr} ${qf_lr} 0 ${discount} 500 ${seed} ${env}

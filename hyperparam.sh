cql_min_q_weights=("0.1" "1.0" "5.0" "10.0")
policy_lrs=("3e-4") # done:1e-4,3e-5 not done:3e-4 ("1e-4") #"3e-5" "3e-4")
qf_lrs=("3e-4" "3e-5")
dts=(".001")
alphas=(".1" ".5")

for cql_min_q_weight in ${cql_min_q_weights[@]} ; do
    for policy_lr in ${policy_lrs[@]} ; do
        for qf_lr in ${qf_lrs[@]} ; do
            for dt in ${dts[@]} ; do
                for alpha in ${alphas[@]} ; do
                    sbatch run.sh ${cql_min_q_weight} ${policy_lr} ${qf_lr} ${dt} ${alpha}
                done
            done
        done
    done
done

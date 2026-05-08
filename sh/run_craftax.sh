project="project-name"
total_timesteps=1_000_000_000

cd ../ && cd agents/craftax

# PPO-V, PPO-Q, PPO-V-RND, RePPO
for seed in 0 1 2;
    for ent_coef in 0.0 0.01; do
        python ppo_v.py --seed=$seed --wandb_project=$project --ent_coef=$ent_coef
        python ppo_q.py --seed=$seed --wandb_project=$project --ent_coef=$ent_coef
        python ppo_v_rnd.py --seed=$seed --wandb_project=$project --ent_coef=$ent_coef
    done

    for M in 1.2 1.4; do
        python reppo.py --seed=$seed --wandb_project=$project --m=$M --use_baseline --replace_q --total_timesteps=$total_timesteps --ent_coef=0.0
    done
done
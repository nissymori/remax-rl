wandb_project="tune_reppo"

for seed in 0 1 2 3 4; do
    for env in minatar-breakout minatar-asterix minatar-freeway minatar-space_invaders; do
        for M in 0.8 0.9 1. 1.2 1.4 1.6 2 3; do
            for gae_lambda in 0.65 0.8 0.95; do
                for replace_type in return q_r_m; do
                    python reppo.py env_name=$env seed=$seed wandb_project=$wandb_project M=$M replace_q=True use_wandb=True use_current_probs=True ent_coef=0.0 gae_lambda=$gae_lambda replace_type=$replace_type
                done
            done
        done
    done
done

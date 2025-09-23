wandb_project="tune_pqn"

for seed in 0 1 2 3 4; do
    for env in minatar-breakout minatar-space_invaders minatar-asterix minatar-freeway; do
        for gae_lambda in 0.65 0.8 0.95; do
            for lr in 0.00030.0005 0.001; do
                python pqn.py env_name=$env seed=$seed wandb_project=$wandb_project lambda_=$gae_lambda lr=$lr use_wandb=True num_envs=1024 num_steps=128 num_minibatches=128 num_epochs=3
            done
        done
    done
done
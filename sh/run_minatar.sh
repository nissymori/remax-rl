wandb_project="project-name"

cd ../ && cd agents/minatar

# PQN, PPO-V, PPO-Q, RePPO
for seed in 0 1 2 3 4 5 6 7 8 9; do
    for env in minatar-breakout minatar-asterix minatar-freeway minatar-space_invaders; do
        python pqn.py env_name=$env seed=$seed wandb_project=$wandb_project num_envs=1024 num_steps=128 num_minibatches=128 num_epochs=3 lambda_=0.8 lr=0.001
        for ent_coef in 0.0 0.01; do
            python ppo_q.py env_name=$env seed=$seed wandb_project=$wandb_project ent_coef=$ent_coef
            python ppo_v.py env_name=$env seed=$seed wandb_project=$wandb_project ent_coef=$ent_coef
        done
        for M in 0.8 0.9 1.0 1.2 1.4 1.6 2.0 3.0; do
            for ent_coef in 0.0 0.01; do
                python reppo.py env_name=$env seed=$seed wandb_project=$wandb_project M=$M replace_q=True use_wandb=True use_current_probs=True ent_coef=$ent_coef
            done
        done
    done
done
# For PPO-V-RND
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python ppo_v_rnd.py env_name=minatar-asterix seed=$seed wandb_project=$wandb_project rnd_lr=0.001 rnd_reward_coeff=1.0
    python ppo_v_rnd.py env_name=minatar-breakout seed=$seed wandb_project=$wandb_project rnd_lr=0.003 rnd_reward_coeff=1.0
    python ppo_v_rnd.py env_name=minatar-space_invaders seed=$seed wandb_project=$wandb_project rnd_lr=0.0001 rnd_reward_coeff=1.0
    python ppo_v_rnd.py env_name=minatar-freeway seed=$seed wandb_project=$wandb_project rnd_lr=0.0003 rnd_reward_coeff=1.5
    python ppo_v_rnd.py env_name=minatar-asterix seed=$seed wandb_project=$wandb_project rnd_lr=0.001 rnd_reward_coeff=1.0 ent_coef=0.0
    python ppo_v_rnd.py env_name=minatar-breakout seed=$seed wandb_project=$wandb_project rnd_lr=0.003 rnd_reward_coeff=1.0 ent_coef=0.0
    python ppo_v_rnd.py env_name=minatar-space_invaders seed=$seed wandb_project=$wandb_project rnd_lr=0.0001 rnd_reward_coeff=1.0 ent_coef=0.0
    python ppo_v_rnd.py env_name=minatar-freeway seed=$seed wandb_project=$wandb_project rnd_lr=0.0003 rnd_reward_coeff=1.5 ent_coef=0.0
done

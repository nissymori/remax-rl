project_name="project-name"
env_names=("Alien-v5" "Amidar-v5" "BattleZone-v5" "Frostbite-v5" "Hero-v5" "MsPacman-v5" "Qbert-v5" "Surround-v5" "WizardOfWor-v5" "Zaxxon-v5")

cd ../ && cd agents/atari

# PPO-V, PPO-Q, RePPO
for seed in 0 1 2 3 4; do
    for env_name in ${env_names[@]}; do
        for ent_coef in 0.0 0.01; do
            python ppo_v.py env_name=$env_name seed=$seed wandb_project=$project_name use_wandb=True ent_coef=$ent_coef
            python ppo_q.py env_name=$env_name seed=$seed wandb_project=$project_name use_wandb=True ent_coef=$ent_coef
        done
        for M in 0.8 0.9 1.0 1.2 1.4; do
            for ent_coef in 0.0 0.01; do
                python reppo.py env_name=$env_name seed=$seed wandb_project=$project_name M=$M replace_q=True use_wandb=True use_current_probs=True ent_coef=$ent_coef
            done
        done
    done
done
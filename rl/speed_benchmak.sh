wandb_project="reppo_speed_benchmark"

for seed in 0 1 2 3 4; do
    for env in minatar-breakout; do
        '''
        python pqn.py env_name=$env seed=$seed wandb_project=$wandb_project num_envs=1024 num_steps=128 num_minibatches=128 num_epochs=3 lambda_=0.8 lr=0.0001 do_eval=False
        for ent_coef in 0.0; do
            python ppo_q.py env_name=$env seed=$seed wandb_project=$wandb_project ent_coef=$ent_coef do_eval=False
            python ppo_v.py env_name=$env seed=$seed wandb_project=$wandb_project ent_coef=$ent_coef do_eval=False
        done
        '''
    
        for M in 1.2; do
            for gae_lambda in 0.8; do
                for replace_type in return; do
                    python reppo.py env_name=$env seed=$seed wandb_project=$wandb_project M=$M replace_q=True use_wandb=True use_current_probs=True ent_coef=0.0 gae_lambda=$gae_lambda replace_type=$replace_type do_eval=False
                done
            done
        done
    done
done
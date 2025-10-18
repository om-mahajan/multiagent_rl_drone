# Tools for performing tests of (OP)AC.

import os
import time


DURATION = 1
FREQ = 440


def doLinACtest(counter, runrun,
            actor_stepsize, critic_stepsize,
            actor_trace, critic_trace,
            gamma, cov, clip_grads, play_sound):
    
    t0 = time.time()
    
    runrun.init_agent(actor_stepsize, critic_stepsize, actor_trace,
                      critic_trace, gamma, cov, clip_grads)
    
    results = runrun.run()
    
    plot_name = 'results/fig{:d}.png'.format(counter)
    runrun.make_plot(results, plot_name)
    
    t1 = time.time()
    
    print('Completed run {} in {:.1f}s.'.format(
            counter, t1-t0))
    
    if play_sound:
        os.system('play -nq -t alsa synth {} sine {}'.format(
                DURATION, FREQ))


def doACtest(counter, runrun, actor_hidden_units, critic_hidden_units,
            actor_stepsize, critic_stepsize,
            actor_trace, critic_trace,
            gamma, cov, clip_grads, normalize, play_sound):
    
    t0 = time.time()
    
    runrun.init_agent(actor_hidden_units, critic_hidden_units,
                      actor_stepsize, critic_stepsize,
                      actor_trace, critic_trace,
                      gamma, cov, clip_grads, normalize)
    results = runrun.run()
    
    plot_name = 'results/fig{:d}.png'.format(counter)
    runrun.make_plot(results, plot_name)
    
    t1 = time.time()
    
    print('Completed run {} in {:.1f}s.'.format(
            counter, t1-t0))
    
    if play_sound:
        os.system('play -nq -t alsa synth {} sine {}'.format(
                DURATION, FREQ))
        

def doOPACtest(counter, trial_length, runrun, actor_hidden_units,
               critic_hidden_units,
               actor_stepsize, critic_stepsize,
               actor_trace, critic_trace,
               gamma, cov, clip_grads, normalize, play_sound):
    
    t0 = time.time()
    
    runrun.init_agent(actor_hidden_units, critic_hidden_units,
                      actor_stepsize, critic_stepsize,
                      actor_trace, critic_trace,
                      gamma, cov, clip_grads, normalize)
    
    runrun.run()
    
    results = runrun.test_target_policy(trial_length)
    
    plot_name = 'results/fig{:d}.png'.format(counter)
    runrun.make_plot(results, plot_name)
    
    t1 = time.time()
    
    print('Completed run {} in {:.1f}s.'.format(
            counter, t1-t0))
    
    if play_sound:
        os.system('play -nq -t alsa synth {} sine {}'.format(
                DURATION, FREQ))
        

# TODO: make a general experiment runner function.


def save_model(runner, filepath):
    """Save the trained multi-agent model parameters to a file.
    
    Args:
        runner: The runner object containing the trained metaagent
        filepath: Path to save the model (will be saved as .npz)
    """
    import numpy as np
    
    # Extract parameters from all agents
    model_data = {}
    
    for i, agent in enumerate(runner.metaagent.agents):
        # Save policy parameters
        model_data[f'agent_{i}_policy_params'] = agent.pi.get_params()
        
        # Save value function parameters
        model_data[f'agent_{i}_value_params'] = agent.v.get_params()
        
        # Save hyperparameters
        model_data[f'agent_{i}_actor_stepsize'] = agent.actor_stepsize
        model_data[f'agent_{i}_critic_stepsize'] = agent.critic_stepsize
        model_data[f'agent_{i}_lambda_pi'] = agent.lambda_pi
        model_data[f'agent_{i}_lambda_v'] = agent.lambda_v
        model_data[f'agent_{i}_gamma'] = agent.gamma
        
    # Save environment info
    model_data['num_agents'] = runner.env.num_agents
    model_data['observation_dim'] = runner.env.observation_dim
    model_data['num_actions'] = runner.env.num_actions
    
    # Save environment-specific attributes
    if hasattr(runner.env, 'num_states'):
        # ConvergenceTest environment
        model_data['num_states'] = runner.env.num_states
        model_data['num_state_features'] = runner.env.num_state_features
        model_data['env_type'] = 'ConvergenceTest'
    elif hasattr(runner.env, 'grid_dim'):
        # MAGridworld environment
        model_data['grid_dim'] = runner.env.grid_dim
        model_data['min_reward_distance'] = runner.env.min_reward_distance
        model_data['env_type'] = 'MAGridworld'
    else:
        model_data['env_type'] = 'Unknown'
    
    np.savez(filepath, **model_data)
    print(f"Model saved to {filepath}")


def load_model(runner, filepath):
    """Load trained model parameters into a runner's metaagent.
    
    Args:
        runner: The runner object with initialized metaagent
        filepath: Path to the saved model file
    """
    import numpy as np
    
    model_data = np.load(filepath, allow_pickle=True)
    
    for i, agent in enumerate(runner.metaagent.agents):
        # Load policy parameters
        agent.pi.set_params(model_data[f'agent_{i}_policy_params'])
        
        # Load value function parameters
        agent.v.set_params(model_data[f'agent_{i}_value_params'])
        
        # Load hyperparameters
        agent.actor_stepsize = float(model_data[f'agent_{i}_actor_stepsize'])
        agent.critic_stepsize = float(model_data[f'agent_{i}_critic_stepsize'])
        agent.lambda_pi = float(model_data[f'agent_{i}_lambda_pi'])
        agent.lambda_v = float(model_data[f'agent_{i}_lambda_v'])
        agent.gamma = float(model_data[f'agent_{i}_gamma'])
    
    print(f"Model loaded from {filepath}")
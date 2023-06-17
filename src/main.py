

import gymnasium as gym
from utils import remove_color, extract_player_position
from llm_agent import LLM_Agent
from termcolor import colored, cprint
import sys
import signal
import pprint
import time




print_green = lambda x: cprint(x, "green", "on_red")
print_red_on_cyan = lambda x: cprint(x, "red", "on_cyan")
print_yellow = lambda x: cprint(x, "yellow", "on_blue")

llm_type = "wizard-vicuna-llama"

metrics = {
    "llm_type": llm_type,
    "number_of_different_actions_taken": 0,
    "number_of_different_positions_visited": 0,
    "number_of_games_won": 0,
    "number_of_games_lost": 0,
    "cumulative_rewards_per_episode": [],
    "rewards_in_episode": {},
    "time_per_action": [],
    "when_won": [],
    "when_lost": [],

}


def signal_handler(sig, frame):
    print('Exiting game early due to ctrl+c')
    pprint.pprint(metrics)
    sys.exit(0)

def timeout_handler(signum, frame):
    print("Timeout")
    pprint.pprint(metrics)
    sys.exit(0)



def run_game(env: gym.Env , agent: LLM_Agent, action_space):

    start_episode_time = time.time()
    
    cumulative_reward = 0
    actions_taken = []
    positions_visited = []

    for i_episode in range(10):
        observation = env.reset()
        position = extract_player_position(env.render())
        agent.save_observation(position)

        metrics["rewards_in_episode"][i_episode] = []

        # Loop over t timesteps 
        for t in range(20):

            print()
            print_green(f"=> TIMESTAMP [{t}]; EPISODE [{i_episode}]<=")



            # Set position ot the last index in observations
            print(f"position: {position}")

            # Give the LLM some information about the current state of the game

            print_red_on_cyan("GENERATING ACTION")

            start = time.time()
        
            action, action_word = agent.generate_action(debug=True)

            end = time.time()
            metrics["time_per_action"].append(end - start)



            if action_word not in actions_taken:
                actions_taken.append(action_word)
                metrics["number_of_different_actions_taken"] += 1


            print(env.render())

            # take action
            # Within the game board state
            obs, reward, terminated, truncated, info = env.step(action)
            metrics["rewards_in_episode"][i_episode].append(reward)

            position = extract_player_position(env.render())

            if position not in positions_visited:
                positions_visited.append(position)
                metrics["number_of_different_positions_visited"] += 1


            agent.save_observation(position)

            print_red_on_cyan(f"GENERATED ACTION: {action_word}")

            # print("RENEDERED ACTION: ", env.render())
            #show the image of the render of the action
            print_red_on_cyan(f"CURRENT MAP")
            print(env.render())


            print_red_on_cyan(f"OUTPUT OF ACTION")
            print("--------------------")
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"terminated: {terminated}")
            print(f"truncated: {truncated}")
            print(f"info: {info}")
            print("--------------------")

            

            # save reward
            agent.save_reward(reward)

            # reflect on action

            print_red_on_cyan("reflecting")

            agent.reflect()

            cumulative_reward += reward

            # If the episode terminated prematurely, save the reward and stop the episode
            if terminated or truncated:
                if reward == 1:
                    metrics["number_of_games_won"] += 1
                    print_yellow("Episode won")
                    metrics["when_won"].append(time.time() - start_episode_time)
                elif reward == -1:
                    metrics["number_of_games_lost"] += 1
                    print_yellow("Episode lost")
                    metrics["when_lost"].append(time.time() - start_episode_time)

                print("Episode finished after {} timesteps".format(t+1))
                metrics["cumulative_rewards_per_episode"].append(cumulative_reward)
                cumulative_reward = 0
                
                agent.reflect_on_episode()

def init():
    # Launch adventure environment with human rendering
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="ansi")
    # render_mode="rgb_array"
    env.reset()

    env_map = remove_color(env.render())

    number_of_actions = env.action_space.n

    # to toggle to test the different LLM models
    # Choose which model to use
    # agent = LLM_Agent(type="llama") 
    agent = LLM_Agent(type=llm_type) 

    action_space = [ 'left', 'down', 'right', 'up']

    agent.set_action_space(action_space=action_space)

    agent.set_map(env_map)

    return env, agent, action_space

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(60 * 60)

 
    env, agent, action_space = init()
    run_game(env, agent, action_space)

    # TODO save agents hypothesis to file
    # TODO do evaluation of agent

if __name__ == "__main__":
    main()
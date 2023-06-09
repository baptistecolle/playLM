

import gymnasium as gym
from utils import remove_color, extract_player_position
from llm_agent import LLM_Agent
from termcolor import colored, cprint



print_green = lambda x: cprint(x, "green", "on_red")
print_red_on_cyan = lambda x: cprint(x, "red", "on_cyan")
print_yellow = lambda x: cprint(x, "yellow", "on_blue")

def run_game(env: gym.Env , agent: LLM_Agent, action_space):

    cumulative_reward = 0

    for i_episode in range(10):
        observation = env.reset()
        print(f"observation: {observation}")
        print()
        position = extract_player_position(env.render())
        
        
        agent.save_observation(position)

        episode_reward = 0
        print(f"episode_reward: {episode_reward}")

        # Loop over t timesteps 
        for t in range(20):

            print()
            print_green(f"=> TIMESTAMP [{t}]; EPISODE [{i_episode}]<=")

            position = extract_player_position(env.render())

            # Set position ot the last index in observations
            print(f"position: {position}")

            # Give the LLM some information about the current state of the game

            agent.save_observation(position)

            print_red_on_cyan("GENERATING ACTION")
        
            action, action_word = agent.generate_action(debug=True)
            print(env.render())

            # take action
            # Within the game board state
            obs, reward, terminated, truncated, info = env.step(action)

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
                    print_yellow("Episode won")
                elif reward == -1:
                    print_yellow("Episode lost")

                print("Episode finished after {} timesteps".format(t+1))
                cumulative_reward = 0
                
                agent.reflect_on_episode()

def init():
    # launch adventure environment with human rendering
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="ansi")
    # todo make this random maps
    # render_mode="rgb_array"
    env.reset()

    env_map = remove_color(env.render())

    number_of_actions = env.action_space.n
    agent = LLM_Agent()

    action_space = [ 'left', 'down', 'right', 'up']

    agent.set_action_space(action_space=action_space)

    agent.set_map(env_map)

    return env, agent, action_space

def main():
 
    env, agent, action_space = init()
    run_game(env, agent, action_space)

    # TODO save agents hypothesis to file
    # TODO do evaluation of agent

if __name__ == "__main__":
    main()
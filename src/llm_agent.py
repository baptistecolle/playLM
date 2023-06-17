from llama_cpp import Llama
import torch
import textwrap
from termcolor import colored, cprint
from llm import LLM

print_light_green = lambda x: cprint(x, "light_green")
print_light_magenta = lambda x: cprint(x, "light_magenta")


class LLM_Agent:

    def __init__(self, type):
        self.llm = LLM(type=type)
        self.type = type
        #, logits_all=True

        # RL specific variables
        self.observations = []
        self.rewards = []
        self.positions = []


        self.memory = [] # extract previous epsiodes from game
        self.belief = [] # extract rules from game
        self.actions_taken = [] # extract actions from game
        self.insights = [""] # extract insights from game
        self.exploration_rate = 1.0 # exploration rate
        self.map = None # extract map from game

        self.character_prompt = """
        You are a human. You are playing a videogame, you are thinking outloud about what you are doing. The game you are playing is a 2D puzzle game with a 4x4 square gridded map. 
        You cannot go outside the map, this is the map you are playing on and you can only move on the map.
        You are learning a game, you do not know the rules of the game but by playing the game you can learn the rules.
        You should try to explore and always be willing to participate in the game.
        """

        # You are located on the top left of the map and want to reach down right. (would this be cheating?)

        self.values = """
        This is a list of your values:
        - You should try to explore as much as possible and not go back to previous positions.
        """

    def set_map(self, map):
        self.map = map

    def set_action_space(self, action_space):
        self.action_space = action_space
        self.text_action_space = ', '.join(f"'{w}'" for w in action_space)
        print("action space: ", self.action_space)

    def set_character_prompt(self, character_prompt):
        self.character_prompt = character_prompt

    def save_observation(self, observation):
        self.observations.append(observation)
        print(f"saved observation: {observation}")

    def save_reward(self, reward):
        self.rewards.append(reward)

    def decrease_exploration_rate(self, amount=0.1):
        self.exploration_rate = max(0, self.exploration_rate - amount)

    def get_base_prompt(self):
        return f"""            
        {self.character_prompt}
        You can move in the following directions: {self.text_action_space}. 
        This is the map you are playing on:
        {' '.join(self.map)}
        {self.values}
        Your starting position is {self.observations[0]}.
        This is your current position in the game {self.observations[-1]}.
        This is the positions you got ranging from starting position to current position:: 
        {self.observations}
        You selected the following actions during your current game:
        {self.actions_taken}
        You received the following rewards during your current game: 
        {self.rewards}
        
        """

    def get_action_prompt(self):

        subset_insights = self.insights[-10:] if self.type == "llm" else self.insights[-1:] # because context window size is limited

        prompt = (
            f"""
            {self.get_base_prompt()}
            This is a list of your insights based on each move you have made, try to learn from them before making your next move:
            {' '.join(subset_insights)}
            What direction should the player move? 
            Answer:
            """
            # (please provide only the direction name, one word) 
           # If you are unsure, please output one of the directions listed before as you are learning how to play
        #     "Please output one of the possible actions. "
        #    "Choose an action to take: "
        )

        #  You have a exploration rate of {self.exploration_rate * 100}%. 
        #     If your exploration rate is 100% you will just explore randomly (by choosing a random action), if it is 0% you will just use your insights.


        print_light_green(f"action prompt: {prompt}")

        generated_action = self.llm.get_next_token_from_set(prompt, self.action_space)
    
        self.actions_taken.append(generated_action)

        

        return generated_action

    def reflect(self):



        prompt = (
            f"""
            {self.get_base_prompt()}
            {"You did not move. This is bad" if self.observations[-1] == self.observations[-2] else "" }

            Please reflect on why you received the rewards you did.
            Are you going in the right direction?
            If not what direction should you go in?
            Make an answer that is concise and actionable.
            Answer:
            As a human,
            """
            # (please provide only the direction name, one word) 
           # If you are unsure, please output one of the directions listed before as you are learning how to play
        #     "Please output one of the possible actions. "
        #    "Choose an action to take: "
        )


        print_light_magenta(f"reflecting prompt: {prompt}")

        insights = self.llm(prompt)



        print(f'generation: {insights}')

        self.insights.append(insights)

        # did you get a reward 
        # is reward negative or positive
        
        # make hypothesis why you got reward
        # did you move 

        pass
        
    def reflect_on_episode(self):

        # self.hypothesis
        """ 
        Please make an hypothesis on the game rules based on the observations and rewards you received.
        """
        self.observations = self.observations[-1:]
        self.rewards = []
        self.positions = []
        self.actions_taken = []




    def generate_action(self, debug=False):

        action_word = self.get_action_prompt()


        # convert action word to action number 
        action_number = self.action_space.index(action_word)

        print(f"action_word: {action_word}, action_number: {action_number}")

        return action_number, action_word
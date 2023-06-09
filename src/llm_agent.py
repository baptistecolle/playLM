from llama_cpp import Llama
import torch
import textwrap

class LLM_Agent:

    def __init__(self):
        self.llm = Llama(model_path="./model/wizardLM-7B.ggmlv3.q4_1.bin", logits_all=True, verbose=False, n_ctx=2048)
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
        You are:
        {self.character_prompt}
        You can move in the following directions: {self.text_action_space}. 
        This is the map you are playing on:
        {' '.join(self.map)}
        {self.values}
        This is your current position in the game {self.observations[-1]}.
        """

    def get_action_prompt(self):
        prompt = (
            f"""
            {self.get_base_prompt()}
            This is a list of your insights based on each move you have made, try to learn from them before making your next move:
            {' '.join(self.insights)}
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


        print("action prompt: ", prompt)

        max_logprob = -100
        max_action = None
        probs = []

        for action in self.action_space:
            # print(f"action: {action}")
            generation = self.llm(f"""{prompt}
                          {action}
                          """, logprobs=10, max_tokens=1, echo=True)
            
            tokens = generation['choices'][0]['logprobs']['tokens']
            # print(f"tokens: {tokens}")
            answer_index = tokens.index(" Answer")

            action_index = tokens.index(f" {action}", answer_index)
            action_logprob = generation['choices'][0]['logprobs']['token_logprobs'][action_index]
            
            action_in_list = tokens[action_index]
            assert action_in_list == f" {action}", f"action_in_list: {action_in_list}, action: {action}"
            # print(f"action_logprob: {action_logprob}")

            probs.append(action_logprob)
            # if (action_logprob > max_logprob):
            #     max_logprob = action_logprob
            #     max_action = action

        # print(f"probs: {probs}")
        probs = torch.nn.Softmax(dim=0)(torch.tensor(probs, dtype=torch.float))
        # print(f"probs: {probs}")
        max_action_index = torch.multinomial(probs, 1)
        max_action = self.action_space[max_action_index]

            
        assert max_action is not None, "max_action is None" 

        self.actions_taken.append(max_action)

        return max_action

    def reflect(self):

        

        prompt = (
            f"""
            {self.get_base_prompt()}
            You selected the following actions:
            {self.actions_taken}.
            You received the following rewards: 
            {self.rewards}.
            This is the positions you got: 
            {self.observations}.
            Please reflect on why you received the rewards you did.
            Are you going in the right direction?
            If not what direction should you go in?
            Make an answer that is concise and actionable.
            Answer:
            """
            # (please provide only the direction name, one word) 
           # If you are unsure, please output one of the directions listed before as you are learning how to play
        #     "Please output one of the possible actions. "
        #    "Choose an action to take: "
        )


        print("reflecting prompt: ", prompt)

        generation = self.llm(prompt)
        insight = generation["choices"][0]["text"]



        print(f'generation: {insight}')

        self.insights.append(insight)

        # did you get a reward 
        # is reward negative or positive
        
        # make hypothesis why you got reward
        # did you move 

        pass
        
    def reflect_on_episode(self):

        raise NotImplementedError

        # self.insights = [""] 
        # self.hypothesis
        """ 
        Please make an hypothesis on the game rules based on the observations and rewards you received.
        """
        pass

    def reflect_on_game(self):
        pass
    
    def generate_action(self, debug=False):

        action_word = self.get_action_prompt()


        # convert action word to action number 
        action_number = self.action_space.index(action_word)

        print(f"action_word: {action_word}, action_number: {action_number}")

        return action_number, action_word
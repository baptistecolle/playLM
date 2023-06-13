from llama_cpp import Llama
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


class LLM:

    def __init__(self, type) -> None:

        self.type = type

        if type == "llama":
            self.llm = Llama(model_path="./model/wizardLM-7B.ggmlv3.q4_1.bin", logits_all=True, verbose=False, n_ctx=2048)
        elif type == "gpt3":
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        else:
            raise Exception("LLM type not supported")

    def __call__(self, prompt):
        if self.type == "llama":
            generation = self.llm(prompt)
            print(generation)
            gen_text = generation["choices"][0]["text"]
            return gen_text
        elif self.type == "gpt3":
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=200,
            )
            gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
            return gen_text
        else:
            raise Exception("LLM type not supported")
        
    def get_next_token_from_set(self, prompt, word_set):

        probs = []

        if self.type == "llama":
            for word in word_set:
                # print(f"word: {word}")
                generation = self.llm(f"""{prompt}
                            {word}
                            """, logprobs=10, max_tokens=1, echo=True)
                
                tokens = generation['choices'][0]['logprobs']['tokens']
                # print(f"tokens: {tokens}")
                answer_index = tokens.index(" Answer")

                word_index = tokens.index(f" {word}", answer_index)
                word_logprob = generation['choices'][0]['logprobs']['token_logprobs'][word_index]
                
                # word_in_list = tokens[word_index]
                # assert word_in_list == f" {word}", f"word_in_list: {word_in_list}, word: {word}"
                probs.append(word_logprob)

        elif self.type == "gpt3":            
            self.tokenizer

            prompt_with_word = f"""{prompt}
                                """
            

            
            input_ids = self.tokenizer(prompt_with_word, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=1,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True
                
            )

            print(f"gen_tokens: {gen_tokens}")

            # for each word in the word set, get the probability of the word to be the next word using the scores of the generated token
            for word in word_set:

                word_index = self.tokenizer.encode(word)[0]
                print(gen_tokens['scores'][0].shape)
                print(word_index)
                word_score = gen_tokens['scores'][0][:, word_index]
                word_score = word_score[0].item() if not torch.isinf(word_score[0]) else 0
                print(f"word: {word}, word_score: {word_score}")
                probs.append(word_score)

           
                
                

                
        probs = torch.nn.Softmax(dim=0)(torch.tensor(probs, dtype=torch.float))
        # print(f"probs: {probs}")
        max_word_index = torch.multinomial(probs, 1).item()

        max_word = word_set[max_word_index]

        return max_word

            

            
        
            



import json
import numpy as np
import re
import functools
import string
import spacy
import sys
import nltk
import openai
from rank_bm25 import BM25Okapi
import os
import time
from nltk.tokenize import sent_tokenize
from scipy.optimize import linear_sum_assignment

from openai_lm import OpenAIModel

nltk.download("punkt")


class BioSplitter(object):
    def __init__(self, key_path, demon_dir, gpt3_cache_file=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = True
        self.demon_path = os.path.join(demon_dir, "split_bio_demons.json")

        #self.openai_lm = OpenAIModel("InstructGPT", cache_file=gpt3_cache_file, key_path=key_path)
        self.openai_lm = OpenAIModel(
            "ChatGPT",
            cache_file=gpt3_cache_file,
            key_path=key_path,
        )


        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_cache(self):
        self.openai_lm.save_cache()

    def run(self, generations, atomic_facts_list, cost_estimate=None):
        """ 
        Perform coreference resolution for the atomic facts based on the generation.
        Return a total words cost if cost_estimate != None.
        """
        assert isinstance(generations, list), "generations must be a list"
        assert isinstance(atomic_facts_list, list), "atomic_facts_list must be a list"
        assert isinstance(generations[0], str), "generation must be a string"
        assert isinstance(atomic_facts_list[0], list), "atomic_facts must be a list"
        return self.group_facts(generations, atomic_facts_list, cost_estimate=cost_estimate)

    def group_facts(self, generations, atomic_facts_list, cost_estimate=None):
        """
        Split the atomic facts into groups based on the generation.
        If the generation contains multiple biographies, split the atomic facts into groups based on the biographies.
        Return a total words cost if cost_estimate != None.
        """

        is_bio = self.is_bio
        demons = self.demons

        k = 0 if is_bio else 0 # number of top demons to consider. Currently it is set to 0 because we only have 4 demos.
        n = 4 if is_bio else 8 # number of initial demons to consider

        prompts = []
        prompt_to_generation = {}
        atoms = {}
        
        no_fact_count = 0
        for generation, atomic_facts in zip(generations, atomic_facts_list):

            # First, handle the case when the model abstained from generating anything and the atomic facts are None
            if atomic_facts is None:
                prompts.append("NO FACTS_" + str(no_fact_count) + " " + generation)
                prompt_to_generation["NO FACTS_" + str(no_fact_count) + " " + generation] = generation
                no_fact_count += 1
                continue

            top_machings = best_demos(generation, self.bm25, list(demons.keys()), k)
            prompt = ""

            for i in range(n):
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
                # The original atomic facts
                for fact in demons[list(demons.keys())[i]][0]:
                    prompt = prompt + "- {}\n".format(fact)

                prompt = prompt + "Next, refer to the paragraph again and see if it explicitly states that it contains the biographies of multiple individuals. If there are multiple biographies, split the independent facts from different biography using \"- ===\". If the paragraph does not contain multiple biographies from different individuals, repeat the independent facts.\n"
                # The atomic facts for different biographies
                for fact in demons[list(demons.keys())[i]][1]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            #for match in top_machings:
            #    prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
            #    # The original atomic facts
            #    for fact in demons[list(demons.keys())[i]][0]:
            #        prompt = prompt + "- {}\n".format(fact)
            #
            #    prompt = prompt + "Next, refer to the paragraph again and see if it explicitly states that it contains the biographies of multiple individuals. If there are multiple biographies, split the independent facts from different biography using \"- ===\". If the paragraph does not contain multiple biographies from different individuals, repeat the independent facts.\n"
            #    # The atomic facts for different biographies
            #    for fact in demons[list(demons.keys())[i]][1]:
            #        prompt = prompt + "- {}\n".format(fact)
            #    prompt = prompt + "\n"

            prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(generation)
            for fact in atomic_facts:
                prompt = prompt + "- {}\n".format(fact)
            prompt = prompt + "Next, refer to the paragraph again and see if it explicitly states that it contains the biographies of multiple individuals. If there are multiple biographies, split the independent facts from different biography using \"- ===\". If the paragraph does not contain multiple biographies from different individuals, repeat the independent facts.\n"
            prompts.append(prompt)
            prompt_to_generation[prompt] = generation

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                if prompt.startswith("NO FACTS"):
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            for prompt in prompts:
                if prompt.startswith("NO FACTS"):
                    # I change this to prompt instead of prompt to generation to avoid the case where the generation is None
                    atoms[prompt] = None
                    continue
                output, _ = self.openai_lm.generate(prompt)
                ungroup_facts = text_to_sentences(output)
                group = []
                for fact_idx, fact in enumerate(ungroup_facts):
                    if fact_idx == 0 or (fact == "- ===" or "===" in fact):
                        group.append([])
                    else:
                        group[-1].append(fact)

                atoms[prompt_to_generation[prompt]] = group
                if len(atoms) % 10 == 0:
                    self.save_cache()
            self.save_cache()
            # I am not sure why this is needed, but it is in the original FActScore code
            # for key, value in demons.items():
            #     if key not in atoms:
            #         atoms[key] = value
            return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# transform InstructGPT output into sentences
def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences

def maximize_selection(matrix):
    # Convert the maximization problem to a minimization problem
    # Subtract all elements from a large number
    max_value = matrix.max()
    modified_matrix = max_value - matrix

    # Apply the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(modified_matrix)

    # Calculate the total sum of the selected elements
    total_sum = matrix[row_indices, col_indices].sum()

    return total_sum, row_indices, col_indices

def main():
    generator = CoreferenceResolver("api.key", "demos", gpt3_cache_dir=None)
    atomic_facts, para_breaks = generator.run("There are multiple people with the name Miguel Suárez. Miguel A. Suárez Fernández was a Cuban lawyer and politician, born in 1902 in Placetas, Cuba. He served as the Cuban Foreign Minister in 1951 and went into exile after Fidel Castro overthrew the government in 1959. Miguel Ángel Suárez was a Puerto Rican soap opera and movie actor, born in 1939 in Santurce, Puerto Rico. Miguel Ángel González Suárez was a Spanish retired footballer, born in 1947 in Ourense, Galicia. Mario Suárez was one of the earliest Chicano writers, born in 1925 in Arizona, USA.")

    print(atomic_facts)
    print(para_breaks)

if __name__ == "__main__":
    main()
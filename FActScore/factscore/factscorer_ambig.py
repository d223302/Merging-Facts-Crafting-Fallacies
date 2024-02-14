import argparse
import string
import json
import numpy as np
import re
import os
import logging

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained, remove_citation
from atomic_facts import AtomicFactGenerator
from coreference import BioSplitter, maximize_selection
from factscore.clm import CLM
from factscore.npm import NPM
from openai_lm import OpenAIModel
from retrieval import DocDB, Retrieval

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.bio_splitter = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  topics,
                  names,
                  generations,
                  psgs=[],
                  gamma=10,
                  atomic_facts=None,
                  coref_resolved_atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "ChatGPT.pkl"))

            # estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="gpt-3.5-turbo")

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, name, generation, facts, passages in zip(topics, names, generations, atomic_facts, psgs):
                if facts is not None:
                    total_words += self._get_score(topic, name, generation, facts, knowledge_source, passages, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        num_supporting_topics = []
        for topic, name, generation, facts, passages in zip(topics, names, generations, atomic_facts, psgs):
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, name, generation, facts, knowledge_source, passages)
                score = np.mean([d["is_supported"] for d in decision])
                
                # num of supporting facts need special consideration
                # First, calculate which topic occurs the most
                topic_count = {}
                for d in decision:
                    if d["supporting_topic"] == []:
                        continue
                    for t in d["supporting_topic"]:
                        topic_count[t] = topic_count.get(t, 0) + 1
                if len(topic_count)==0:
                    num_supporting_topics.append(0)
                else:
                    # Sort the topic based on the occurence from the most to the least
                    all_topics = sorted(topic_count.keys(), key=lambda x: topic_count[x], reverse=True)
                    for d in decision:
                        if d["supporting_topic"] == []:
                           d["supporting_topic"] = None 
                           continue
                        # Select the topic based on the occurence from the most to the least
                        for t in all_topics:
                            if t is None:
                                continue
                            if t in d["supporting_topic"]:
                                d["supporting_topic"] = t
                                break
                    num_supporting_topics.append(len(set([d["supporting_topic"] for d in decision if d["supporting_topic"] is not None])))
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/(len(facts) + 1e-10))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {
            "score": np.mean(scores),
            "respond_ratio": respond_ratio,
            "decisions": decisions,
            "num_supporting_topics": np.mean(num_supporting_topics),
            "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])
        }

        if gamma:
            out["init_score"] = np.mean(init_scores)

        print(f"number of atomic facts: {len(atomic_facts)}")

        # The following parts are for GlobalFActScore
        if coref_resolved_atomic_facts is None:
            if self.bio_splitter is None:
                self.bio_splitter = BioSplitter(key_path=self.openai_key,
                                                          demon_dir=os.path.join(self.data_dir, "demos"),
                                                          gpt3_cache_file=os.path.join(self.cache_dir, "ChatGPT.pkl"))
            # estimate the total cost of coreference resolution
            total_words = self.bio_splitter.run(
                generations, 
                atomic_facts, 
                cost_estimate=self.cost_estimate
            )
            self.print_cost_estimates(total_words, task="split biographies", model="gpt-3.5-turbo")
            # The returned coref_resolved_atomic_facts is dictionary and the key is the index of the generation (full bio)
            coref_resolved_atomic_facts = self.bio_splitter.run(generations, atomic_facts)
            
            assert len(coref_resolved_atomic_facts)==len(topics), print(f"len(coref_resolved_atomic_facts): {len(coref_resolved_atomic_facts)}, len(topics): {len(topics)}")
            self.bio_splitter.save_cache()
        

        coref_resolved_atomic_facts = [v for k, v in coref_resolved_atomic_facts.items()]
        assert len(coref_resolved_atomic_facts)==len(topics)


        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, name, generation, resolved_facts, passages in zip(topics, names, generations, coref_resolved_atomic_facts, psgs):
                if resolved_facts is not None:
                    total_words += self._get_global_score(topic, name, generation, resolved_facts, knowledge_source, passages, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="globalfactscore evaluation", model="gpt-3.5-turbo")

        global_scores = []
        init_global_scores = []
        global_decisions = []
        global_num_supporting_topics = []
        num_groups = []


        for topic, name, generation, resolved_facts, passages in zip(topics, names, generations, coref_resolved_atomic_facts, psgs):
            if isinstance(topic, str):
                topic = [topic]

            if resolved_facts is None:
                global_decisions.append(None)
            else:
                global_decision = self._get_global_score(topic, name, generation, resolved_facts, knowledge_source, passages)

                potential_score_matrix = np.ones((len(topic), len(global_decision))) * -10

                if len(global_decision) > len(topic):
                    logging.critical("The number of groups is more than the number of topics")
                    
                for group_id, group in enumerate(global_decision):
                    for topic_id, t in enumerate(topic):
                        potential_score_matrix[topic_id, group_id] \
                            = np.sum([d["is_supported"] for d in group[t]])

                # Use Hungarian algorithm to find the best matching
                # _, selected_topics, _ = maximize_selection(potential_score_matrix)
                # I just use greedy algorithm to find the best matching
                selected_topics = np.argmax(potential_score_matrix, axis=0)
                assert len(selected_topics) == len(global_decision)
                # print(f"selected_topics: {selected_topics}")
                # print(f"potential_score_matrix: \n{potential_score_matrix}")
                
                final_decision = []
                for group_id, group in enumerate(global_decision):
                    topic_id = selected_topics[group_id]
                    final_decision.extend(group[topic[topic_id]])
                    
                #print(f"final_decision: {final_decision}")
                global_score = np.mean([d["is_supported"] for d in final_decision])
                
                # num of supporting facts need special consideration
                new_final_decision = []
                for d in final_decision:
                    if d["supporting_topic"] is not None and d["supporting_topic"] != []:
                        assert isinstance(d["supporting_topic"], list)
                        d["supporting_topic"] = d["supporting_topic"][0]
                    else:
                        d["supporting_topic"] = None
                    new_final_decision.append(d)
                final_decision = new_final_decision

                if len(resolved_facts)==0:
                    print(f"resolved_facts: {resolved_facts}")
                    print(f"generation: {generation}")

                if gamma:
                    init_global_scores.append(global_score)
                    penalty = 1.0 if len(resolved_facts)>gamma else np.exp(1-gamma/(len(resolved_facts) + 1e-10))
                    global_score = penalty * global_score

                global_scores.append(global_score)
                global_decisions.append(final_decision)
                num_groups.append(len(global_decision))

                if len(global_scores) % 5 == 0:
                    self.save_cache()
        self.save_cache()

        out["global_score"] = np.nanmean(global_scores)
        if gamma:
            out["init_global_score"] = np.nanmean(init_global_scores)
        out["global_decisions"] = global_decisions
        out["num_groups"] = np.mean(num_groups)
        #out["global_num_supporting_topics"] = np.mean(global_num_supporting_topics)
        out["nan"] = np.isnan(global_scores).sum() * 1.0
        return out

    def _get_score(self, topic, name, generation, atomic_facts, knowledge_source, passages, cost_estimate=None):
        decisions = []
        total_words = 0
        if not isinstance(topic, list):
            topic = [topic]
        for atom in atomic_facts:
            atom = atom.strip()
            supporting_topics = []
            if self.lm:
                
                topic_outputs = []
                topic_passages = self.retrieval[knowledge_source].get_passages(topic, name, atom, k=5)
                
                for passages in topic_passages:
                    definition = "Answer the question about {} based on the given context.\n\n".format(name)
                    context = ""
                    
                    for psg_idx, psg in enumerate(reversed(passages)):
                        context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                    definition += context.strip()
                    if not definition[-1] in string.punctuation:
                        definition += "."
                    prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())



                    if cost_estimate:
                        if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                            total_words += len(prompt.split())
                        elif cost_estimate == "ignore_cache":
                            total_words += len(prompt.split())
                        continue

                    output = self.lm.generate(prompt)
                    topic_outputs.append(output)

                if cost_estimate:
                    continue

                for topic_id, output in enumerate(topic_outputs):
                    if type(output[1])==np.ndarray:
                        # when logits are available
                        logits = np.array(output[1])
                        assert logits.shape[0] in [32000, 32001]
                        true_score = logits[5852]
                        false_score = logits[7700]
                        is_supported = true_score > false_score
                    else:
                        # when logits are unavailable
                        generated_answer = output[0].lower()
                        if "true" in generated_answer or "false" in generated_answer:
                            if "true" in generated_answer and "false" not in generated_answer:
                                is_supported = True
                            elif "false" in generated_answer and "true" not in generated_answer:
                                is_supported = False
                            else:
                                is_supported = generated_answer.index("true") > generated_answer.index("false")
                        else:
                            is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])
                    # If the answer can be supported by any of the passages, we consider it as supported
                    # This matches the original idea of FActScore
                    if is_supported:
                        supporting_topics.append(topic[topic_id])
                        
                if len(supporting_topics) != 0:
                    is_supported = True
            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3

            decisions.append({"atom": atom, "is_supported": is_supported, "supporting_topic": supporting_topics})
        if cost_estimate:
            return total_words
        else:
            return decisions

    def _get_global_score(self, topic_list, name, generation, grouped_atomic_facts, knowledge_source, passages, cost_estimate=None):
        total_words = 0
        global_decisions = [{} for _ in range(len(grouped_atomic_facts))]
        for group_id, group in enumerate(grouped_atomic_facts):
            for topic in topic_list:
                maybe_decisions_or_words =  self._get_score(topic, name, generation, group, knowledge_source, passages = None, cost_estimate=cost_estimate)
                if cost_estimate:
                    total_words += maybe_decisions_or_words
                else:
                    global_decisions[group_id][topic] = maybe_decisions_or_words
        
        if cost_estimate:
            return total_words
        else:
            return global_decisions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)


    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts, passages, names = [], [], [], [], []
    # The input format is the output format from ALCE
    f = json.load(open(args.input_path))['data']
    for dp in f:
        tot += 1
        if args.use_atomic_facts:
            assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
            if dp["annotations"] is None:
                continue
            topics.append(dp["topic"])
            generations.append(remove_citation(dp["output"]))
            atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            ## TODO: add passages
            passages.append(dp["docs"]) 
        else:
            names.append(dp['question'].split('a bio of')[-1].strip())
            topics.append(sorted(list(set([doc['title'] for doc in dp['docs']]))))
            generations.append(remove_citation(dp["output"]))
            passages.append(dp["docs"])
        if args.n_samples is not None and tot==args.n_samples:
            break
    out = fs.get_score(topics=topics,
                       names=names,
                       generations=generations,
                       psgs=passages,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))
    logging.critical("# Number of actual individuals per valid response = %.1f" % (out["num_supporting_topics"]))

    logging.critical("Global FActScore = %.1f%%" % (100*out["global_score"]))
    if "init_global_score" in out:
        logging.critical("Global FActScore w/o length penalty = %.1f%%" % (100*out["init_global_score"]))

    if "num_groups" in out:
        logging.critical("# Bios per valid response = %.1f" % (out["num_groups"]))
    logging.critical("# NaN = %.1f" % (out["nan"]))
    #logging.critical("# Supporting topics per valid response (GlobalFActScore) = %.1f" % (out["global_num_supporting_topics"]))

    # Save out as a json file
    with open(args.input_path.replace(".json", f"_factscore_output.json"), 'w') as f:
        f.write(json.dumps(out) + "\n")


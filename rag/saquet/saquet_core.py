"""SAQUET item-writing-flaw criteria converted from the supplied notebook.

Technical deviations from the notebook are limited to: reading the OpenAI API
key from ``OPENAI_API_KEY``; fixing ``absolute_terms_verify`` to accept the
``question`` object that its unchanged prompt formats; and lazy-loading the
unused CoLA / wellformedness HuggingFace models (eager import crashed natively
on Windows). See ``TECHNICAL_DEVIATIONS.md``.
"""

#Import the required libraries, lots of these are required for the LLMs we utilize for three criteria. 
import os
import Levenshtein
import spacy
import string
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import torch
from sentence_transformers import SentenceTransformer, util
from lexicalrichness import LexicalRichness
import re
import math
import random
import numpy as np
nlp = spacy.load('en_core_web_lg')
import time
from datetime import datetime
from collections import Counter
import warnings
import json
warnings.filterwarnings('ignore')

#NLTK Imports
import nltk
nltk.download('stopwords') #Needed for query wellformedness
nltk.download('punkt') #Needed for query wellformedness
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger') #Needed for query wellformedness
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import trigrams
from nltk import ngrams
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize, sent_tokenize

#OpenAI, but could be replaced with Gemini, Claude, etc.
from openai import OpenAI
model_engine = 'gpt-4o' #or the other models 'gpt-4o-mini','chatgpt-4o-latest', 'o1-preview' etc.

# Execution adaptation: bound API attempts and require an explicit request timeout.
# Prompts, thresholds, valid-response aggregation, and early stopping are unchanged.
SAQUET_LLM_MAX_ATTEMPTS = int(os.environ.get("SAQUET_LLM_MAX_ATTEMPTS", "8"))
SAQUET_LLM_TIMEOUT_S = float(os.environ.get("SAQUET_LLM_TIMEOUT_S", "60"))
SAQUET_LLM_RETRY_SLEEP_S = float(os.environ.get("SAQUET_LLM_RETRY_SLEEP_S", "10"))
SAQUET_LLM_RATING_MIN = 1
SAQUET_LLM_RATING_MAX = 10

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=SAQUET_LLM_TIMEOUT_S,
)


class SaquetEvaluationError(Exception):
    """Bounded SAQUET model calls exhausted without a valid response."""


def _parse_llm_rating_value(raw):
    """Accept only a single integer in the prompt's 1–10 range."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text.isdigit():
        return None
    value = int(text)
    if value < SAQUET_LLM_RATING_MIN or value > SAQUET_LLM_RATING_MAX:
        return None
    return value


def _chat_completion(*, messages, response_format=None):
    kwargs = {
        "model": model_engine,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "timeout": SAQUET_LLM_TIMEOUT_S,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)

#Libraries for Perplexity, Diversity, Grammatical Error, Complexity, Answerability
from evaluate import load
import language_tool_python

#Our class used to represent multiple-choice questions
nl = '\n'
class MultipleChoiceQuestion:
    def __init__(self, stem, options, correct_option, qid = None, quality = None):
        self.stem = stem
        self.options = options
        self.correct_option = correct_option
        self.qid = qid
        self.quality = quality
        
    def __str__(self):
        return f"Question: {self.stem}\n {nl.join(self.options)}\nCorrect option: {self.correct_option}\nQuality: {self.quality}"

def mode_or_average(lst):
    # Count occurrences of each number
    try:
        lst = [float(x) for x in lst]  # Handles strings of numbers
    except ValueError:
        raise ValueError("All elements in the list must be convertible to numbers.")
        
    count = Counter(lst)
    
    # Find the maximum frequency
    max_freq = max(count.values())
    # Find all numbers with the maximum frequency
    modes = [k for k, v in count.items() if v == max_freq]

    if len(modes) == 1:
        # If there's a single mode, return it
        return modes[0]
    else:
        # If no single mode, compute the average of two closest numbers
        sorted_lst = sorted(lst)
        min_diff = float('inf')
        closest_pair = None

        # Find the closest pair
        for i in range(len(sorted_lst) - 1):
            diff = sorted_lst[i + 1] - sorted_lst[i]
            if diff < min_diff:
                min_diff = diff
                closest_pair = (sorted_lst[i], sorted_lst[i + 1])
        
        # Compute and return the average of the closest pair
        return np.mean(closest_pair)

#Rating-based LLM Logic used for all verification steps where the LLM is needed
def llm_rating(sysrole, prompt, threshold=5):
    done = False
    ratings = []
    attempts = 0
    last_error = ""
    while not done:
        if attempts >= SAQUET_LLM_MAX_ATTEMPTS:
            raise SaquetEvaluationError(
                "SAQUET llm_rating exhausted {} attempts without enough valid "
                "1–10 ratings (collected {}, last_error={!r})".format(
                    SAQUET_LLM_MAX_ATTEMPTS, len(ratings), last_error
                )
            )
        attempts += 1
        try:
            o = _chat_completion(
              messages=[
                 {"role": "system",
                  "content": sysrole},
                {"role": "user", 
                 "content": prompt},
              ],
             )
            result = o.choices[0].message.content.lower().strip()
            parsed = _parse_llm_rating_value(result)
            if parsed is not None:
                ratings.append(str(parsed))
            else:
                last_error = "malformed_rating:{!r}".format(result)
        except SaquetEvaluationError:
            raise
        except Exception as error:
            last_error = str(error) or type(error).__name__
            print('errored in LLM API call: ', error)
            time.sleep(SAQUET_LLM_RETRY_SLEEP_S)
        if len(ratings) == 3:
            done = True
        if len(ratings) == 2:
            if float(ratings[0]) >= 7 and float(ratings[1]) >= 7:
                done = True
            if float(ratings[0]) <= 4 and float(ratings[1]) <= 4:
                done = True
                
    rating = mode_or_average(ratings)

    return bool(rating >= threshold)

#Uses NER, so if the score is too low, if they're matching entities (i.e. people) then we can ignore this case and say True
def implausible_distractors(question):
    #MiniLM from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    model = SentenceTransformer('all-MiniLM-L6-v2')
    correct = question.correct_option
    options = question.options.copy()

    # Split the correct answers (if multiple) and remove them from options
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)
            
    #Compute embedding for both lists
    embeddings1 = model.encode([correct] * len(options), convert_to_tensor=True)
    embeddings2 = model.encode(options, convert_to_tensor=True)
    
    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    #Output the pairs with their score
    for i, opt in enumerate(options):
        similarity_score = cosine_scores[i][i].item()

        if similarity_score < 0.15:
            # Perform Named Entity Recognition (NER) checks
            opt_entity = nlp(opt)
            lemma_nouns_opt = get_lemma_nouns(opt)
            
            ans_entity = nlp(correct)
            lemma_nouns_ans = get_lemma_nouns(correct)

            opt_entity_label = opt_entity.ents[0].label_ if opt_entity.ents else None
            ans_entity_label = ans_entity.ents[0].label_ if ans_entity.ents else None

            # If no entities or nouns can be parsed, we assume it's safe
            if not lemma_nouns_ans and not lemma_nouns_opt:
                return True

            # If it’s an "All of the above" or "None of the above" case, ignore similarity check
            if not all_of_the_above(question) or not none_of_the_above(question):
                return True

            # Apply Jaccard similarity and Levenshtein distance rules
            if jaccard_similarity(correct, opt) > 0.15 or Levenshtein.distance(correct, opt) < (len(correct) * 0.7):
                return True

            # LLM-based final check
            if implausible_distractors_verify(question):
                print('LLM confirms similarity: ', correct, ' -and- ', opt)
                return True

            print(f"Distractor too different: '{correct}' vs. '{opt}' | Score: {similarity_score:.4f}")
            return False

    return True

#Statistic used for gauging the similarity and diversity of text
def jaccard_similarity(str1, str2):
    # Convert strings to sets of words
    set1 = set(str1.split())
    set2 = set(str2.split())

    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate Jaccard Similarity
    similarity = len(intersection) / len(union)
    return similarity

def implausible_distractors_verify(question):
    sysrole = """You are a seasoned academic professional with extensive experience in designing and reviewing multiple-choice assessments. Your primary objective is to identify whether any distractors (incorrect answer choices) in a multiple-choice question are overly implausible—so unrelated or off-topic that students, even those with minimal subject knowledge, would not consider selecting them. Your assessment should focus on how well each distractor aligns with common misconceptions, partial knowledge, or thematic similarity to the correct answer."""
    prompt = """
Instructions:
    Assign a single numeric score from 1 to 10 indicating the plausibility of the distractors:

    Scoring Guidelines:
        1–3: At least one distractor is clearly off-topic, illogical, or so unrelated to the question, such that no student, even with minimal subject knowledge, would choose it.
        4–6: Distractors have noticeable plausibility issues; there may be some relevance, but at least one distractor still seems very out of place or unconnected to the question or other options.
        7–9: Distractors are reasonably plausible and relevant, though there might be minor clues or inconsistencies that reduce their effectiveness.
        10: All distractors are reasonably related or plausible, so a student with limited knowledge might actually pick any of them due to confusion, misunderstanding, or similarity to the correct answer.
        
    Respond with only the numeric score (1–10) and nothing else. Do not provide an explanation or any additional commentary.

Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)
    
    return llm_rating(sysrole, prompt)

def none_of_the_above(question):
    for opt in question.options:
        cleaned_opt = opt.strip().lower()
        if 'none of the above' in cleaned_opt or ('none' in cleaned_opt and 'above' in cleaned_opt) or cleaned_opt.startswith('none of') or cleaned_opt == 'neither' or 'none' in question.options[len(question.options)-1]:
            return False
    return True

def all_of_the_above(question):
    for opt in question.options:
        cleaned_opt = opt.strip().lower()
        if 'all of the above' in cleaned_opt or ('all' in cleaned_opt and 'above' in cleaned_opt) or ('all if the' in cleaned_opt)  or ('all of the' in cleaned_opt):
            return False

    return True

#Programming questions might contain a single underscore, so check for multiple
def fill_in_the_blank(question):
    if "__" in question.stem or ('fill in the blank' in question.stem.lower()):
        return False
    
    return True

def true_or_false(question):
    options = question.options.copy()
    
    #Check for true & false mentioned in the stem
    for sent in question.stem.split('.'):
        sent = sent.lower()
        if 'false' in sent and 'true' in sent:
            return False    
    
    for opt in options:
        cleaned_opt = opt.strip().lower() 
        if cleaned_opt == 'true' or cleaned_opt == 'false' or cleaned_opt == 'yes' or cleaned_opt == 'no':
            return False

    return True

#The list of absolute terms can be different for the stem and options, but we need to be careful here, as sometimes these can be used in proper ways
absolutes = ["always", "never", "none", "all", "completely", "absolutely", "totally", "definitely", "incapable", "inevitable"]
def absolute_terms(question):

    #Check for terms in the question stem, if we we find any, have GPT-4 help us verify the use of it.
    stem = question.stem.lower()
    if any(word in stem.split() for word in absolutes):
        if not true_or_false(question):
            return True
        else:
            return absolute_terms_verify(question)

    #Check for terms in the options, if we we find any, have GPT-4 help us verify the use of it.
    absolutes_options = ["always", "never", "none", "completely", "absolutely", "totally", "definitely", "incapable", "inevitable", "all"]
    for opt in question.options:
        cleaned_opt = opt.strip().lower()
        
        #Count all, which is a special case, but not in the case of "all of the above"
        if any(word in cleaned_opt for word in absolutes_options):        
            if none_of_the_above(question) and all_of_the_above(question) and true_or_false(question):
                if "all" in cleaned_opt: 
                    return absolute_terms_verify(question)
                return False
    
    return True

def absolute_terms_verify(question):
    sysrole = """You are an expert educator evaluating multiple-choice questions for the presence of absolute terms. Your task is to examine the question’s text and answer options for any words or phrases that might give away the correct answer or help a student eliminate an incorrect answer due to their overly certain or extreme language."""
    prompt = """
Instructions:
    Provide a single numeric score from 1 to 10, reflecting how the presence of absolute terms (e.g., “always”, “never”, “all”, “must”, "none", “only”) might help a student guess or eliminate incorrect options or if the question's stem contains absolute terms that are used in a way which constitutes a blanket generalization or hyperbole.

    Scoring Guidelines:
        1–3: The question or answer choices contain multiple or very strong absolute terms that greatly simplify finding the correct answer.
        4–6: The question or choices contain some absolute terms that might help clue the student toward the answer.
        7–9: The question or choices contain minimal absolute terms with limited impact on revealing the correct answer.
        10: The question and all answer options are free of absolute terms that could reveal the answer.
    
    Your response should only be the number from 1 to 10. Do not include any explanation or additional text.

Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)

    return llm_rating(sysrole, prompt)

#If the correct answer is noticably longer (more than 25%) than the second longest answer, flag it.
def longest_answer_correct(question):

    #Ignore this criteria for True/False questions
    if not true_or_false(question) or '[SEP]' in question.correct_option:
        return True
        
    correct = question.correct_option
    options = question.options.copy()
    
    # Split the correct answers (if multiple) and remove them from options
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)

    longest_option = 0
    for opt in options:
        if len(opt) >= longest_option:
            longest_option = len(opt)
        
    #If the longest option is only by 25% or it's a three words or less, then this passes
    if longest_option >= len(correct) *.75 or len(correct.split()) < 4:
        return True
    
    return False

#The list of negative words can potentially cause this to be too restrictive, particularly for words such as can't and won't
def negative_worded_stem(question):
    negatives = ["none", "never", "without", "exclude", "deny", "refuse", "oppose", "dispute", "can't", "won't", "not"] 

    stem = question.stem.lower()
    if any(word in stem.split() for word in negatives):
        return False

    for sent in question.stem.split('.'):
        sent = sent.lower()        
        if 'which' in sent and ('false' in sent or 'not' in sent or 'incorrect' in sent or 'except' in sent) or \
        'what' in sent and ('false' in sent or 'not' in sent or 'incorrect' in sent or 'except' in sent):
            return False    
    
    return True

#Find the nouns in question.correct_option and question.stem --> stem them --> compare cosine similiary (using sentence transformer)
#Also check for the synonyms, compare them. However, if the word(s) are used in the other options, then it's fine.
#Nouns: NN noun, singular ‘- desk’, NNS noun plural – ‘desks’, NNP proper noun, singular – ‘Harrison’, NNPS proper noun, plural – ‘Americans’ 
lemmatizer = WordNetLemmatizer()
nouns = ['NN', 'NNS', 'NNP', 'NNPS']

def word_repeats_in_stem_and_correct_answer(question):   
    options = question.options.copy()
    
    all_options = ' '.join(options)
    
    # Split the correct answers (if multiple) and remove them from options
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)
        
    #This code checks for matching words, specifically nouns and verbs, between the correct answer and stem
    word_types = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBN', 'VBP', 'VBZ']
    stem = strip_punctuation(question.stem)
    matching_words = []
    for wrd in stem.split():
        if wrd not in stop_words and wrd in question.correct_option:
            matching_words.append(wrd)
            
    matching_words = list(set(matching_words)).copy()
    matching_words_copy = matching_words.copy()
    for wrd in matching_words:
        for opt in options:
            if wrd in opt:
                matching_words_copy.remove(wrd)
                break

    #If the word is longer than 4 characters, because non-matching verbs/nouns of smaller characters typically are not cues
    if len([s for s in matching_words_copy if len(s) >= 4]) > 0:
        again = []
        tagged = nltk.pos_tag(matching_words_copy)
        for t in tagged:
            if t[1] in word_types:
                again.append(t[0].lower())
        if len(again) > 0:
            if '[SEP]' in question.correct_option:
                for mwc in matching_words_copy:
                    if mwc in question.correct_option.split('[SEP]')[0] and mwc in question.correct_option.split('[SEP]')[1]:
                        print('*** SEP')
                        return False
                    else:
                        return True
            else:
                #There's the potential false positive where all answer choices are repeated in the question's stem
                all_ops_in_stem = 0
                for opt in question.options:
                    opt = opt.lower()
                    stem = question.stem.lower()
                    if opt in stem:
                        all_ops_in_stem = all_ops_in_stem + 1
                if all_ops_in_stem == len(question.options):
                    return True
                
                return False
    return True

def strip_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation)

#This is now used for Logical Cue
def get_lemma_nouns(text):
    all_nouns = []
    tokenized = sent_tokenize(text)
    
    for i in tokenized:

        # Word tokenizers is used to find the words and punctuation in a string
        wordsList = nltk.word_tokenize(i)

        # removing stop words from wordList
        wordsList = [w for w in wordsList if not w in stop_words]

        # Using a Tagger. Which is part-of-speech tagger or POS-tagger.
        tagged = nltk.pos_tag(wordsList)
        
        # Add any nouns to this list
        for t in tagged:
            if t[1] in nouns:
                all_nouns.append(t[0].lower())
    
    lemmatized_nouns = []
    for n in all_nouns:
        lemmatized_word = lemmatizer.lemmatize(n, pos="n")
        lemmatized_nouns.append(lemmatized_word.lower())
    
    return lemmatized_nouns   

def word_repeats_verify(question):
    options = question.options.copy()
    
    # Split the correct answers (if multiple) and remove them from options
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)
        
    sysrole = """You are an expert educator evaluating multiple-choice questions to identify whether a unique key word, phrase, or term is shared exclusively between the stem and the correct answer. Such repetition can provide an unintended clue for testwise students. If no exclusive repetition of a unique key word or phrase exists, the question should be considered free of this issue."""
    prompt = """
Instructions:
    Respond with a score from 1–10 based on how prominently a unique key word, phrase, or term from the stem is repeated only in the correct answer (not appearing in any other options). Use the following Scoring Guidelines:

    Scoring Guidelines:
        1–3: A clearly noticeable or direct match of a unique key term between the stem and the correct answer makes the correct option stand out strongly.
        4–6: Some repetition of a key term provides a potential clue but is not blatantly revealing.
        7–9: Only minimal or subtle overlap in wording, unlikely to give a significant advantage to a testwise student.
        10: No unique repeated key words, phrases, or terms appear exclusively in the stem and the correct answer.

    Do not lower the score for common or generic terms (e.g., articles, conjunctions, basic verbs) unless they serve as an unmistakable clue.
    
    Only penalize repeated, uniquely identifying words or phrases that are exclusive to the stem and correct answer.
    
    If no exclusive repetition is found, assign a score of 10.

    Your response should only be the number from 1 to 10. Do not include any explanation or additional text.

Question Stem: {}
Options: {}
Answer: {}""".format(question.stem, options, question.correct_option)
    
    return llm_rating(sysrole, prompt)

#An example of a logical cue is asking students to select the most appropriate pharmaceutical intervention for a problem and only having one or two options which
#Using NER, if the question asks for a <certain type of noun, like a <person> then the options should all be <people> too.

def avoid_logical_cues(question):
    options = question.options.copy()

    # Split the correct answers (if multiple) and remove them from options
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)
    
    if len(options) < 2:
        return True
   
    lemma_nouns_answ = get_lemma_nouns(question.correct_option)
    lemma_nouns_options = [get_lemma_nouns(opt) for opt in options]
        
    entities_in_options = []
    for opt in lemma_nouns_options:
        for val in opt:
            doc = nlp(val)
            if doc.ents:
                entities_in_options.append(doc.ents[0].label_)
    
    entities_in_answer = []
  
    for val in lemma_nouns_answ:
        doc = nlp(val)
        
        #If the noun(s) in the answer choice can be tagged with an entity
        if doc.ents:
            answer_entity = doc.ents[0].label_
            if answer_entity not in entities_in_options:
                return logical_cue_verify(question)

    
    #If the stem has a number and only one option has a number
    numbers_in_stem = extract_all_numerical_values(question.stem)
    numbers_in_options = sum(1 for opt in options if extract_all_numerical_values(opt))

    if len(numbers_in_stem) > 0 and numbers_in_options == 1:
        return False
    
    return True

def logical_cue_verify(question):
    sysrole = """You are an expert educator evaluating multiple-choice questions for any unintended hints or logical cues in the stem and answer choices that might allow a testwise student to identify the correct option without true mastery of the content. Your primary objective is to spot clues, patterns, or giveaways that compromise the fairness or validity of the question."""
    prompt = """
Instructions:
    Respond with a score from 1–10 based on how likely it is that a testwise student could identify the correct answer through logical cues, based on the following Scoring Guidelines:

    Scoring Guidelines:
        1–3: The question or options contain significant or obvious clues, making it easy for a savvy testwise student to guess the correct answer.
        4–6: The question or options have some identifiable cues, but they are less obvious and require closer attention to exploit.
        7–9: The question is generally well-constructed, with only minor or subtle cues that might help a testwise student.
        10: The question is free of any noticeable cues, providing no advantages to a student guessing logically.

    Please do not lower the score for common question-design elements that are unavoidable or do not genuinely compromise fairness.  

    Your response should only be a single number from 1 to 10, with no additional explanation.

Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)

    return llm_rating(sysrole, prompt)

#If answer choices are numeric, sort them, compare to current order
#If all but one are numerical, make sure they are in order and the "word" option is last.

def lost_sequence(question):
    options = question.options.copy() 
    opts = []
    non_numerical_option = 0
    for opt in options:
        #First check for fractions
        fraction = extract_fraction_to_float(opt)
        if fraction:
            opts.append(fraction)
        else:
            val = extract_all_numerical_values(opt)
            if len(val) == 1:
                opts.append(float(val[0].replace(',', '')))
            else:
                non_numerical_option = non_numerical_option + 1
    
    if non_numerical_option > 0 and not(non_numerical_option == 1 and len(opts) == len(options)-1):
        return True

    float_options = [float(x) for x in opts]    
    sorted_options = sorted(float_options)
    reverse_sorted_options = sorted(float_options, reverse=True)
    
    if sorted_options == float_options:
        #Numeric options are sorted
        return True
    elif reverse_sorted_options == float_options:
        #Numeric options are sorted in reverse order, which might make sense for the question
        return True
    else:
        return False

def extract_all_numerical_values(s):
    pattern = r'-?\d*(?:,\d{3})*\.\d+|-?\d+(?:,\d{3})*'
    return re.findall(pattern, s)

# Regex pattern to match fractions with optional decimal numerator and/or denominator
def extract_fraction_to_float(s):
    pattern = r'-?\b\d+(\.\d+)?/\d+(\.\d+)?\b'
    match = re.search(pattern, s)
    if match:
        a , b = match.group().split("/")
        fraction = float(a) / float(b)
        return fraction
    else:
        return None

format_mtoc ={
    "Stem": "The Question",
    "correct_answers": [
        {
            "A": "The choice"
        },
        {
            "C": "The choice"
        }
    ]
}

def more_than_one_correct(question):
    # Ensure the question has at least 4 options
    if len(question.options) < 2:
        raise ValueError("A multiple-choice question must have at least two options.")
    
    sysrole = """You are an expert and an astute instructor. 
    Given a multiple-choice question and possible answers, determine the correct answers. 
    Your reply must be in the following JSON format: {}""".format(format_mtoc)

    # Dynamically construct the answer choices
    answer_choices = "\n".join([f"{chr(65+i)}: {option}" for i, option in enumerate(question.options)])

    # Create the prompt
    prompt = f"""
        question: {question.stem}
        
        answers:
        {answer_choices}
    """

    done = False
    expert_reasoning = 'blank'
    attempts = 0
    last_error = ""
    response = None

    while not done:
        if attempts >= SAQUET_LLM_MAX_ATTEMPTS:
            raise SaquetEvaluationError(
                "SAQUET more_than_one_correct exhausted {} attempts without a "
                "valid JSON response (last_error={!r})".format(
                    SAQUET_LLM_MAX_ATTEMPTS, last_error
                )
            )
        attempts += 1
        try:
            response = _chat_completion(
              response_format={"type": "json_object"},
              messages=[
                 {"role": "system", "content": sysrole},
                {"role": "user", "content": prompt},
              ],
             )
            parsed = json.loads(response.choices[0].message.content)
            if not _valid_more_than_one_correct_payload(parsed):
                last_error = "malformed_mtoc_json"
                continue
            expert_reasoning = parsed
            done = True
        except SaquetEvaluationError:
            raise
        except json.JSONDecodeError as error:
            last_error = "malformed_mtoc_json:{!r}".format(str(error))
        except Exception as error:
            last_error = str(error) or type(error).__name__
            print('errored in LLM API call: ', error)
            time.sleep(SAQUET_LLM_RETRY_SLEEP_S)

    return len(expert_reasoning["correct_answers"]) == 1


def _valid_more_than_one_correct_payload(payload):
    if not isinstance(payload, dict):
        return False
    answers = payload.get("correct_answers")
    if not isinstance(answers, list):
        return False
    for item in answers:
        if not isinstance(item, dict) or not item:
            return False
        for key, value in item.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return False
    return True

#If the answer options share the same words between one another and there are commas present then it's k type
def complex_k_type(question):    
    options = question.options.copy()
    
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    
    # Ensure all correct answers are removed safely
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)

    # If the question is not an "all of the above", "none of the above", or "true/false" type, return True
    if not (all_of_the_above(question) or none_of_the_above(question) or true_or_false(question)):
        return True 

    # If fewer than 2 options remain after removing correct answers, return True
    if len(options) < 2:
        return True 

    # Check if all remaining options contain a comma
    all_contain_comma = all(',' in opt for opt in options)

    # Extract lemma nouns from the correct answer and options
    lemma_nouns_answ = get_lemma_nouns(question.correct_option)
    lemma_nouns_options = [get_lemma_nouns(opt) for opt in options]

    # Count options that share at least one noun with the correct answer
    options_that_share_noun = sum(
        bool(set(lno) & set(lemma_nouns_answ)) for lno in lemma_nouns_options
    )

    # Check if all options are variations of "Yes" or "No" answers
    yes_no_options = sum(opt.lower().startswith(("yes", "no")) for opt in options)
    if yes_no_options == len(options):
        return True

    # If options share a key noun, contain multiple nouns, and include commas, it might be a K-type question
    if options_that_share_noun > 0 and all_contain_comma:
        return False

    # Normalize options by removing list-like formatting
    cleaned_options = [clean_string(opt) for opt in options]

    # Check if all cleaned options contain the same words
    options_word_sets = [set(opt.split()) for opt in cleaned_options]
    
    if len(set(map(frozenset, options_word_sets))) == 1:
        return False

    return complex_k_type_verify(question)

def clean_string(string):
    # remove whitespace
    cleaned_string = string.strip()
    
    # remove punctuation
    cleaned_string = re.sub(r'[^\w\s]', '', cleaned_string)
    
    # remove list notation
    cleaned_string = re.sub(r'\b(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\b', '', cleaned_string)
    cleaned_string = re.sub(r'\b(A|B|C|D|E|F)\b', '', cleaned_string)
    return cleaned_string


def complex_k_type_verify(question):
    sysrole = """You are an expert educator and assessment designer specializing in evaluating complex multiple-choice questions. Your primary task is to identify K-Type (Complex) multiple-choice questions. These questions require students to select from a range of combinations of responses, often using phrases like "and," "or," "only," or presenting multiple items and pairs separated by commas, semicolons, periods, or numbers.
    
    K-Type questions are characterized by options that represent different combinations of correct or partially correct answers. Unlike standard multiple-choice questions that have a single correct answer, K-Type questions involve evaluating multiple statements to determine which combination is valid.
    
    Your goal is to identify questions that fit this K-Type structure and distinguish them from traditional single-answer multiple-choice questions."""

    prompt = """
Instructions:
    Evaluate the multiple-choice question below to determine if it qualifies as a K-Type (Complex) question, where students are required to choose from a two or more combinations of responses, rather than a single correct answer, often using phrases like "and," "or," "only," or presenting multiple items and pairs separated by commas, semicolons, periods, or numbers.
    
    A K-Type question should have options that share different combinations of the same answer, just because an option has a few repeated keywords or several commas, does not mean it is K-Type.
    
    Provide a score from 1 to 10 based on the following Scoring Guidelines:
    
    Scoring Guidelines:
        1-3: The question is clearly a K-Type question, where options represent different combinations of possible responses. These options often use phrases like "and", "or", "only", or lists and pairings separated by commas or semicolons. The question requires students to assess multiple statements and determine which combination is valid.
        4-6: The question shows strong characteristics of a K-Type question, but the combinations may be less explicit or not as complex.
        7-9: The question has some elements of a K-Type structure, but it is closer to a traditional multiple-choice question with a single correct answer.
        10: The question is not a K-Type question. It is a standard multiple-choice question that requires selecting a single correct answer without evaluating combinations of responses.
    
    Your response should be a single number between 1 and 10, without any additional explanation or rationale.

Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)

    return llm_rating(sysrole, prompt)

#Roberta model from: https://huggingface.co/cointegrated/roberta-large-cola-krishna2020
#In addition to cola, added a query wellformedness score metric from https://huggingface.co/Ashishkr/query_wellformedness_score
# Lazy-loaded: the notebook loads these at import, but no criterion function uses them
# (ambiguous_unclear_information uses only llm_rating). Eager load of roberta-large
# causes a native Windows access-violation crash (0xC0000005) before evaluation can start.
_cola = None
_wellformedness_tokenizer = None
_wellformedness_model = None


def _load_cola_pipeline():
    global _cola
    if _cola is None:
        _cola = pipeline(
            'text-classification',
            model='cointegrated/roberta-large-cola-krishna2020',
            truncation=True,
        )
    return _cola


def _load_wellformedness_models():
    global _wellformedness_tokenizer, _wellformedness_model
    if _wellformedness_tokenizer is None:
        _wellformedness_tokenizer = AutoTokenizer.from_pretrained(
            "Ashishkr/query_wellformedness_score",
            ignore_mismatched_parameters=True,
        )
    if _wellformedness_model is None:
        _wellformedness_model = AutoModelForSequenceClassification.from_pretrained(
            "Ashishkr/query_wellformedness_score"
        )
    return _wellformedness_tokenizer, _wellformedness_model


class _LazyHFAttr:
    """Module-level ``cola`` / ``tokenizer`` / ``model`` names matching the notebook."""

    def __init__(self, kind: str):
        self._kind = kind

    def _resolve(self):
        if self._kind == "cola":
            return _load_cola_pipeline()
        tok, mdl = _load_wellformedness_models()
        return tok if self._kind == "tokenizer" else mdl

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


cola = _LazyHFAttr("cola")
tokenizer = _LazyHFAttr("tokenizer")
model = _LazyHFAttr("model")


def check_string(s):
    # Check for presence of '<' or '>'
    if '<' in s or '>' in s:
        return True

    # Check for two or more ':'
    if len(re.findall(r':', s)) >= 2:
        return True

    # Check for two or more ':'
    if len(re.findall(r'=', s)) >= 2:
        return True
        
    return False

scores_list = []

def ambiguous_unclear_information(question):

    sysrole = """You are an expert educator evaluating the clarity of multiple-choice questions (MCQs) for ambiguity or comprehension issues. Your primary goal is to assess whether a well-prepared student would easily grasp the intended meaning of the question and confidently select the correct answer."""

    prompt= """
Instructions:
    Respond with a score of 1 - 10, where a 1 indicates a high presence of ambiguous or unclear information and 10 indicates an absence of ambiguity and a clear question.

    Scoring Guidelines:
        1–3: The question is severely ambiguous, unclear, or confusing, even for a knowledgeable student.
        4–6: The question has noticeable clarity issues or structural problems that may impede understanding, even when looking at the options.
        7–9: The question is mostly clear and understandable, with only minor or occasional issues that do not significantly hinder comprehension.
        10:  The question is fully clear and understandable, despite any minor imperfections or complexities.

    Focus on major clarity issues that could seriously hinder comprehension.
       - Do not penalize for minor typos, grammatical issues, or stylistic preferences unless they cause confusion.
       - Standard phrasing like "Which of the following..." is acceptable if the options provide enough context.
       - Technical terms or challenging vocabulary are acceptable if they would be understandable to a knowledgeable student.

    If the question depends on the answer options for clarity, this is acceptable as long as the overall meaning is still discernible given the options. In other words, the combination of the stem and options should clearly convey what the question is asking.

    Your response should only be a value in the range of 1 - 10, do not include an explanation or rationale.

Question: {}
Answer: {}
Options: {}""".format(question.stem, question.correct_option, question.options)
    
    return llm_rating(sysrole, prompt)

grat_scores_list = []

def gratuitous_information_in_stem(question):  
    #How effective are lexical richness measures for differentiations of vocabulary proficiency? A comprehensive examination with clustering analysis
    #From: https://github.com/LSYS/LexicalRichness
    stem = LexicalRichness(question.stem)
    
    if stem.cttr > 4.5:
        scores_list.append(1)
        return False
    
    return gratuitous_information_in_stem_verify(question)


def gratuitous_information_in_stem_verify(question):
    sysrole = """You are an expert educator evaluating multiple-choice questions for the presence of gratuitous or unnecessary information in the question's text that could confuse students. Your focus is on identifying significant issues where extraneous details might distract or mislead a student, not on minor additional information that doesn't impact understanding."""
    prompt = """
Instructions:
    Respond with a score of 1 - 10, where 1 means the question contains significant gratuitous information that could confuse students, and 10 means the question is concise and contains only relevant information necessary for understanding and answering the question correctly.

    Scoring Guidelines:
        1–3: The question contains excessive or irrelevant details that significantly distract or confuse students.
        4–6: The question contains noticeable extraneous information that may hamper clarity but does not severely mislead.
        7–9: The question is mostly concise, with minor additional details that do not distract from the main point.
        10: The question is entirely focused on essential information needed to answer correctly, with no unnecessary details.

    Do not assign a low score (1–3 or 4–6) for minor details that are typical in multiple-choice questions and do not affect comprehension.  
    
    Do not assign a low score if the question is scenario-based and the details meaningfully contribute to the scenario.

    Your response should be **only** a single integer from 1–10. Do not include any explanation or rationale.

Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)

    return llm_rating(sysrole, prompt)

#Check for synonyms, because they'll know it's the word they've most recently come across in the text
#The correct option is likely to be used more (when in pairs, etc.) --> k-type (super similar by description)
def avoid_convergence_cues(question):
    
    # Copy options to avoid modifying the original question object
    options = question.options.copy()
    
    # Split the correct answers (if multiple) and remove them from options
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)
    
    # Ensure there are enough distractors for meaningful analysis
    if len(options) < 2:  
        return True  # Not enough distractors to compare
    
    # Get lemmas (base forms) of nouns in the correct answer(s)
    lemma_nouns_answ = set()
    for answer in correct_answers:
        lemma_nouns_answ.update(get_lemma_nouns(answer))
    
    # Get lemmas for all distractors
    lemma_nouns_options = [set(get_lemma_nouns(opt)) for opt in options]
    
    # Collect synonyms of the correct answer(s)
    synonyms = set()
    for noun in lemma_nouns_answ:
        for syn in wn.synsets(noun):
            synonyms.update(l.name().lower().replace('_', ' ') for l in syn.lemmas())

    # Check if any synonym appears in only some of the distractors
    for opt in lemma_nouns_options:
        repeating_nouns_synonyms = synonyms.intersection(opt)
        if repeating_nouns_synonyms:
            for rns in repeating_nouns_synonyms:
                # Ensure the term is not present in every distractor (i.e., it's unevenly distributed)
                if not all(rns in value for value in lemma_nouns_options):
                    return avoid_convergence_cues_verify(question)
                    
    return True

def avoid_convergence_cues_verify(question):
    sysrole = """You are an expert in educational assessment specializing in detecting convergence cues in multiple-choice questions. Convergence cues are present when the options share overlapping elements, refer to each other, or include combinations of other options (e.g., "A and B," "B or C"). Your goal is to identify problematic convergence cues that are based on logical overlap between options (e.g., subsets, combinations, or dependencies). However, you should not flag distinct, independent numerical values or factual answers (like scientific measurements) as convergence cues unless they are structured to confuse students by implying a relationship between the options."""
    prompt = """
Instructions:
    Respond with a score of 1 - 10, where a 1 indicates a high presence of convergence cues and 10 indicates an absence of convergence cues.

    Scoring Guidelines:  
        1-3: The options are heavily interrelated, sharing combinations or subsets that could confuse students (e.g., “A and B”, “B and C”, “A or C”).  
        4-6: There is moderate convergence, with some overlap in the options but not enough to significantly confuse students.  
        7-9: Minimal convergence cues; the options are distinct and clearly differentiated.  
        10: No convergence cues detected; each option is unique and stands on its own without reference to other options.

    Your response should only be a value in the range of 1 - 10, do not include an explanation or rationale.
    
Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)

    return llm_rating(sysrole, prompt)

#If verb exists in answer choice, ensure it's the same tense as verb in other options
#We want the stem to be the same, but as long as all the answers are the same, then it's fine, to avoid false positive.
#https://huggingface.co/Unbabel/gec-t5_small
def grammatical_cues_in_stem(question):
    answer_tense = get_verb_tense(question.correct_option)

    options = question.options.copy()

    # Split the correct answers (if multiple) and remove them from options
    correct_answers = [opt.strip() for opt in question.correct_option.split('[SEP]')]
    for opt in correct_answers:
        if opt in options:
            options.remove(opt)
    
    for opt in options:
        opt_tense = get_verb_tense(opt)    
        if opt_tense != 'none' and answer_tense != 'none' and answer_tense is not opt_tense:
            return False
            
    return grammatical_cues_verify(question)

#Longer options might contain verbs of different tenses.
#We want options that specifically have a single tense (past or present) and for it to be consistent with all other options.
def get_verb_tense(text):
    verbs = []
    doc = nlp(text)
    for token in doc:
        if token.pos_ == 'VERB':
            if token.tag_ in ['VBP', 'VBZ']:
                verbs.append('present')
            elif token.tag_ in ['VBD', 'VBN']:
                verbs.append('past')
            else:
                verbs.append('none')

    verb_tenses = list(set(verbs))
    if len(verb_tenses) == 1 and verb_tenses[0] == 'past':
        return 'past'
    elif len(verb_tenses) == 1 and verb_tenses[0] == 'present':
        return 'present'
    return 'none'

def grammatical_cues_verify(question):
    sysrole = """You are an expert educator evaluating multiple-choice questions for any grammatical cues in the stem and answer choices that could inadvertently reveal the correct answer or mislead students. This includes mismatched tenses, pronoun inconsistencies, lack of parallelism in structure, or other errors that a testwise student might exploit to identify or eliminate certain options without relying on content knowledge."""
    prompt = """
Instructions:
    Respond with a score from 1–10 based on the overall grammatical consistency and parallelism of the question and its options. Use the following Scoring Guidelines:

    Scoring Guidelines:
        1–3: The question and/or options contain noticeable grammatical or structural inconsistencies that strongly hint at the correct answer.
        4–6: Some grammatical mismatches or parallelism issues exist, but these are not glaring enough to immediately reveal the correct option.
        7–9: The question and options are mostly consistent, with only minor discrepancies that are unlikely to provide a significant clue.
        10: The question and options are grammatically consistent, parallel in style, and contain no identifiable cues that would give an advantage to a testwise student.

    Do not lower the score for trivial wording differences that are typical and do not compromise fairness.
    
    Only penalize clear grammatical or structural errors that could be used to identify the correct option.

    Your response should be only a single integer (1–10). Do not provide any explanations.

Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)
    
    #It's so strict on this, we're setting the threshold to 4 instead of 5.
    #We get better results when we just reutnr True instead of verify, but the LLM could be sided with in these cases.
    return llm_rating(sysrole, prompt,4)

#Like the ohter criteria that use a list of terms, these can be modified
def vague_terms(question):
    vagues = ["frequently", "occasionally", "rarely", "seldom", "sometimes", "usually", "regularly", "periodically", "infrequently", "generally", "nearly", "more or less", "somewhat", "partly"]
    
    #check the options then check the stem
    for opt in question.options:
        opt = opt.lower()
        if any(word in opt for word in vagues):
            return False

    #In particular, these words can sometimes be used in the stem in a way that is not a flaw, but more likely than not, it is
    if any(word in question.stem.lower() for word in vagues):
        return False
    
    return True

def unfocused_stem(question):   
    if not true_or_false(question) or not all_of_the_above(question) or not none_of_the_above(question) or not fill_in_the_blank(question):
        return True
        
    #Traits of an unfocused question (not being a question, etc.)
    if '?' not in question.stem and ":" not in question.stem:
        if not question.stem.endswith(('.', ':', '?', ';')):
            return False

        if not check_if_first_word_is_a_verb(question.stem):
            return False
        
        contains_question = False
        doc = nlp(question.stem)
        for sent in doc.sents:
            if is_question(sent.text.strip()):
                contains_question = True
                break
                
        return contains_question

    return unfocused_stem_verify(question)

def check_if_first_word_is_a_verb(sent):
    d = nlp(sent)
    token = d[0] # gets the first token in a sentence
    if token.pos_ == "VERB" and token.dep_ == "ROOT": # checks if the first token is a verb and root or not
        return True
    return False


#From https://stackoverflow.com/questions/4083060/determine-if-a-sentence-is-an-inquiry
def is_question(sent):
    d = nlp(sent)
    token = d[0] # gets the first token in a sentence
    if token.pos_ == "VERB" and token.dep_ == "ROOT": # checks if the first token is a verb and root or not
        return True
    for token in d: # loops through the sentence and checks for WH tokens
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB" or token.text == '?':
            return True
    return  False


#Teachers should avoid using MCQs with unfocused stems which do not ask a clear question or state a clear problem in the sentence completion format
#The stem should present a clear and focused question that can be understood and answered without looking at the options
def unfocused_stem_verify(question):
    sysrole = """You are an expert, but lenient, educator evaluating multiple-choice questions for the presence of an unfocused stem. A question's stem is considered unfocused if it is not a clear query that can be understood and answered on its own. Your task is to assess how clear and self-contained the stem is, without relying on the answer choices for clarity."""
    prompt = """
Instructions:
    Provide a single numeric score from 1 to 10, reflecting how focused and self-contained the question’s stem is.
    It should ask a clear question or state a clear problem in a sentence completion format.
    Ignore questions with options that are similar to "yes"/"no" or "true"/"false" followed by an explanation.

    Scoring Guidelines:
        1–3: The stem is extremely unfocused or unclear, making it difficult for students to understand what is being asked, even after re-reading.
        4–6: The stem has noticeable clarity issues; students might need to look at the options for guidance or context.
        7–9: The stem is generally clear, with only minor ambiguities; students can understand the question but could benefit from the options for further context.
        10: The stem is entirely clear and focused; students can fully understand and answer the question without looking at the options.
    
    Your response should only be a single number from 1 to 10, with no additional explanation.

Question: {}
Options: {}
Answer: {}""".format(question.stem, question.options, question.correct_option)

    #It's so strict on this, we're setting the threshold to 4 instead of 5.
    return llm_rating(sysrole, prompt, 4)

all_criteria = [
    'ambiguous_unclear_information',
    'implausible_distractors',
    'none_of_the_above',
    'longest_answer_correct',
    'gratuitous_information_in_stem',
    'true_or_false',
    'avoid_convergence_cues',
    'avoid_logical_cues',
    'all_of_the_above',
    'fill_in_the_blank',
    'absolute_terms',
    'word_repeats_in_stem_and_correct_answer',
    'unfocused_stem',
    'complex_k_type',
    'grammatical_cues_in_stem',
    'lost_sequence',
    'vague_terms',
    'more_than_one_correct',
    'negative_worded_stem'
]

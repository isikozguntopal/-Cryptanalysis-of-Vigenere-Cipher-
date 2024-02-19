# %%
import collections
from random import randint
from random import seed
import copy
import math

# %%
seed(42)

# %%
fitness_iter = 0


# %%
meaningful_keys = [
    "apple",# 5 letters
    "tomato",# 6 letters
    "kingdom",# 7 letters
    "elephant",# 8 letters
    "butterfly",# 9 letters
    "strawberry",# 10 letters
    "keyboardist",# 11 letters
    "refrigerator",# 12 letters
    "cabinetmakers",# 13 letters
    "thermodynamics",# 14 letters
    "ultracentrifuge",# 15 letters
]

# %%
avg_5000 = [0.9513805522208884,
 0.9323729491796718,
 0.9231692677070829,
 0.7882486327864479,
 0.7591036414565826,
 0.6813058556756036,
 0.5941376550620246,
 0.5928037881819394,
 0.5481392557022811,
 0.40320128051220494,
 0.4389289048952915,
 0.44296480496960705,
 0.35922654776196183,
 0.35655881400179085,
 0.29964700165780583,
 0.2990891594733132,
 0.26643848015396593,
 0.24166222044373298,
 0.24700610402891274,
 0.21458631071476195]

avg_2500 = [0.8547547547547548,
 0.8062062062062062,
 0.8038038038038039,
 0.5906239572906239,
 0.517600934267601,
 0.421221221221221,
 0.33254921588254915,
 0.33777777777777757,
 0.294404404404404,
 0.18108775442108763,
 0.2327360694027358,
 0.22883931550598172,
 0.17736760570093776,
 0.1865551265551252,
 0.14531340864674105,
 0.15296010296010246,
 0.13025724136835173,
 0.11716295660740077,
 0.1205489905489901,
 0.10161200883423105]

avg_1000 = [0.6839301449971962,
 0.6189617880317231,
 0.6081070255547544,
 0.36340089187962293,
 0.26439958343346964,
 0.19889716681353317,
 0.16377206333947505,
 0.14419610670511931,
 0.12161606451440622,
 0.07142780848620778,
 0.09988918262169849,
 0.10409623754973436,
 0.07522344674471472,
 0.08392394609049149,
 0.05957111195034698,
 0.06841380467912554,
 0.051834939962793224,
 0.05147766041181068,
 0.05137342341139517,
 0.04015515068747468]

avg_750 = [0.6333513578466715,
 0.5742710085716576,
 0.561944244172074,
 0.32314447648802375,
 0.21915805495473858,
 0.15896118721461205,
 0.1314337498998639,
 0.11719838981014256,
 0.09451754385964918,
 0.05487520170288716,
 0.0813331845595723,
 0.08418134377038522,
 0.059671152767763525,
 0.06829860038223336,
 0.04659539354474409,
 0.05380116959064316,
 0.039373548025313995,
 0.04099964046341098,
 0.04011121792304233,
 0.03191190018425059]

avg_500 = [0.5802988422865841,
 0.5062826316281429,
 0.5196490806393462,
 0.281523321182016,
 0.17602718690328367,
 0.12063520677269024,
 0.10597951635086604,
 0.08556864158955342,
 0.06909092123008735,
 0.039029096395999205,
 0.06092683838748016,
 0.061275258811613054,
 0.04259494528092019,
 0.05021975380706254,
 0.03397294662767517,
 0.04064688755230702,
 0.028584749874258103,
 0.02999987600696638,
 0.02967978322202623,
 0.02285940912608378]

avg_400 = [0.5582155561965195,
 0.48266726489974254,
 0.49360638400153833,
 0.26183379803224044,
 0.15347616148019969,
 0.1087876165753293,
 0.0948808341078316,
 0.07661175741648822,
 0.06035423944278893,
 0.033744618573000235,
 0.05365456312961366,
 0.05354132615453697,
 0.0367375771255204,
 0.04322478149994723,
 0.029483161722849805,
 0.03546816316501853,
 0.024626867379967925,
 0.02657932768979776,
 0.02631439589905187,
 0.02033396005369852]

avg_300 = [0.5283386212864148,
 0.45400282024164346,
 0.47007499278915493,
 0.24663894497323968,
 0.13856440085889185,
 0.09094518155305571,
 0.08180102874723592,
 0.06480867224305344,
 0.05055122904848911,
 0.029606127615934175,
 0.04481328489476766,
 0.046016236991865,
 0.030573353752615263,
 0.036212749689819364,
 0.02486665659437483,
 0.02923688656310429,
 0.019571530820449143,
 0.022039681013577714,
 0.021241986042246336,
 0.017043844331484908]

avg_200 = [0.5019228920296125,
 0.42522353619844244,
 0.44760119219305833,
 0.22365904987768262,
 0.11836468715614952,
 0.0774129624288261,
 0.07042485231120948,
 0.05311535215630908,
 0.03936987896462947,
 0.02169235864072891,
 0.0363980951160072,
 0.036506677471489106,
 0.024656130439228447,
 0.028837867390651407,
 0.01937395112089844,
 0.023962299616082042,
 0.015934924857144057,
 0.0180699836045476,
 0.017696418588160137,
 0.013812507344379244]

avg_150 = [0.48384169791366216,
 0.4080196615710028,
 0.4296101336409961,
 0.2108881758164279,
 0.10923328686344254,
 0.06695810498990451,
 0.06119503733615363,
 0.045211798545010694,
 0.03396528378681524,
 0.019674790885491882,
 0.03228209512821615,
 0.03260240544636317,
 0.02225800461036305,
 0.024898475879151644,
 0.017225750385723063,
 0.019905676232291128,
 0.013165186154261946,
 0.015359524534423625,
 0.014288575732187109,
 0.011505437511350379]

avg_100 = [0.4660689677274621,
 0.3864331955260712,
 0.40436015767714645,
 0.18903564294031563,
 0.09703327137347888,
 0.059168322490358895,
 0.05707063423388773,
 0.039402728370135066,
 0.027675357070366946,
 0.015339069961221833,
 0.02544071998571608,
 0.026050417923795827,
 0.017008390142584488,
 0.01946614374686292,
 0.01250617118622608,
 0.015227731078971555,
 0.009559167464056608,
 0.011750199283849953,
 0.010981214158793171,
 0.009002969036859982]

# %%
avg_list_total = [
    {"text_len" : 5000, "avg_list" : avg_5000},
    {"text_len" : 2500, "avg_list" : avg_2500},
    {"text_len" : 1000, "avg_list" : avg_1000},
    {"text_len" : 750, "avg_list" : avg_750},
    {"text_len" : 500, "avg_list" : avg_500},
    {"text_len" : 400, "avg_list" : avg_400},
    {"text_len" : 300, "avg_list" : avg_300},
    {"text_len" : 200, "avg_list" : avg_200},
    {"text_len" : 150, "avg_list" : avg_150},
    {"text_len" : 100, "avg_list" : avg_100},
]

# %%
def find_key_conf(text_len, key_len, avg_list_total):

    closest = min(avg_list_total, key=lambda x: abs(x['text_len'] - text_len))
    closest_avg_list = closest['avg_list']

    return closest_avg_list[key_len - 5]


# %%
### Expected Values
IC_expected = 0.067
monogram_expected = [
    ["a", 8.55], ["k", 0.81], ["u", 2.68],
    ["b", 1.60], ["l", 4.21], ["v", 1.06],
    ["c", 3.16], ["m", 2.53], ["w", 1.83],
    ["d", 3.87], ["n", 7.17], ["x", 0.19],
    ["e", 12.10], ["o", 7.47], ["y", 1.72],
    ["f", 2.18], ["p", 2.07], ["z", 0.11],
    ["g", 2.09], ["q", 0.10],
    ["h", 4.96], ["r", 6.33],
    ["i", 7.33], ["s", 6.73],
    ["j", 0.22], ["t", 8.94]
]
bigram_expected = [
    ["th", 2.71], ["en", 1.13], ["ng", 0.89],
    ["he", 2.33], ["at", 1.12], ["al", 0.88],
    ["in", 2.03], ["ed", 1.08], ["it", 0.88],
    ["er", 1.78], ["nd", 1.07], ["as", 0.87],
    ["an", 1.61], ["to", 1.07], ["is", 0.86],
    ["re", 1.41], ["or", 1.06], ["ha", 0.83],
    ["es", 1.32], ["ea", 1.00], ["et", 0.76],
    ["on", 1.32], ["ti", 0.99], ["se", 0.73],
    ["st", 1.25], ["ar", 0.98], ["ou", 0.72],
    ["nt", 1.17], ["te", 0.98], ["of", 0.71]
]
trigram_expected = [
    ["the", 1.81], ["ere", 0.31], ["hes", 0.24],
    ["and", 0.73], ["tio", 0.31], ["ver", 0.24],
    ["ing", 0.72], ["ter", 0.30], ["his", 0.24],
    ["ent", 0.42], ["est", 0.28], ["oft", 0.22],
    ["ion", 0.42], ["ers", 0.28], ["ith", 0.21],
    ["her", 0.36], ["ati", 0.26], ["fth", 0.21],
    ["for", 0.34], ["hat", 0.26], ["sth", 0.21],
    ["tha", 0.33], ["ate", 0.25], ["oth", 0.21],
    ["nth", 0.33], ["all", 0.25], ["res", 0.21],
    ["int", 0.32], ["eth", 0.24], ["ont", 0.20]
]

# %%
def n_gram_ratio_converter(n_gram_list, text_len):
    n_gram_temp = []
    coeff = text_len / 100

    for i in range(len(n_gram_list)):
        temp_n_gram_element = [n_gram_list[i][0], (n_gram_list[i][1] * coeff)]
        n_gram_temp.append(temp_n_gram_element)
    
    return n_gram_temp

# %%
def n_gram_analysis(cipher_text):
    # MONOGRAM
    letter_count = collections.Counter(cipher_text).most_common()
    letter_count_dict = dict(letter_count)

    # BIGRAM
    bigrams_in_text = [cipher_text[i : i+2] for i in range(len(cipher_text) - 1)]
    bigram_count = collections.Counter(bigrams_in_text).most_common(30)
    bigram_count_dict = dict(bigram_count)

    # TRIGRAM
    trigrams_in_text = [cipher_text[i : i+3] for i in range(len(cipher_text) - 2)]
    trigram_count = collections.Counter(trigrams_in_text).most_common(30)
    trigram_count_dict = dict(trigram_count)

    return letter_count_dict, bigram_count_dict, trigram_count_dict
    

# %%
def index_of_coincidence(text):
    total_letters = len(text)

    letter_count = {}
    for letter in text:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1
            
    ioc = 0.0
    for letter, count in letter_count.items():
        ioc += (count * (count - 1)) / (total_letters * (total_letters - 1))
    
    return ioc


# %%
def find_most_frequent(cipher_text, key_len):
    frequency_list = [[0] * 26 for _ in range(key_len)]
    cipher_list = list(cipher_text)

    for i in range(0, len(cipher_list)):
        index_of_frequency = i % key_len
        temp_char_index = (ord(cipher_list[i]) - 97)
        frequency_list[index_of_frequency][temp_char_index] += 1
    
    return frequency_list

# %%
def find_max_values(list_of_freq):
    max_value_list = [chr((sublist.index(max(sublist))) + 97) for sublist in list_of_freq]
    return max_value_list

# %%
def find_duplicates_indices(lst):
    indices = {}
    for index, element in enumerate(lst):
        if element not in indices:
            indices[element] = []
        indices[element].append(index)

    return [idxs for idxs in indices.values() if len(idxs) > 1]


# %%
def key_rep_index_wrapper(ciphered_text, key_len):
    most_freq_list = find_most_frequent(ciphered_text, key_len)
    max_values_of_most_freq = find_max_values(most_freq_list)
    return find_duplicates_indices(max_values_of_most_freq)

# %%
def ascii_converter_to_string(number_asc):
    result = ""
    for i in range(len(number_asc)):
        result += (chr(number_asc[i]))
    return result

# %%
def decipher(keys, ciphered):
    result = ""
    for j in range(len(ciphered)):
        index = j % (len(keys))
        cipher_key = keys[index]
        temp = ord(ciphered[j])
        temp = temp - 97
        result += chr((((temp + 26) - (cipher_key - 97)) % 26)+ 97)

    return result

# %%
def fitness_function(test_key, cipher_text, monogram, bigram, trigram, IC_ex):
    
    deciphered_text = decipher(test_key, cipher_text)
    mono_pred, bi_pred, tri_pred = n_gram_analysis(deciphered_text)
    IC_pred = (index_of_coincidence(deciphered_text) * 1000) - (IC_ex * 1000)
        
    monogram_diff = {key: abs(monogram[key] - mono_pred.get(key, 0)) for key in monogram}
    bigram_diff = {key: bigram[key] - bi_pred.get(key, 0) for key in bigram}
    trigram_diff = {key: trigram[key] - tri_pred.get(key, 0) for key in trigram}

    particle_score = sum([sum(monogram_diff.values()), sum(bigram_diff.values()), sum(trigram_diff.values()), IC_pred])

    global fitness_iter
    fitness_iter += 1
    return particle_score

# %%
# Create Population
def create_population(pop_amount, key_len):
    new_pop = []
    for i in range(pop_amount):
        temp_indi = {"key" : [], "score" : 0}
        for j in range(key_len):
            temp_indi["key"].append(randint(97, 122))

        new_pop.append(temp_indi)
    return new_pop

# %%
# Mutation Func
def mutation_func(parent_key_1, parent_key_2, rep_flag, rep_index):
    rand_start = randint(0, (len(parent_key_1) - 1))
    rand_end = randint(rand_start, (len(parent_key_1) - 1))


    child_key_1 = copy.deepcopy(parent_key_1)
    child_key_2 = copy.deepcopy(parent_key_2)

    swap_list = [i for i in range(rand_start, rand_end)]

    if (rep_flag == True):
        for index in swap_list:
            is_index_in_rep = next((sublist for sublist in rep_index if index in sublist), None)
            if (is_index_in_rep != None):
                temp_rep_holder_1 = child_key_1[index]
                temp_rep_holder_2 = child_key_2[index]
                for i in is_index_in_rep:
                    child_key_1[i] = temp_rep_holder_2
                    child_key_2[i] = temp_rep_holder_1
            else:
                temp_key_holder = child_key_1[index]
                child_key_1[index] = child_key_2[index]
                child_key_2[index] = temp_key_holder
        
        return child_key_1, child_key_2

    for index in swap_list:
        temp_key_holder = child_key_1[index]
        child_key_1[index] = child_key_2[index]
        child_key_2[index] = temp_key_holder
    
    return child_key_1, child_key_2

# %%
def crossover_func(parent_key_1, parent_key_2, rep_flag, rep_index):
    rand_idx_1 = randint(0, (len(parent_key_1) - 1))
    rand_idx_2 = randint(0, (len(parent_key_1) - 1))

    
    child_key_1 = copy.deepcopy(parent_key_1)
    child_key_2 = copy.deepcopy(parent_key_2)

    # First Swap
    temp_key_holder = child_key_1[rand_idx_1]
    child_key_1[rand_idx_1] = child_key_2[rand_idx_2]
    child_key_2[rand_idx_2] = temp_key_holder

    # Second Swap
    temp_key_holder = child_key_2[rand_idx_1]
    child_key_2[rand_idx_1] = child_key_1[rand_idx_2]
    child_key_1[rand_idx_2] = temp_key_holder

    if (rep_flag == True):
        for value in [rand_idx_1, rand_idx_2]:
            is_index_in_rep = next((sublist for sublist in rep_index if value in sublist), None)
            if (is_index_in_rep != None):
                temp_rep_holder_1 = child_key_1[value]
                temp_rep_holder_2 = child_key_2[value]
                for i in is_index_in_rep:
                    child_key_1[i] = temp_rep_holder_1
                    child_key_2[i] = temp_rep_holder_2
    
    return child_key_1, child_key_2

# %%
def mutation_gene_insert_func(parent_key_1, parent_key_2, rep_flag, rep_index):
    child_key_1 = copy.deepcopy(parent_key_1)
    child_key_2 = copy.deepcopy(parent_key_2)

    change_limit = randint(0, len(parent_key_1)-1)

    for _ in range(0, change_limit):
        rand_index = randint(0, (len(parent_key_1) - 1))
        rand_letter_1 = (randint(0, 25) + 97)
        rand_letter_2 = (randint(0, 25) + 97)

        if (rep_flag == True):
            is_index_in_rep = next((sublist for sublist in rep_index if rand_index in sublist), None)
            if (is_index_in_rep != None):
                for i in is_index_in_rep:
                    child_key_1[i] = rand_letter_1
                    child_key_2[i] = rand_letter_2
            else:
                child_key_1[rand_index] = rand_letter_1
                child_key_2[rand_index] = rand_letter_2

        else:
            child_key_1[rand_index] = rand_letter_1
            child_key_2[rand_index] = rand_letter_2
    
        
    return child_key_1, child_key_2

# %%
# Child Maker
def create_new_indi(parent_1, parent_2, key_confidence, rep_index, ciphered_text_param, monogram_param, bigram_param, trigram_param, IC_param):
    threshold_value = randint(0, 100)
    rep_flag = False

    parent_1_key = copy.deepcopy(parent_1["key"])
    parent_2_key = copy.deepcopy(parent_2["key"])

    if (threshold_value > key_confidence * 100):
        rep_flag = False
    
    new_key_1, new_key_2 = mutation_func(parent_1_key, parent_2_key, rep_flag, rep_index)
    new_key_3, new_key_4 = crossover_func(parent_1_key, parent_2_key, rep_flag, rep_index)
    new_key_5, new_key_6 = mutation_gene_insert_func(parent_1_key, parent_2_key, rep_flag, rep_index)

    new_key_list = [new_key_1, new_key_2, new_key_3, new_key_4, new_key_5, new_key_6]

    new_indi_list = []
    new_indi_list.append({"key" : copy.deepcopy(parent_1_key), "score": parent_1["score"]})
    new_indi_list.append({"key" : copy.deepcopy(parent_2_key), "score": parent_2["score"]})

    for key_to_test in new_key_list:
        key_score = fitness_function(
            key_to_test,
            ciphered_text_param,
            monogram_param,
            bigram_param,
            trigram_param,
            IC_param
        )
        new_indi_list.append({"key" : copy.deepcopy(key_to_test), "score": key_score})

    sorted_new_indi = sorted(new_indi_list, key=lambda x: x['score'])
    best_two = sorted_new_indi[:2]

    return best_two

# %%
def compare_strings(correct, guessed):
    if len(correct) != len(guessed):
        raise ValueError("Both strings must be of the same length.")

    match_count = sum(c1 == c2 for c1, c2 in zip(correct, guessed))
    corr_ratio = f"{match_count}/{len(correct)}"
    return corr_ratio

# %%
def loop_manager(key_arg, ciphered_text):
    monogram_expected_tuned_dict = dict(n_gram_ratio_converter(monogram_expected, len(ciphered_text)))
    bigram_expected_tuned_dict = dict(n_gram_ratio_converter(bigram_expected, len(ciphered_text)))
    trigram_expected_tuned_dict = dict(n_gram_ratio_converter(trigram_expected, len(ciphered_text)))

    key_len =  key_arg[0]
    key_rep_conf = find_key_conf(len(ciphered_text), key_len, avg_list_total) * 100
    index_of_freq_rep = key_rep_index_wrapper(ciphered_text,  key_len)

    pop_list = [[], []]
    pop_size = 75
    pop_list[0] = create_population(pop_size, key_len)

    for indi in pop_list[0]:
        indi["score"] = fitness_function(indi["key"],
                                    ciphered_text,
                                    monogram_expected_tuned_dict,
                                    bigram_expected_tuned_dict,
                                    trigram_expected_tuned_dict,
                                    IC_expected)

    iter = 0
    limit = ((key_len**2 / math.sqrt(len(ciphered_text))) * 10) + 5
    
    while True:
        print("KEY LEN : ", key_len)
        print("ITER : ", iter)
        
        if (iter >= limit):
            key_len += 1
            if (key_len == key_arg[1]+1):
                min = float("inf")
                best_child = {}
                for item in pop_list[new_pop_location]:
                    if (item["score"] < min):
                        min == item["score"]
                        best_child = item
                
                true_key = meaningful_keys[len(best_child["key"]) - 5]
                guessed_key = ascii_converter_to_string(best_child["key"])
                return [guessed_key, compare_strings(true_key, guessed_key), len(best_child["key"]), best_child["score"], len(ciphered_text), fitness_iter, pop_size]

            iter = 0

            pop_list = [[], []]
            pop_list[0] = create_population(pop_size, key_len)
            for indi in pop_list[0]:
                indi["score"] = fitness_function(indi["key"],
                                    ciphered_text,
                                    monogram_expected_tuned_dict,
                                    bigram_expected_tuned_dict,
                                    trigram_expected_tuned_dict,
                                    IC_expected)

            key_rep_conf = find_key_conf(len(ciphered_text), key_len, avg_list_total) * 100
            index_of_freq_rep = key_rep_index_wrapper(ciphered_text,  key_len)

            limit = ((key_len**2 / math.sqrt(len(ciphered_text))) * 10) + 5
        
        else:
            new_pop_location = (iter+1) % 2
            cur_pop = (iter % 2)
            while (len(pop_list[new_pop_location]) < pop_size):
                rand_parent_1 = randint(0, pop_size-1)
                rand_parent_2 = randint(0, pop_size-1)

                parent_1 = pop_list[cur_pop][rand_parent_1]
                parent_2 = pop_list[cur_pop][rand_parent_2]


                new_pop_indi = create_new_indi(parent_1, parent_2, key_rep_conf, index_of_freq_rep,
                                            ciphered_text,
                                            monogram_expected_tuned_dict,
                                            bigram_expected_tuned_dict,
                                            trigram_expected_tuned_dict,
                                            IC_expected)

                pop_list[new_pop_location].append(new_pop_indi[0])
                pop_list[new_pop_location].append(new_pop_indi[1])
            
            # END OF LOOP
            pop_list[cur_pop] = []
            iter += 1
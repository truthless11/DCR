import re
import json

# since the reference numbers have nothing to do with context, so we just treat them as symbols
# since the taxi type and taxi phone number is random generated, and have no relation with the user's requests. (in this dataset, users never require certain color or type of taxies)
# there is only one police station, so once the type of this slot is predicted, we get the true value.
# there is only one hospital, so once the type of this slot is predicted, we get the true value
symbol_list = {
    "[hotel_reference]",
    "[train_reference]",
    "[restaurant_reference]",
    "[attraction_reference]",
    "[hospital_reference]",
    "[taxi_type]",
    "[taxi_phone]",
    "[police_address]",
    "[police_phone]",
    "[police_postcode]",
    "[police_name]",
    "[hospital_postcode]",
    "[hospital_address]",
    "[hospital_name]",
    "[value_time]",
    "[value_price]",
    "[value_place]",
    "[value_day]",
    "[train_id]",
    "[train_trainID]",
    "[value_count]"
}

# all the values in "entity_attr_list" almost only appear in system's utterence, so we remove them from user's utterence
# for which we only predict the value's type and corresponding entity, and then query from the database
entity_attr_list = {
    "[attraction_address]",
    "[restaurant_address]",
    "[attraction_phone]",
    "[restaurant_phone]",
    "[hotel_address]",
    "[restaurant_postcode]",
    "[attraction_postcode]",
    "[hotel_phone]",
    "[hotel_postcode]",
    "[hospital_phone]"
}

# the entities which should be queried from KG, it contains both relations and entities. Here we treat these relations (in fact the attributes of entities) also as entities.
entity_type_list = {
    # attr values (relations)
    "[value_area]",
    "[value_pricerange]",
    "[value_food]",
    # entity names
    "[hotel_name]",
    "[restaurant_name]",
    "[attraction_name]",
    "[hospital_department]"
}

def stat_enttype(texts):
    enttype_stat = {}
    for text in texts:
        words = text.split()
        for word in words:
            if "::" in word:
                ent_type = word.split("::")[0]
                if ent_type not in enttype_stat:
                    enttype_stat[ent_type] = 1
                else: 
                    enttype_stat[ent_type] += 1

    return enttype_stat


def generate_data(attr_ent_map, entity_name_map):
    delex_data = json.load(open("./delex.json"))

    ENTITY_TOK = "<$>"

    attr_value_map = {}
    preprop_res = {}
    for id_, res in delex_data.items():
        user_utt = []
        user_ent = []
        sys_utt = []
        sys_ent = []
        for idx, each in enumerate(res["log"]):
            text = each["text"]

            words = text.split()
            entities = [""] * len(words)
            for word_idx, word in enumerate(words):
                if "::" in word:
                    attr_val = word.split("::")
                    if len(attr_val) != 2:
                        #print(id_, text, word)
                        exit()
                    else:
                        attr, val = attr_val

                        # to count the number of every "attr"
                        if attr not in attr_value_map:
                            attr_value_map[attr] = {val}
                        else:
                            attr_value_map[attr].add(val)

                        if attr in symbol_list:
                            words[word_idx] = attr
                        else:
                            if word in entity_name_map: 
                                #print(word)
                                words[word_idx] = ENTITY_TOK
                                entities[word_idx] = entity_name_map[word] 
                            elif attr in entity_type_list:
                                words[word_idx] = ENTITY_TOK
                                entities[word_idx] = "::".join([attr, val])
                            elif attr in entity_attr_list:
                                #value = "::".join([attr, val])
                                try:
                                    corres_ent = attr_ent_map[attr[1:-1]][val]
                                except:
                                    if attr == "[hospital_phone]":
                                        #print("hospital_phone does not find: ", attr, val)
                                        words[word_idx] = "[hospital_central_phone]"
                                        continue
                                    print("attr_ent_map dont find res:")
                                    print(attr, val)
                                    exit()
                                entities[word_idx] = corres_ent
                            else:
                                print("find non-existing attr-val pair:")
                                print(attr_val)
                                exit()

            if idx % 2 == 0:
                user_utt.append(words)
                user_ent.append(entities)
            else:
                sys_utt.append(words)
                sys_ent.append(entities)

        preprop_res[id_] = {"user_utt": user_utt, "user_ent": user_ent, "sys_utt": sys_utt, "sys_ent": sys_ent}

    for attr, vals in attr_value_map.items():
        attr_value_map[attr] = list(vals)

    json.dump(attr_value_map, open("./attr_value_map.json", "w"))
    json.dump(preprop_res, open("./preprocess_utt_ent.json", "w"))


replacements = []
for line in open('mapping.pair'):
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


# this normalize function is borrowed from multivoz
def normalize(text):
    timepat = re.compile("(\d{1,2}[:]\d{1,2})")
    pricepat = re.compile("(\d{1,3}[.]\d{1,2})")

    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    text = re.sub(timepat, ' [value_time] ', text)
    text = re.sub(pricepat, ' [value_price] ', text)
    #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)
    
    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    text = text.replace(" ", "_")

    return text


def cvt_address(val):
    if "road" in val:
        val = val.replace("road", "rd")
    elif "rd" in val:
        val = val.replace("rd", "road")
    elif "st" in val:
        val = val.replace("st", "street")
    elif "street" in val:
        val = val.replace("street", "st")

    return val


def cvt_name(val):
    if "b & b" in val:
        val = val.replace("b & b", "bed and breakfast")
    elif "bed and breakfast" in val:
        val = val.replace("bed and breakfast", "b & b")
    elif "hotel" in val and 'gonville' not in val:
        val = val.replace("hotel", "")
    elif "restaurant" in val:
        val = val.replace("restaurant", "")
    val = normalize(val)

    return val


def generate_info(input_path):
    entity_info = {}
    attr_ent_map = {
        "attraction_address": {},
        "attraction_phone": {},
        "attraction_postcode": {},
        "restaurant_address": {},
        "restaurant_phone": {},
        "restaurant_postcode": {},
        "hotel_phone": {},
        "hotel_postcode": {},
        "hotel_address": {},
        "hospital_phone":{}
    }

    # multiple entity names may correspond to one same entity
    entity_name_map = {}
     
    attraction_db = json.load(open(input_path + "/attraction_db.json"))
    for each in attraction_db:
        name = normalize(each["name"])

        new_name = cvt_name(each["name"])
        new_name = "[attraction_name]::" + new_name
        if new_name != name and new_name not in entity_name_map:
            entity_name_map[new_name] = "[attraction_name]::" + name

        name = "[attraction_name]::" + name
        address = normalize(each["address"])
        postcode = each["postcode"].strip()
        phone = each["phone"].strip()
        pricerange = each["pricerange"].strip()
        if pricerange == "?":
            pricerange = ""
        if name in entity_info:
            print("repeated entity in attraction ", name)
            exit()

        if address in attr_ent_map["attraction_address"]:
            attr_ent_map["attraction_address"][address].add(name)
            attr_ent_map["attraction_address"][cvt_address(address)].add(name)
        else:
            attr_ent_map["attraction_address"][address] = {name}
            attr_ent_map["attraction_address"][cvt_address(address)] = {name}

        if phone in attr_ent_map["attraction_phone"]:
            attr_ent_map["attraction_phone"][phone].add(name)
        else:
            attr_ent_map["attraction_phone"][phone] = {name}

        if postcode in attr_ent_map["attraction_postcode"]:
            attr_ent_map["attraction_postcode"][postcode].add(name)
        else:
            attr_ent_map["attraction_postcode"][postcode] = {name}

        entity_info[name] = {
            "address": address,
            "postcode": postcode,
            "phone": phone,
            "area": each["area"],
            "attraction_type": each["type"]
        }
        if pricerange != "": 
            entity_info[name]["pricerange"] = pricerange

    hospital_db = json.load(open(input_path + "/hospital_db.json"))
    for each in hospital_db:
        name = normalize(each["department"])

        new_name = cvt_name(each["department"])
        new_name = "[hospital_department]::" + new_name
        if new_name != name and new_name not in entity_name_map:
            entity_name_map[new_name] = "[hospital_department]::" + name

        name = "[hospital_department]::" + name
        phone = each["phone"].strip()
        if name in entity_info:
            print("repeated entity in hospital", name)
            exit()

        if phone in attr_ent_map["hospital_phone"]:
            attr_ent_map["hospital_phone"][phone].add(name)
        else:
            attr_ent_map["hospital_phone"][phone] = {name}

        entity_info[name] = {
            "phone": phone,
        }

    restaurant_db = json.load(open(input_path + "/restaurant_db.json"))
    for each in restaurant_db:
        name = normalize(each["name"])

        new_name = cvt_name(each["name"])
        new_name = "[restaurant_name]::" + new_name
        if new_name != name and new_name not in entity_name_map:
            entity_name_map[new_name] = "[restaurant_name]::" + name

        name = "[restaurant_name]::" + name
        address = normalize(each["address"])
        postcode = each["postcode"].strip()
        if "phone" not in each:
            phone = "no phone"
        else:
            phone = each["phone"].strip()
        pricerange = each["pricerange"].strip()
        food = normalize(each["food"])
        if pricerange == "?":
            pricerange = ""
        if name in entity_info:
            print("repeated entity in restaurant", name)
            exit()

        if address in attr_ent_map["restaurant_address"]:
            attr_ent_map["restaurant_address"][address].add(name)
            attr_ent_map["restaurant_address"][cvt_address(address)].add(name)
        else:
            attr_ent_map["restaurant_address"][address] = {name}
            attr_ent_map["restaurant_address"][cvt_address(address)] = {name}

        if phone in attr_ent_map["restaurant_phone"]:
            attr_ent_map["restaurant_phone"][phone].add(name)
        else:
            attr_ent_map["restaurant_phone"][phone] = {name}

        if postcode in attr_ent_map["restaurant_postcode"]:
            attr_ent_map["restaurant_postcode"][postcode].add(name)
        else:
            attr_ent_map["restaurant_postcode"][postcode] = {name}

        entity_info[name] = {
            "address": address,
            "postcode": postcode,
            "phone": phone,
            "area": each["area"],
            "food": food
        }
        if pricerange != "": 
            entity_info[name]["pricerange"] = pricerange
        if "introduction" in each:
            entity_info[name]["introduction"] = each["introduction"].strip()

    hotel_db = json.load(open(input_path + "/hotel_db.json"))
    for each in hotel_db:
        name = normalize(each["name"])

        new_name = cvt_name(each["name"])
        new_name = "[hotel_name]::" + new_name
        if new_name != name and new_name not in entity_name_map:
            entity_name_map[new_name] = "[hotel_name]::" + name

        name = "[hotel_name]::" + name
        address = normalize(each["address"])
        postcode = each["postcode"].strip()
        phone = each["phone"].strip()
        pricerange = normalize(each["pricerange"])
        if pricerange == "?":
            pricerange = ""
        if name in entity_info:
            print("repeated entity in hotel ", name)
            exit()

        if address in attr_ent_map["hotel_address"]:
            attr_ent_map["hotel_address"][address].add(name)
            attr_ent_map["hotel_address"][cvt_address(address)].add(name)
        else:
            attr_ent_map["hotel_address"][address] = {name}
            attr_ent_map["hotel_address"][cvt_address(address)] = {name}

        if phone in attr_ent_map["hotel_phone"]:
            attr_ent_map["hotel_phone"][phone].add(name)
        else:
            attr_ent_map["hotel_phone"][phone] = {name}

        if postcode in attr_ent_map["hotel_postcode"]:
            attr_ent_map["hotel_postcode"][postcode].add(name)
        else:
            attr_ent_map["hotel_postcode"][postcode] = {name}

        entity_info[name] = {
            "address": address,
            "postcode": postcode,
            "phone": phone,
            "area": each["area"],
            "parking": each["parking"],
            "internet": each["internet"],
            "stars": each["stars"],
            "hotel_type": each["type"]
        }
        if pricerange != "": 
            entity_info[name]["pricerange"] = pricerange

    """
    train_db = json.load(open(input_path + "/train_db.json"))
    for each in train_db:
        name = each["trainID"].strip().lower()
        day = each["day"].strip()
        if name in entity_info:
            print("repeated entity in train ", name)
            exit()
        entity_info[name] = {
            "day": day
        }
    """

    json.dump(entity_info, open("./entity_info.json", "w"))
    json.dump(entity_name_map, open("./entity_name_map.json", "w"))

    new_attr_ent_map = {}
    for domain, res in attr_ent_map.items():
        new_attr_ent_map[domain] = {}
        for attr, vals in res.items():
            new_attr_ent_map[domain][attr] = list(vals)
    json.dump(new_attr_ent_map, open("./attr_ent_map.json", "w"))

    return entity_info, new_attr_ent_map, entity_name_map


def generate_graph(entity_info):
    entity_list = set(entity_info.keys())
    val_set = {"pricerange", "area", "food"}
    latent_rel_set = {"internet", "parking", "stars", "attraction_type", "hotel_type"}
    #latent_rel_set = set()

    adj = {}
    attr_ents = {}
    for ent, info in entity_info.items():
        if ent not in adj:
            adj[ent] = set()
        for attr, val in info.items():
            attr_val = "::".join([attr, val])
            if attr in val_set:
                attr = "[value_" + attr + "]"
                attr_val = "::".join([attr, val])
                adj[ent].add(attr_val)
            if attr in latent_rel_set:
                if attr_val not in attr_ents:
                    attr_ents[attr_val] = {ent}
                else:
                    attr_ents[attr_val].add(ent)

    for attr, ents in attr_ents.items():
        for i, ent_i in enumerate(ents):
            for j, ent_j in enumerate(ents):
                if i != j:
                    adj[ent_i].add(ent_j)

    all_ents = set()
    for ent, ents in adj.items():
        all_ents.add(ent)
        for each in ents:
            all_ents.add(each)

    new_adj = {ent: set() for ent in all_ents}
    for ent, ents in adj.items():
        new_adj[ent] |= ents
        for each in ents:
            new_adj[each].add(ent)

    for ent, ents in new_adj.items():
        new_adj[ent] = list(ents)

    json.dump(new_adj, open("./adj_simple.json", "w"))
    json.dump(sorted(all_ents), open("./entity_list_simple.json", "w"))


def disambiguate(prepro_res):
    for dial_id, res in prepro_res.items():
        last_single_ent = ""
        for i, (user_ents, sys_ents) in enumerate(zip(res["user_ent"], res["sys_ent"])):
            for j, user_ent in enumerate(user_ents):
                if isinstance(user_ent, str):
                    attr = user_ent.split("::")[0]
                    if "name" in attr:
                        last_single_ent = user_ent
                elif isinstance(user_ent, list):
                    res_candi = ""
                    for each_candi in user_ent:
                        if each_candi == last_single_ent:
                            res_candi = each_candi
                            break
                    if res_candi == "":
                        res_candi = user_ent[0]
                        last_single_ent = res_candi
                    prepro_res[dial_id]["user_ent"][i][j] = res_candi

            for j, sys_ent in enumerate(sys_ents):
                if isinstance(sys_ent, str):
                    attr = sys_ent.split("::")[0]
                    if "name" in attr:
                        last_single_ent = sys_ent
                elif isinstance(sys_ent, list):
                    res_candi = ""
                    for each_candi in sys_ent:
                        if each_candi == last_single_ent:
                            res_candi = each_candi
                            break
                    if res_candi == "":
                        res_candi = sys_ent[0]
                        last_single_ent = res_candi
                    prepro_res[dial_id]["sys_ent"][i][j] = res_candi

    return prepro_res


def generate_final_res(prepro_res):
    for dial_id, res in prepro_res.items():
        user_ori_utt = []
        sys_ori_utt = [] 
        for i, (user_ents, sys_ents, user_utts, sys_utts) in enumerate(zip(res["user_ent"], res["sys_ent"], res["user_utt"], res["sys_utt"])) :
            tmp_user_ori_utt = [] 
            for j, word in enumerate(user_utts):
                if "::" in word:
                    token, words = word.split("::")
                    prepro_res[dial_id]["user_utt"][i][j] = token
                    if token in symbol_list:
                        tmp_user_ori_utt.append(token)
                    else:
                        tmp_user_ori_utt.append(words.replace(" ", "_"))
                else:
                    if word == "<$>":
                        tmp_user_ori_utt.append(user_ents[j].split("::")[1])
                    else:
                        word = re.sub(r'^\d+$', '[value_count]', word)
                        tmp_user_ori_utt.append(word) 
                        prepro_res[dial_id]["user_utt"][i][j] = word
            user_ori_utt.append(tmp_user_ori_utt)

            tmp_sys_ori_utt = []
            for j, word in enumerate(sys_utts):
                if "::" in word:
                    token, words = word.split("::")
                    prepro_res[dial_id]["sys_utt"][i][j] = token
                    if token in symbol_list:
                        tmp_sys_ori_utt.append(token)
                    else:
                        tmp_sys_ori_utt.append(words)
                else:
                    if word == "<$>":
                        tmp_sys_ori_utt.append(sys_ents[j].split("::")[1])
                    else:
                        word = re.sub(r'^\d+$', '[value_count]', word)
                        tmp_sys_ori_utt.append(word)
                        prepro_res[dial_id]["sys_utt"][i][j] = word
            sys_ori_utt.append(tmp_sys_ori_utt)

            for user_ent in user_ents:
                if isinstance(user_ent, list):
                    print("still have list: ", user_ent, i, dial_id)
                    exit()
            for sys_ent in sys_ents:
                if isinstance(sys_ent, list):
                    print("still have list: ", sys_ent, i, dial_id)
                    exit()

        prepro_res[dial_id]["user_utt_ori"] = user_ori_utt
        prepro_res[dial_id]["sys_utt_ori"] = sys_ori_utt

    return prepro_res


def get_list(inputfile):
    x_set = set()
    for line in open(inputfile):
        x_set.add(line.strip())
    
    return x_set


def helper(res, dial_turns):
    len_ = len(res["user_utt"])

    blank_ent = "none"

    stack_utt = []
    for user_utts, sys_utts in zip(res["user_utt"], res["sys_utt"]):
        stack_utt.append(" ".join(user_utts))
        stack_utt.append(" ".join(sys_utts))

    utts = []
    ents = []
    for i in range(len_):
        utts.append("\t".join(stack_utt[max((i+1)*2-dial_turns, 0) : (i+1)*2]))

        tmp_ent = []
        for j, ent in enumerate(res["sys_ent"][i]):
            if ent == "":
                tmp_ent.append(blank_ent)
            else:
                tmp_ent.append(ent) 
        ents.append(" ".join(tmp_ent))

    return utts, ents


def helper_ver1(res, dial_turns):
    len_ = len(res["user_utt"])

    blank_ent = "none"

    utts = []
    ents = []
    ans_utt_ori = []
    for i in range(len_):
        tmp_utts = []
        ans_utt_ori.append(" ".join(res["sys_utt_ori"][i]))
        for j in reversed(range(dial_turns)):
            hist = round(j / 2)
            idx = i - hist
            if idx < 0:
                continue
            
            if not j % 2:
                if hist == 0:
                    tmp_utts.append(" ".join(res["sys_utt"][idx]))
                else:
                    tmp_utts.append(" ".join(res["sys_utt_ori"][idx]))
            else:
                tmp_utts.append(" ".join(res["user_utt_ori"][idx]))
        utts.append("\t".join(tmp_utts))

        tmp_ent = []
        for j, ent in enumerate(res["sys_ent"][i]):
            if ent == "":
                tmp_ent.append(blank_ent)
            else:
                tmp_ent.append(ent) 
        ents.append(" ".join(tmp_ent))

    return utts, ents, ans_utt_ori


def dump2txt(data, output_path):
    output = open(output_path, "w")
    for each in data:
        output.write(each + "\n")
    output.close()


def split_train_val_test(dial_turns, dial_data):
    val_list = get_list("./valListFile.json")
    test_list = get_list("./testListFile.json")

    train_utts, train_ents, train_ans_utt_ori, val_utts, val_ents, val_ans_utt_ori, test_utts, test_ents, test_ans_utt_ori = [[] for i in range(9)]     

    for id_, res in dial_data.items():
        #utts, ents = helper(res, dial_turns)
        utts, ents, ans_utt_ori = helper_ver1(res, dial_turns)
        if id_ in val_list:
            val_utts += utts
            val_ents += ents
            val_ans_utt_ori += ans_utt_ori
        elif id_ in test_list:
            test_utts += utts
            test_ents += ents
            test_ans_utt_ori += ans_utt_ori
        else:
            train_utts += utts
            train_ents += ents
            train_ans_utt_ori += ans_utt_ori

    dump2txt(train_utts, "./train_utt_1.txt")
    dump2txt(train_ents, "./train_ent_1.txt")
    dump2txt(train_ans_utt_ori, "./train_ans_utt_ori_1.txt")
    dump2txt(val_utts, "./valid_utt_1.txt")
    dump2txt(val_ents, "./valid_ent_1.txt")
    dump2txt(val_ans_utt_ori, "./valid_ans_utt_ori_1.txt")
    dump2txt(test_utts, "./test_utt_1.txt")
    dump2txt(test_ents, "./test_ent_1.txt")
    dump2txt(test_ans_utt_ori, "./test_ans_utt_ori_1.txt")


def main():
    entity_info, attr_ent_map, entity_name_map = generate_info("./graph_data") 
    generate_graph(entity_info)
    generate_data(attr_ent_map, entity_name_map)

    data = json.load(open("./preprocess_utt_ent.json"))
    new_data = disambiguate(data)
    new_data = generate_final_res(new_data)
    json.dump(new_data, open("./preprocess_utt_ent_new.json", "w"))

    data = json.load(open("./preprocess_utt_ent_new.json"))
    split_train_val_test(3, data)


if __name__ == "__main__":
    main()

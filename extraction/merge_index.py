from inference.candidate_retriever import normalize_string, get_stopwords, load_index
import json

def load_old_index(file_path):

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

        dict = json.loads(content)
    return dict

stopwords = get_stopwords()
freebase_index = load_index("../data/surface_forms.txt")
dbpedia_index = load_old_index("../data/freebaseEntityIndex.txt")

for entry in dbpedia_index.keys():
    tuple = dbpedia_index[entry]
    u = ""
    f = 0
    for k in tuple.keys():
        u = k
        f = tuple[k]

    label = normalize_string(entry)

    if label in freebase_index.keys():
        added_uris = freebase_index[label]

        ## sum the frequencies
        if u in added_uris.keys():
            added_uris[u] = f + added_uris[u]
            freebase_index[label] = added_uris
    else:
        added_uris = {}
        added_uris[u] = f
        freebase_index[label] = added_uris


f = open('../data/surface_forms_new.txt', 'w')

for label in freebase_index.keys():
    # if len(label) < 3 or len(label) > 50 :
    #     continue

    added_uris = freebase_index[label]

    for u in added_uris.keys():
        line = label+"\t"+u+"\t"+str(added_uris[u])

        f.write(line+"\n")

f.close()
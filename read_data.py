import pandas as pd
import numpy as np
import csv
from nltk.cluster import KMeansClusterer, euclidean_distance
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from itertools import product, filterfalse
from nltk import stem


def random_key_from_dict(dictionary, seed=123):
    keys_list = list(dictionary.keys())
    np.random.seed(seed)
    random_index = np.random.choice(len(keys_list))
    return keys_list[random_index]


DATA_PATH = "./"
wasabi_songs = pd.read_csv(
    DATA_PATH + "wasabi_songs.csv",
    sep="\t",
    usecols=[
        "_id",
        "abstract",
        "albumTitle",
        "album_genre",
        "artist",
        "genre",
        "has_emotion_tags",
        "lastfm_id",
        "title",
    ],
)


### LastFM tags
emotion_tags = pd.read_pickle(DATA_PATH + "lastfm_id_to_emotion_tags.pickle")
emotion_tags = {k: dict(v) for k, v in emotion_tags.items()}

# track t that has both social and emotion tags
socemo = list(set(emotion_tags.keys()))
np.random.seed(123)
t = socemo[np.random.choice(len(socemo))]
print("LastFM ID:", t, "\n")
print("Emotion tags:", "\n", emotion_tags[t])

emotion_frame = pd.DataFrame.from_dict(emotion_tags, orient="index")
emotion_frame.fillna(0, inplace=True)
all_emotions = emotion_frame.columns.to_list()

emotion_lemmas = dict()
emotion_synsets = dict()
similar_to = dict()
for emotion in all_emotions:
    similar_to[emotion] = []
    lemmas = set()
    synsets = wn.synsets(emotion, lang="eng", check_exceptions=True)
    others = list()
    for es in synsets:
        others.extend(es.similar_tos())
        others.extend(es.also_sees())
    synsets = set(synsets + others)
    emotion_synsets[emotion] = synsets
    for es in synsets:
        for lemma in es.lemmas():
            lemmas.add(lemma)
            related = lemma.derivationally_related_forms()
            related.extend(lemma.pertainyms())
            for r in related:
                lemmas.add(r)
                if r.name() in all_emotions:
                    similar_to[emotion].append(r.name())
            if lemma.name() in all_emotions:
                similar_to[emotion].append(lemma.name())
                all_emotions.remove(lemma.name())
    emotion_lemmas[emotion] = lemmas
    print(lemmas)

emotion_categories = dict()
emotion_categories["happy"] = set(
    similar_to["happy"]
    + similar_to["playful"]
    + similar_to["lively"]
    + similar_to["jolly"]
    + ["ebullient", "fun", "cheer", "celebrate", "excite", "excitement"]
)
emotion_categories["relaxing"] = set(
    similar_to["mellow"]
    + similar_to["relaxing"]
    + similar_to["calm"]
    + similar_to["comfort"]
    + similar_to["serenity"]
    + similar_to["untroubled"]
    + similar_to["delicate"]
    + ["calmed", "quietness", "quietly", "gentle"]
)

emotion_categories["sad"] = set(
    similar_to["sad"]
    + similar_to["heartbreak"]
    + similar_to["miserable"]
    + similar_to["gloomy"]
    + ["misery", "heartbroken", "somber", "bleak", "lament"]
)

emotion_categories["angry"] = set(
    similar_to["aggressive"]
    + similar_to["fight"]
    + similar_to["furious"]
    + similar_to["severe"]
    + [
        "anger",
        "harsh",
        "tense",
        "confrontational",
        "annoyed",
        "feral",
        "incredulous",
    ]
)

emotion_categories["romantic"] = set(
    similar_to["sexy"] + similar_to["fiery"] + ["romantic", "desire", "thrill"]
)

emotion_categories["fear"] = set(
    ["fear", "eerie"] + similar_to["spooky"] + similar_to["anxious"]
)

emotion_categories["funny"] = set(
    ["silly", "outrageous", "confident"] + similar_to["funny"]
)

emotion_categories["poignant"] = set(
    similar_to["poignant"]
    + similar_to["introverted"]
    + ["lament", "dreamy", "plaintive", "springlike"]
)


# def aggregate_emotions(row):
#     sums = dict()
#     for k, v in emotion_categories.items():
#         sums[k] = 0
#         values = row.loc[v]
#         sums[k] = values.sum()
#     max_sum = max(sums.items(), key=lambda x: x[1])
#     return max_sum[0]


# emotion_tagged_songs = emotion_frame.agg(aggregate_emotions, axis=1)
# emotion_tagged_songs.to_csv("main_emotions.csv")
# def merge(sets):
#     merged = True
#     while merged:
#         merged = False
#         results = []
#         while sets:
#             (key_common, common), rest = sets[0], sets[1:]
#             sets = []
#             for k, v in rest:
#                 if v.isdisjoint(common):
#                     sets.append((k, v))
#                 else:
#                     merged = True
#                     common |= v
#             results.append((key_common, common))
#         sets = results
#     return sets


# def merge_synsets(sets):
#     merged = True
#     while merged:
#         merged = False
#         results = []
#         while sets:
#             (key_common, common), rest = sets[0], sets[1:]
#             sets = []
#             similar = []
#             for k, v in rest:
#                 for s1, s2 in product(common, v):
#                     if s1.pos() == s2.pos():
#                         similarity = s1.path_similarity(s2)
#                         if similarity > 0.3:
#                             merged = True
#                             similar.append(k)
#                             break
#                 if not merged:
#                     sets.append((k, v))
#             results.append((key_common, similar))
#         sets = results
#     return sets


# pairs = list(emotion_lemmas.items())
# merged = merge(pairs)
# print(merged)

# new_merged = []
# for k, v in merged:
#     v = set([x.name() for x in v if all_emotions.count(x.name()) > 0])
#     new_merged.append((k, v))
# print(new_merged)
glove_embeddings = pd.read_csv(
    "glove.6B.100d.txt",
    delim_whitespace=True,
    header=None,
    index_col=0,
    quoting=csv.QUOTE_NONE,
)
words = glove_embeddings.index.to_list()
embedding_vals = glove_embeddings.to_numpy()

word_embed = dict()
embed_word = dict()
embeddings = list()
embedding_strs = list()
for i, emotion in enumerate(all_emotions):
    if words.count(emotion) > 0:
        idx = words.index(emotion)
        embedding = embedding_vals[idx]
        word_embed[i] = idx
        embed_word[emotion] = embedding
        embeddings.append(embedding)
        embedding_strs.append(np.array2string(embedding))


## Show a summary ###
summaries = pd.read_pickle(DATA_PATH + "id_to_summary_lines.pickle")
song_id = random_key_from_dict(summaries, seed=12)
print("\n".join(summaries[song_id]))
emotion_tagged_songs = pd.read_csv("main_emotions.csv", header=None, names=["emotion"])
summaries = pd.DataFrame.from_dict(summaries, orient="index")
summaries = summaries.rename(index=lambda x: "ObjectId(" + x + ")")
join_summaries = wasabi_songs.join(summaries, how="inner", on="_id")
join_emotions = join_summaries.join(emotion_tagged_songs, how="inner", on="lastfm_id")

join_emotions.set_index("_id", inplace=True)
join_emotions["summary"] = (
    join_emotions[0] + join_emotions[1] + join_emotions[2] + join_emotions[3]
)
join_emotions.drop(
    columns=["abstract", "albumTitle", "genre", "has_emotion_tags", 0, 1, 2, 3],
    inplace=True,
)
print("joining")

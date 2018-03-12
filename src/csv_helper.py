import csv
from random import shuffle

from typing import List, Dict

SCORE = 'score'
OWNER_ID = 'owner_id'
ANS_COUNT = 'ans_count'
TAG_ID = 'tag_id'


def read(file_questions, file_tags) -> List[Dict[str, int]]:
    result = []
    with open(file_questions) as questions, open(file_tags) as tags:
        for q, t in zip(csv.reader(questions), csv.reader(tags)):
            try:
                result.append({
                    SCORE: int(q[4]),
                    OWNER_ID: int(q[5]),
                    ANS_COUNT: int(q[6]),
                    TAG_ID: t[1]
                })
            except:
                pass

    return result


# Method reduces question set to n often used tags.
def reduce(raw, n) -> List:
    counts = {}
    for i in raw:
        if i[TAG_ID] is not None:
            counts[i[TAG_ID]] = 0
    for i in raw:
        counts[i[TAG_ID]] += 1
    most = []
    for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        for q in raw:
            if q[TAG_ID] == k:
                most.append(q)
        n -= 1
        if n == 0:
            break
    return most


def preprocess(most) -> (List, List):
    train_input = []
    train_output = []
    train = []
    tags_vec = {}
    tag_vec = 0
    for m in most:
        input = [m[SCORE], m[OWNER_ID], m[ANS_COUNT]]
        try:
            target = tags_vec[m[TAG_ID]]
        except:
            tags_vec[m[TAG_ID]] = tag_vec
            tag_vec += 1
            target = tags_vec[m[TAG_ID]]
        train.append([input, target])
    shuffle(train)
    for i in train:
        train_input.append(i[0])
        train_output.append(i[1])
    return train_input, train_output


def export(path, input, output, n_export):
    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile)
        for i, o in zip(input, output):
            if n_export == 0:
                break
            w.writerow(i + [o])
            n_export -= 1
    pass


def _import(path):
    pass


if __name__ == "__main__":
    read_tuple = read('data/questions.csv', 'data/question_tags.csv')
    reduced_tuple = reduce(read_tuple, 20)
    input, output = preprocess(reduced_tuple)
    export('data/tagged_questions.csv', input, output, 10000)

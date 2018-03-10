import csv

from typing import List, Dict


def read(file_questions, file_tags) -> List[Dict[str, int]]:
    result = []
    with open(file_questions) as questions, open(file_tags) as tags:
        for q, t in zip(csv.reader(questions), csv.reader(tags)):
            try:
                result.append({
                    'score': int(q[4]),
                    'owner_id': int(q[5]),
                    'ans_count': int(q[6]),
                    'tag_id': t[1]
                })
            except:
                pass

    return result


# Method reduces question set to n often used tags.
def reduce(result, n):
    counts = {}
    for i in result:
        if i['tag_id'] is not None:
            counts[i['tag_id']] = 0
    for i in result:
        counts[i['tag_id']] += 1
    most = []
    for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        for q in result:
            if q['tag_id'] == k:
                most.append(q)
        n -= 1
        if n == 0:
            break

    return most


if __name__ == "__main__":
    print(len(reduce(read('data/questions.csv', 'data/question_tags.csv'), 20)))

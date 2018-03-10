import csv_helper

nbr_tags = 20
train_input = []
train_target = []


def preprocess_data():
    raw = csv_helper.read('data/questions.csv', 'data/question_tags.csv')
    most = csv_helper.reduce(raw, nbr_tags)
    tags_vec = {}
    tag_vec = 0
    for m in most:
        train_input.append([m['score'], m['owner_id'], m['ans_count']])
        try:
            train_target.append([tags_vec[m['tag_id']]])
        except:
            tags_vec[m['tag_id']] = tag_vec
            tag_vec += 1
            train_target.append([tags_vec[m['tag_id']]])

    print(train_input)
    print(train_target)


def setup_model():
    pass


if __name__ == "__main__":
    preprocess_data()

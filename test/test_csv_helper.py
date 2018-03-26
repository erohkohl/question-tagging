import src.util.csv_helper as csv_helper

result = []


def setup_module(module):
    global result
    result = csv_helper.read('data/questions.csv', 'data/question_tags.csv')


def test_csv_helper_should_parse_correct():
    assert result[0] == {'score': 458, 'owner_id': 8, 'ans_count': 13, 'tag_id': 'c#'}
    assert result[1] == {'score': 207, 'owner_id': 9, 'ans_count': 5, 'tag_id': 'winforms'}
    assert result[2] == {'score': 1410, 'owner_id': 1, 'ans_count': 58, 'tag_id': 'decimal'}
    assert result[3] == {'score': 1129, 'owner_id': 1, 'ans_count': 33, 'tag_id': 'opacity'}
    assert result[4] == {'score': 451, 'owner_id': 9, 'ans_count': 25, 'tag_id': 'html'}
    assert result[5] == {'score': 290, 'owner_id': 11, 'ans_count': 8, 'tag_id': 'css'}
    assert result[6] == {'score': 78, 'owner_id': 2, 'ans_count': 5, 'tag_id': 'css3'}
    assert result[7] == {'score': 114, 'owner_id': 2, 'ans_count': 11, 'tag_id': 'internet-explorer-7'}
    assert result[8] == {'score': 222, 'owner_id': 13, 'ans_count': 21, 'tag_id': 'c#'}


def test_parsed_result_should_have_correct_size():
    assert len(result) == 11539365


def test_reduce():
    reduced = csv_helper.reduce(result, 20)
    assert reduced[0] == {'score': 458, 'owner_id': 8, 'ans_count': 13, 'tag_id': 'c#'}
    assert reduced[1] == {'score': 222, 'owner_id': 13, 'ans_count': 21, 'tag_id': 'c#'}
    assert reduced[2] == {'score': 55, 'owner_id': 33, 'ans_count': 2, 'tag_id': 'c#'}
    assert reduced[3] == {'score': 83, 'owner_id': 59, 'ans_count': 12, 'tag_id': 'c#'}
    assert reduced[4] == {'score': 56, 'owner_id': 91, 'ans_count': 4, 'tag_id': 'c#'}
    assert reduced[5] == {'score': 51, 'owner_id': 91, 'ans_count': 9, 'tag_id': 'c#'}
    assert reduced[6] == {'score': 125, 'owner_id': 59, 'ans_count': 13, 'tag_id': 'c#'}
    assert reduced[7] == {'score': 12, 'owner_id': 157, 'ans_count': 5, 'tag_id': 'c#'}
    assert reduced[8] == {'score': 25, 'owner_id': 91, 'ans_count': 10, 'tag_id': 'c#'}


def test_reduce_length():
    assert len(csv_helper.reduce(result, 20)) == 3367577

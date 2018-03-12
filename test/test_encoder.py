import src.encoder as encoder


def test_one_hot_trivial_case():
    assert encoder.one_hot(1, 1) == [1]


def test_one_hot_three_size_five():
    assert encoder.one_hot(3, 5) == [0, 0, 1, 0, 0]


def test_one_hot_five_size_five():
    assert encoder.one_hot(5, 5) == [0, 0, 0, 0, 1]


def test_one_hot_five_size_five_neg():
    assert encoder.one_hot(5, 5) != [1, 0, 0, 0, 1]

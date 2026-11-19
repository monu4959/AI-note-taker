from pipeline.numbers import numbers_in, value_in


def test_spoken_decimal_forms_match():
    assert value_in("12.50", numbers_in("production cost's at twelve fifty"))
    assert value_in("12.50", numbers_in("for twelve Euros fifty"))
    assert value_in("12.5", numbers_in("it costs 12.50"))
    assert value_in("4.2", numbers_in("churn at four point two percent"))


def test_different_number_does_not_match():
    assert not value_in("25", numbers_in("how for twenty or twelve Euros fifty"))
    assert not value_in("42", numbers_in("churn at four point two percent"))


def test_disfluent_repeats_and_halves():
    # AMI dev cases that were false "wrong numbers" before the fix
    assert value_in("12, 12.5", numbers_in("up around about twelve twelve and a half as to what"))
    assert value_in("8 million", numbers_in("absorb eight eight million units per year"))
    assert value_in("12 and a half", numbers_in("only twelve and a half Euros you have to spend"))
    # and the real ASR error stays wrong
    assert not value_in("three", numbers_in("so now we're thirty minutes alone again"))


def test_large_numbers_and_multiple_values():
    assert value_in("50 million", numbers_in("profit aim is fifty million Euros"))
    assert value_in("30-40", numbers_in("maybe 30 to 40% of the buttons"))

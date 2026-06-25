from zero_keras import wrappers


def test_wrappers():
    """Test wrappers module."""
    wrappers.SKLearnClassifier()
    wrappers.SKLearnRegressor()
    wrappers.SKLearnTransformer()

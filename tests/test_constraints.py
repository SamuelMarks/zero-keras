def test_constraints_methods():
    from zero_keras.constraints import Constraint, MaxNorm, MinMaxNorm, NonNeg, UnitNorm

    for cls in [Constraint, MaxNorm, MinMaxNorm, NonNeg, UnitNorm]:
        c = cls()
        assert c.get_config() == {}
        c2 = cls.from_config({})
        assert isinstance(c2, cls)

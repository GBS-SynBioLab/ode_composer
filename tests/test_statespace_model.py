import pytest
from ode_composer.statespace_model import StateSpaceModel


def test_from_string():
    """test that the correct State-Space model is built from strings"""
    states = {"x": "sigma*(y-x)", "y": "x*(rho-z)-y", "z": "x*y-beta*z"}
    parameters = {"sigma": 10, "beta": 8 / 3, "rho": 28}
    ss = StateSpaceModel.from_string(states=states, parameters=parameters)
    assert len(ss.state_vector) == len(states)
    assert len(ss.parameters) == len(parameters)


def test_add_state_equation():
    # define a MAK model
    states = {"x1": {"x1": "m11"}, "x2": {"x1": "m21", "x2": "m22"}}
    parameters = {"m11": -0.5, "m12": 0.1, "m21": 0.6, "m22": -0.3}
    ss = StateSpaceModel(states=states, parameters=parameters)

    other_state = {"x3": {"x1": "m11"}}
    ss.add_state_equation(other_state, ss.parameters)

    assert len(ss.state_vector) == len(states) + len(other_state)


def test_get_rhs():
    states = {"x1": {"x1": "m11"}, "x2": {"x1": "m21", "x2": "m22"}}
    parameters = {"m11": -0.5, "m12": 0.1, "m21": 0.6, "m22": -0.3}
    ss = StateSpaceModel(states=states, parameters=parameters)
    y = []
    states_names = ["x1", "x2"]
    with pytest.raises(ValueError, match=r"#states:\d+ must be equal to \d+$"):
        ss.get_rhs(t=[], y=y, states=states_names)

    # TODO test the other functionalities


def test_repr():
    """test that the repr of StateSpace model returns"""
    states = {"x": "sigma*(y-x)", "y": "x*(rho-z)-y", "z": "x*y-beta*z"}
    parameters = {"sigma": 10, "beta": 8 / 3, "rho": 28}
    ss = StateSpaceModel.from_string(states=states, parameters=parameters)

    number_of_new_lines = repr(ss).count("\n")
    assert len(ss.state_vector) == number_of_new_lines

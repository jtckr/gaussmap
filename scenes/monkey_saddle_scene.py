from gaussmap import parameterizations, scene


class MonkeySaddleScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a monkey saddle."""

    def __init__(self, *args, **kwargs) -> None:
        monkey_saddle = parameterizations.MonkeySaddleParameterization()
        expression = monkey_saddle.expression
        u_range = monkey_saddle.u_range
        v_range = monkey_saddle.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

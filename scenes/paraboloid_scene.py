from gaussmap import parameterizations, scene


class ParaboloidScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a paraboloid."""

    def __init__(self, *args, **kwargs) -> None:
        paraboloid = parameterizations.ParaboloidParameterization()
        expression = paraboloid.expression
        u_range = paraboloid.u_range
        v_range = paraboloid.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

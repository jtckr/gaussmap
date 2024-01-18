from gaussmap import parameterizations, scene


class HyperbolicParaboloidScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a hyperbolic
    paraboloid.
    """

    def __init__(self, *args, **kwargs) -> None:
        hyperbolic_paraboloid = parameterizations.HyperbolicParaboloidParameterization()
        expression = hyperbolic_paraboloid.expression
        u_range = hyperbolic_paraboloid.u_range
        v_range = hyperbolic_paraboloid.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

from gaussmap import parameterizations, scene


class HyperboloidScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a hyperboloid of
    one-sheet.
    """

    def __init__(self, *args, **kwargs) -> None:
        hyperboloid = parameterizations.HyperboloidParameterization()
        expression = hyperboloid.expression
        u_range = hyperboloid.u_range
        v_range = hyperboloid.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

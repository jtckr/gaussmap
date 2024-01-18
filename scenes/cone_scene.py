from gaussmap import parameterizations, scene


class ConeScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a cone."""

    def __init__(self, *args, **kwargs) -> None:
        cone = parameterizations.ConeParameterization()
        expression = cone.expression
        u_range = cone.u_range
        v_range = cone.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

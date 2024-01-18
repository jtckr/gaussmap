from gaussmap import parameterizations, scene


class SphereScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a sphere."""

    def __init__(self, *args, **kwargs) -> None:
        sphere = parameterizations.SphereParameterization()
        expression = sphere.expression
        u_range = sphere.u_range
        v_range = sphere.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

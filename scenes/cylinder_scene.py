from gaussmap import parameterizations, scene


class CylinderScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a cylinder."""

    def __init__(self, *args, **kwargs) -> None:
        cylinder = parameterizations.CylinderParameterization()
        expression = cylinder.expression
        u_range = cylinder.u_range
        v_range = cylinder.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

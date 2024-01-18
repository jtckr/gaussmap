from gaussmap import parameterizations, scene


class CatenoidScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a catenoid."""

    def __init__(self, *args, **kwargs) -> None:
        catenoid = parameterizations.CatenoidParameterization()
        expression = catenoid.expression
        u_range = catenoid.u_range
        v_range = catenoid.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

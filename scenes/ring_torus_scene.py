from gaussmap import parameterizations, scene


class RingTorusScene(scene.GaussMapScene):
    """A scene that renders the Gauss map transformation of a ring torus."""

    def __init__(self, *args, **kwargs) -> None:
        ring_torus = parameterizations.RingTorusParameterization()
        expression = ring_torus.expression
        u_range = ring_torus.u_range
        v_range = ring_torus.v_range

        super().__init__(expression, u_range, v_range, *args, **kwargs)

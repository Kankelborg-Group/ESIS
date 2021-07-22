from . import as_designed_single_channel, as_designed_active_channels


def test_as_designed_single_channel():
    optics = as_designed_single_channel()
    assert optics.rays_output is not None


def test_as_designed_active_channels():
    optics = as_designed_active_channels()
    assert optics.rays_output is not None

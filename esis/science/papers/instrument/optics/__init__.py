import typing as typ
import pathlib
import esis

__all__ = ['as_designed_single_channel']


def _cache(factory: typ.Callable[[], esis.optics.Optics]):
    def func():
        path = pathlib.Path(__file__).parent / f'{factory.__name__}.pickle'
        if path.exists():
            optics = esis.optics.Optics.from_pickle(path)
        else:
            optics = factory()
            rays = optics.rays_output
            optics.to_pickle(path)

        return optics

    return func


default_pupil_samples = 21
default_pupil_is_stratified_random = True
default_field_samples = 21
default_field_is_stratified_random = False


@_cache
def as_designed_single_channel() -> esis.optics.Optics:
    return esis.optics.design.final(
        pupil_samples=default_pupil_samples,
        pupil_is_stratified_random=default_pupil_is_stratified_random,
        field_samples=default_field_samples,
        field_is_stratified_random=default_field_is_stratified_random,
        all_channels=False,
    )


@_cache
def as_designed_active_channels() -> esis.optics.Optics:
    return esis.optics.design.final_active(
        pupil_samples=default_pupil_samples,
        pupil_is_stratified_random=default_pupil_is_stratified_random,
        field_samples=default_field_samples,
        field_is_stratified_random=default_field_is_stratified_random,
    )


@_cache
def as_measured_single_channel() -> esis.optics.Optics:
    return esis.flight.optics.as_measured(
        pupil_samples=default_pupil_samples,
        pupil_is_stratified_random=default_pupil_is_stratified_random,
        field_samples=default_field_samples,
        field_is_stratified_random=default_field_is_stratified_random,
        all_channels=False,
    )

"""Opt-in, stdlib-only timing spans. Inactive calls never read a clock."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_OBSERVER = ContextVar('spire_learn_profile_observer', default=None)


@contextmanager
def observe(observer):
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


@contextmanager
def span(name):
    observer = _OBSERVER.get()
    if observer is None:
        yield
    else:
        with observer.span(name):
            yield


def observed(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            observer = _OBSERVER.get()
            if observer is None:
                return function(*args, **kwargs)
            with observer.span(name):
                return function(*args, **kwargs)
        return call
    return decorate

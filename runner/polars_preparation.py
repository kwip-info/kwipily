"""Explicitly typed, bounded Polars adapter. Never infer or reparse source values.

The v1 Table/lineage/artifact contract remains authoritative. Unproven casts,
expressions and complete-population algorithms deliberately use existing adapters.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.metadata import version

ENGINE_VERSION='2.0.0'
_engine=ContextVar('preparation_engine',default='disk')


def engine_identity(engine=None):
    engine=current_engine() if engine is None else engine
    if engine not in ('disk','polars'):raise ValueError('Unknown preparation engine')
    if engine=='disk':return {'name':'disk','version':'1'}
    for distribution in ('polars','polars-runtime-32'):
        try:installed=version(distribution)
        except Exception as error:raise ValueError('Polars preparation requires the pinned development dependencies') from error
        if installed!=ENGINE_VERSION:raise ValueError('Polars preparation runtime differs from the pinned version')
    return {'name':'polars','version':ENGINE_VERSION,'runtime':'polars-runtime-32'}


def current_engine():return _engine.get()


@contextmanager
def preparation_engine(engine):
    engine_identity(engine)
    token=_engine.set(engine)
    try:yield
    finally:_engine.reset(token)


class _ExactFallback(Exception):pass


def apply(table,op,config,check):
    raise ValueError('Retired Polars table execution is unavailable; use current Columns, Filter or Preprocess.')

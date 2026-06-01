# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Utility functions for AFC controller wrapper.
"""

import importlib

def resolve_wrapper_callable(callable_spec, default_callable=None, spec_name='callable'):
    """Resolve wrapper callable from JSON-serializable specification.

    Parameters
    ----------
    callable_spec : dict or None
        - None: loads default_callable
        - dict: must include:
            {
              "module": "python.module.path",
              "name": "callable_name"
            }
    default_callable : callable or None, optional
        Default callable to return when callable_spec is None.
    spec_name : str, optional
        Name used in error messages (e.g., "control_model", "pre_processor").

    Returns
    -------
    callable or None
        Resolved callable for wrapper use.

    Raises
    ------
    ValueError, ImportError, AttributeError, TypeError
        If the specification cannot be resolved to a callable.
    """

    if callable_spec is None:
        return default_callable

    if not isinstance(callable_spec, dict):
        raise ValueError(
            f'wrapper.{spec_name} must be None or a dict with keys '
            '"module" and "name".'
        )

    module_name = callable_spec.get('module')
    function_name = callable_spec.get('name')

    if not module_name or not function_name:
        raise ValueError(
            f'wrapper.{spec_name} must include non-empty "module" and "name".'
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f'Failed to import module "{module_name}" for wrapper.{spec_name}.'
        ) from exc

    try:
        resolved_callable = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f'Function "{function_name}" was not found in module "{module_name}" '
            f'for wrapper.{spec_name}.'
        ) from exc

    if not callable(resolved_callable):
        raise TypeError(
            f'wrapper.{spec_name} "{module_name}.{function_name}" is not callable.'
        )

    return resolved_callable

import sys
import importlib.util


def import_from_file(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def dict_to_simple_namespace(d):
    from types import SimpleNamespace

    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_simple_namespace(v) for k, v in d.items()})
    elif isinstance(d, (list, tuple)):
        return type(d)(dict_to_simple_namespace(v) for v in d)
    else:
        return d

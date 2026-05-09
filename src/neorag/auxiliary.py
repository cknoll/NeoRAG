

# TODO-AIDER: add docstring
from .config import INDEX_DIR


def ensure_dirs():
    """Create all required directories for NeoRAG (called by --bootstrap)."""
    INDEX_DIR.mkdir(exist_ok=True)
    # Future directory creation can be added here


class OneToOneMapping(object):
    """
    """
    def __init__(self, a_dict: dict = None, **kwargs):
        if a_dict is None:
            self.a = dict(**kwargs)
            self.b = dict([(v, k) for k, v in kwargs.items()])
        else:
            # handle the case where we do not map strings
            assert len(kwargs) == 0

            # make a copy
            self.a = dict(a_dict)
            self.b = dict([(v, k) for k, v in a_dict.items()])

        # assert 1to1-property
        assert len(self.a) == len(self.b)

    def add_pair(self, key_a, key_b):
        if key_a in self.a:
            msg = f"key_a '{key_a}' does already exist."
            raise KeyError(msg)

        if key_b in self.b:
            msg = f"key_b '{key_b}' does already exist."
            raise KeyError(msg)

        self.a[key_a] = key_b
        self.b[key_b] = key_a

        # assert 1to1-property
        assert len(self.a) == len(self.b)

    def remove_pair(self, key_a=None, key_b=None, strict=True):
        try:
            if key_a is not None:
                key_b = self.a.pop(key_a)
                self.b.pop(key_b)
            elif key_b is not None:
                key_a = self.b.pop(key_b)
                self.a.pop(key_a)
            else:
                msg = "Both keys are not allowed to be `None` at the the same time."
                raise ValueError(msg)
        except KeyError:
            if strict:
                raise
            # else -> pass

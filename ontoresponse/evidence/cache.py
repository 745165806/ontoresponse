import fcntl
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..common import Blocked, digest

SCHEMA = "ontoresponse-evidence-1"


class Cache:
    _shared_verified = OrderedDict()

    def __init__(self, root, max_bytes=None):
        self.max_bytes = max_bytes
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._verified = self._shared_verified

    def key(self, meta):
        return digest({"schema": SCHEMA, **meta})

    def get(self, key, expected=None):
        path = self.root / (key + ".npz")
        if not path.exists():
            return None
        stat = path.stat()
        version = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        if version in self._verified:
            arrays, meta = self._verified[version]
            self._verified.move_to_end(version)
            if expected is not None and meta["spec"] != expected:
                raise ValueError("Cache metadata mismatch")
            return {k: v.copy() for k, v in arrays.items()}, meta
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["metadata"]))
            if meta["key"] != key or (
                expected is not None and meta["spec"] != expected
            ):
                raise ValueError("Cache key/metadata mismatch")
            arrays = {k: f[k].copy() for k in f.files if k != "metadata"}
        actual = (
            array_hash(arrays)
            if meta.get("array_hash_algorithm") == "binary_v1"
            else digest({k: digest(v.tolist()) for k, v in arrays.items()})
        )
        if actual != meta["arrays_hash"]:
            raise ValueError("Corrupt evidence arrays")
        self._verified[version] = (arrays, meta)
        while len(self._verified) > 2048:
            self._verified.popitem(last=False)
        return {k: v.copy() for k, v in arrays.items()}, meta

    def put(self, key, spec, arrays, extra):
        if key != self.key(spec):
            raise ValueError("Wrong cache key")
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        if any(
            v.dtype.kind not in "bifu" or not np.isfinite(v).all()
            for v in arrays.values()
        ):
            raise ValueError("Only finite numeric observations may be cached")
        meta = {
            "key": key,
            "spec": spec,
            "extra": extra,
            "schema": SCHEMA,
            "array_hash_algorithm": "binary_v1",
            "arrays_hash": array_hash(arrays),
        }
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(
                    f, metadata=json.dumps(meta, sort_keys=True), **arrays
                )
                f.flush()
                os.fsync(f.fileno())
            # Identical keys must represent identical measurements, including concurrent shards.
            old = self.get(key, spec)
            if old is not None and any(
                not np.array_equal(old[0][k], v) for k, v in arrays.items()
            ):
                raise ValueError("Measurement drift for identical cache key")
            target = self.root / (key + ".npz")
            if self.max_bytes is None:
                os.replace(tmp, target)
            else:
                with (self.root / ".quota.lock").open("a") as guard:
                    fcntl.flock(guard, fcntl.LOCK_EX)
                    current = sum(
                        p.stat().st_size
                        for p in self.root.glob("*.npz")
                        if len(p.stem) == 64
                    )
                    projected = (
                        current
                        + Path(tmp).stat().st_size
                        - (target.stat().st_size if target.exists() else 0)
                    )
                    if projected > self.max_bytes:
                        raise Blocked(
                            "CACHE_QUOTA: configured cache byte limit would be exceeded"
                        )
                    os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return arrays, meta


def array_hash(arrays):
    return digest(
        {
            k: {
                "dtype": str(v.dtype),
                "shape": list(v.shape),
                "bytes_sha256": hashlib.sha256(
                    np.ascontiguousarray(v).tobytes()
                ).hexdigest(),
            }
            for k, v in arrays.items()
        }
    )

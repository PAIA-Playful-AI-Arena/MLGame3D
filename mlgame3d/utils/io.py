import dataclasses
import datetime
import json
import os


def write_json(path: str, data) -> None:
    """
    Serialize data to JSON at the given path.

    Dataclasses are serialized via dataclasses.asdict and datetimes as
    ISO 8601 UTC strings (e.g. "2026-08-17T05:03:11.105643Z").
    """
    def _default(o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, datetime.datetime):
            return o.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, default=_default, ensure_ascii=False))


def check_folder_existed_and_readable_or_create(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.isdir(path) and os.access(path, os.R_OK):
        return path
    else:
        raise NotADirectoryError(f"{path} is not a readable directory or does not exist")

import json
from typing import Iterator, Dict
from pathlib import Path

class JSONLStorage:
    """
    Хранение/чтение комментариев в формате JSON Lines.
    Каждый комментарий - отдельная JSON-строка: {"source":"youtube", "id": "...", "text":"...", ...}
    Это позволяет читать файл построчно без загрузки всего в память.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: Dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def iter(self) -> Iterator[Dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def count(self) -> int:
        c = 0
        with self.path.open("r", encoding="utf-8") as f:
            for _ in f:
                c += 1
        return c

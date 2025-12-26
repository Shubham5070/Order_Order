import redis
import json
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

r = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)

# ---------- Generic helpers ----------

def set_json(key: str, value: dict, ttl: int | None = None):
    data = json.dumps(value)
    r.set(key, data)
    if ttl:
        r.expire(key, ttl)


def get_json(key: str):
    data = r.get(key)
    return json.loads(data) if data else None


def delete(key: str):
    r.delete(key)

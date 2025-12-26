from services.redis_store import get_session

def ensure_ordering(session_id):
    session = get_session(session_id)
    return session and session.get("state") == "ORDERING"

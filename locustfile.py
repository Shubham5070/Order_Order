from locust import HttpUser, task, between
import random

class RestaurantUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        res = self.client.post(
            "/session/start",
            json={"table_id": "T1"}
        )
        self.session_id = res.json()["session_id"]

    @task(3)
    def add_item(self):
        msg = random.choice([
            "add paneer",
            "add coffee",
            "add one pizza"
        ])
        self.client.post(
            "/agent/chat",
            json={
                "session_id": self.session_id,
                "message": msg
            }
        )

    @task(1)
    def suggest(self):
        self.client.post(
            "/agent/chat",
            json={
                "session_id": self.session_id,
                "message": "suggest something spicy"
            }
        )

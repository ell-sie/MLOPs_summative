# locustfile.py

from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def predict(self):
        self.client.post("/predict/", json={"features": [5.1, 3.5, 1.4, 0.2]})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)

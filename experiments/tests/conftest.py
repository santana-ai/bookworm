import socket

import pytest

INTERNET_FAMILIES = (socket.AF_INET, socket.AF_INET6)


class NetworkAccessError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in INTERNET_FAMILIES:
            raise NetworkAccessError(f"tests must not open network connections ({address})")
        return original(self, address)

    def refused(*args, **kwargs):
        raise NetworkAccessError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", refused)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

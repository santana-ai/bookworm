import socket

import pytest
from conftest import NetworkAccessError


def test_tests_cannot_resolve_a_network_host() -> None:
    with pytest.raises(NetworkAccessError, match="resolve network hosts"):
        socket.create_connection(("example.com", 80), timeout=1)


def test_tests_cannot_connect_to_a_network_address() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client,
        pytest.raises(NetworkAccessError, match="open network connections"),
    ):
        client.connect(("192.0.2.1", 80))

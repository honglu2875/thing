import time

import thing


def another_fn(client):
    obj = 1
    client.catch(obj, server="localhost:8123").result()


def test_vars():
    obj = "abc"
    client = thing.Client(server_addr="localhost", server_port="8123")
    with thing.Server(port=8123) as server:
        client.catch(obj, server="localhost:8123").result()
        res = server.store.get_object_by_name("obj")
        assert res == obj
        obj = 2
        client.catch(obj, server="localhost:8123").result()
        res = server.store.get_object_by_name("obj")
        assert res == obj
        another_fn(client)
        res = server.store.get_object_by_name("obj")
        assert res == obj
        res = server.store.get_object_by_name("obj_1")
        assert res == 1

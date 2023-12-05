from thing.argument import ServerArguments, ServicerArguments


def test_args():
    args = {"port": 1234}
    servicer_args = ServicerArguments.from_args(args)
    assert servicer_args.port == 1234

    args["hello"] = 1234
    # Make sure there is no error
    servicer_args = ServicerArguments.from_args(args)
    assert servicer_args.port == 1234

    s = ServicerArguments(max_size=123)
    servicer_args = ServicerArguments.from_args(s, args)
    assert servicer_args.port == 1234
    assert servicer_args.max_size == 123

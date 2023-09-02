def convert_argpasere_to_dict(args):
    """
    Convert argparse to dict
    """
    args_dict = {}
    for arg in vars(args):
        args_dict[arg] = getattr(args, arg)
    return args_dict
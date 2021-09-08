


def main():
    parser = parse_opts()
    args = parser.parse_args()

    model = get_model(args)
    loss_fn = get_loss(args)

    print(model)
    print(loss_fn)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    
    from core.config.base_opts import parse_opts
    from core.model import get_model, get_loss

    main()
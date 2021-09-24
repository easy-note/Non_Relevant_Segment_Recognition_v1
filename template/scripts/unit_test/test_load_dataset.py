


def main():
    parser = parse_opts()
    args = parser.parse_args()

    robot_dataset = RobotDataset(args=args, state='train') 

    print(robot_dataset)
    print(len(robot_dataset))

    # lapa_dataset = LapaDataset(args)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    
    from core.config.base_opts import parse_opts
    from core.dataset import RobotDataset, LapaDataset

    main()
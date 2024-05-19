from training import Train
import argparse

def main(args):
    trainer = Train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
#!/usr/bin/env python3
import argparse

def gflops(ms, n):
    time_s = ms / 1000.0
    flops = 2 * (n ** 3)
    return flops / (time_s * 1e9)

def main():
    parser = argparse.ArgumentParser(
        description="Convert matmul time (ms) and matrix size (N) to GFLOPs."
    )
    parser.add_argument("ms", type=float, help="Time in milliseconds")
    parser.add_argument("n", type=int, help="Matrix dimension N for NxN multiply")

    args = parser.parse_args()

    result = gflops(args.ms, args.n)
    print(f"{result:.3f} GFLOPs")

if __name__ == "__main__":
    main()

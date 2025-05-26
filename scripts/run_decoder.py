from quickcodec_decoder.codec import run_parallel_decode
import argparse
import multiprocessing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="frames", help="Output directory")
    args = parser.parse_args()

    run_parallel_decode(args.video, args.fps, args.workers, args.output)

if __name__ == "__main__":
    main()

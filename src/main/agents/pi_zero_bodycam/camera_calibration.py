from pizero_capture_client import PiZeroCaptureClient
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera calibration using checkerboard.')
    parser.add_argument('host', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('xres', type=int)
    parser.add_argument('yres', type=int)
    parser.add_argument('n', type=int)

    args = parser.parse_args()
    capture_client = PiZeroCaptureClient(host=args.host,
                                         port=args.port,
                                         res=(args.xres, args.yres))
    capture_client.connect()
    capture_client.capture_img_sequence(args.n, 2)


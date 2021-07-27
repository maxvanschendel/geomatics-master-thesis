from ..pi_zero_bodycam.pizero_capture_client import PiZeroCaptureClient
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Raspberry Pi Zero video streaming over websocket.')
    parser.add_argument('host', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('xres', type=int)
    parser.add_argument('yres', type=int)
    parser.add_argument('n', type=int)

    args = parser.parse_args()
    client = PiZeroCaptureClient(host=args.host,
                                 port=args.port,
                                 res=(args.xres, args.yres))

    client.connect()

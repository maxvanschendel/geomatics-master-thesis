import argparse
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib.cm as cm

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multiple ArUco markers.')
    parser.add_argument('nx', type=int, help='Number of markers in horizontal direction.')
    parser.add_argument('ny', type=int, help='Number of markers in vertical direction.')
    parser.add_argument('res', type=int, help='Resolution of markers in pixels.')
    parser.add_argument('start', type=int, help='Marker starting index.')
    parser.add_argument('fn', type=str, help='Output filename and path.')
    args = parser.parse_args()

    fig = plt.figure()
    for i in range(args.nx * args.ny):
        # Generate marker at specified resolution, offset marker index by start index
        img = aruco.drawMarker(aruco_dict, i + args.start, args.res)

        # Add marker as subplot to matplotlib figure
        ax = fig.add_subplot(args.ny, args.nx, i+1)
        plt.imshow(img, cmap = cm.gray, interpolation = "none")
        ax.axis("off")

    plt.savefig(args.fn)
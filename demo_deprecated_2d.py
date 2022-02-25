import max_inner_ellipsoid
import numpy as np
import matplotlib.pyplot as plt


def demo2D():
    pts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.2, 0.4]])
    max_iterations = 10
    volume_increase_tol = 0.001
    dut = max_inner_ellipsoid.FindLargeEllipsoid(pts)
    P, q, r, outside_pts, inside_pts = \
        dut.search(max_iterations, volume_increase_tol)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_inner_ellipsoid.draw_ellipsoid(ax, P, q, r, outside_pts, inside_pts)
    fig.show()


if __name__ == "__main__":
    demo2D()

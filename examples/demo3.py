"""
Test a simple example in 3D.
"""
import max_inner_ellipsoid
import numpy as np
import matplotlib.pyplot as plt


def demo3D():
    pts = np.array([[1., 1., 1.],
                    [1., 1., -1.],
                    [1., -1., 1.],
                    [1., -1., -1.],
                    [-1., 1., 1.],
                    [-1., 1., -1.],
                    [-1., -1., 1.],
                    [-1., -1., -1],
                    [0., 0., 1.],
                    [0., 0., -1.],
                    [0., 1., 0],
                    [0., -1., 0.],
                    [1., 0., 0.],
                    [-1., 0., 0.]])
    max_iterations = 10
    convergence_tol = 0.001
    dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
    P, q, r = dut.search(np.array([0., 0., 0.1]), max_iterations,
                         convergence_tol)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_inner_ellipsoid.draw_ellipsoid(ax, P, q, r, pts, np.empty((0, 3)))
    fig.show()


if __name__ == "__main__":
    demo3D()

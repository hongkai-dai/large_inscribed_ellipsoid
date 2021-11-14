import max_inner_ellipsoid
import numpy as np

def demo2D():
    pts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.2, 0.4]])
    max_iterations = 10
    P, q, r, outside_pts, inside_pts = \
        max_inner_ellipsoid.find_large_ellipsoid(pts, max_iterations)


if __name__ == "__main__":
    demo2D()

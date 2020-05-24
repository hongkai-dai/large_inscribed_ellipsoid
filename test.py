import max_inner_ellipsoid

import numpy as np
import unittest

class TestFindEllipsoid(unittest.TestCase):
    def test_feasible(self):
        outside_pts = np.array(
            [[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.2, 0.4]])
        inside_pts = np.array([[0.4, 0.5], [0.5, 0.6], [0.7, 0.2]])

        A, b, hull = max_inner_ellipsoid.get_hull(outside_pts)

        P, q, r, lambda_val = max_inner_ellipsoid.find_ellipsoid(
            outside_pts, inside_pts, A, b)

        self.assertIsNotNone(P)
        # check inside_pts are actually inside the ellipsoid.
        np.testing.assert_array_less(
            np.sum(inside_pts.T * (P @ inside_pts.T), axis=0) + 
            2 * inside_pts @ q, (r + 1e-5) * np.ones(inside_pts.shape[0]))
        # check outside_pts are actually outside the ellipsoid
        np.testing.assert_array_less(
            (r - 1e-5) * np.ones(outside_pts.shape[0]),
            np.sum(outside_pts.T * (P @ outside_pts.T), axis=0) + 
            2 * outside_pts @ q, )
        # check that the ellipsoid is actually within the convex hull.
        np.testing.assert_array_less(0, lambda_val)
        for i in range(A.shape[0]):
            Qi = np.zeros((3, 3))
            Qi[:2, :2] = P
            Qi[:2, 2] = q - lambda_val[i] * A[i, :]/2
            Qi[:2, 2] = q - lambda_val[i] * A[i, :]/2
            Qi[2, 2] = lambda_val[i] * b[i] - r
            eig_value, _ = np.linalg.eig(Qi)
            np.testing.assert_array_less(0, eig_value)

    def test_infeasible(self):
        outside_pts = np.array(
            [[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.5, 0.5]])
        inside_pts = np.array([[0.3, 0.3], [0.7, 0.7]])

        A, b, hull = max_inner_ellipsoid.get_hull(outside_pts)
        P, q, r, lambda_val = max_inner_ellipsoid.find_ellipsoid(
            outside_pts, inside_pts, A, b)
        self.assertIsNone(P)
        self.assertIsNone(q)
        self.assertIsNone(r)
        self.assertIsNone(lambda_val)


if __name__ == "__main__":
    unittest.main()

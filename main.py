from scipy.spatial import ConvexHull
import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_hull(pts):
    dim = pts.shape[1]
    hull = ConvexHull(pts)
    A = hull.equations[:, 0:dim]
    b = hull.equations[:, dim]
    return A, -b, hull


def get_random_z(pts):
    """
    Now find a random point z that is in the convex hull of the points
    v₁, ..., vₙ. We will constrain that the ellipsoid contains the point z.
    Note that our ellipsoid depends on where the point z is. For a different
    z point, we might end up with the different ellipsoid.
    """
    num_pts = pts.shape[0]
    pts_weights = np.random.uniform(0, 1, num_pts)
    z = (pts_weights @ pts) / np.sum(pts_weights)
    return z


def maximal_inner_ellipsoid(outside_pts, A, b, z):
    """
    For a given sets of points v₁, ..., vₙ, find the ellipsoid with a large
    volume, satisfying three constraints:
    1. The ellipsoid is within the convex hull of these points.
    2. The ellipsoid doesn't contain any of the points.
    3. The ellipsoid contains the point z
    This ellipsoid is parameterized as {x | xᵀPx + 2qᵀx ≤ r }.
    We find this ellipsoid by solving a semidefinite programming problem.
    @param outside_pts outside_pts[i, :] is the i'th point vᵢ.
    @param A, b The convex hull of v₁, ..., vₙ is Ax<=b
    @param z A point that should be contained inside the ellipsoid.
    @return (P, q, r) The parameterization of this ellipsoid.
    """
    assert(isinstance(outside_pts, np.ndarray))
    (num_outside_pts, dim) = outside_pts.shape

    constraints = []
    P = cp.Variable((dim, dim), symmetric=True)
    q = cp.Variable(dim)
    r = cp.Variable()

    # Impose the constraint that v₁, ..., vₙ are all outside of the ellipsoid.
    for i in range(num_outside_pts):
        constraints.append(
            outside_pts[i, :] @ (P @ outside_pts[i, :]) + 2 * q @ outside_pts[i, :] >= r)
    # P is strictly positive definite.
    epsilon = 1e-6
    constraints.append(P - epsilon * np.eye(dim) >> 0)

    # Add the constraint that the ellipsoid contains z.
    constraints.append(z @ (P @ z) + 2 * q @ z <= r)

    # Now add the constraint that the ellipsoid is in the convex hull Ax<=b.
    # Using s-lemma, we know that the constraint is
    # ∃ λᵢ > 0,
    # s.t [P            q -λᵢaᵢ/2]  is positive semidefinite.
    #     [(q-λᵢaᵢ/2)ᵀ     λᵢbᵢ-r]
    num_faces = A.shape[0]
    lambda_var = cp.Variable(num_faces)
    constraints.append(lambda_var >= 0)
    Q = [None] * num_faces
    for i in range(num_faces):
        Q[i] = cp.Variable((dim+1, dim+1), PSD=True)
        constraints.append(Q[i][:dim, :dim] == P)
        constraints.append(Q[i][:dim, dim] == q - lambda_var[i] * A[i, :]/2)
        constraints.append(Q[i][-1, -1] == lambda_var[i] * b[i] - r)

    # Add the constraint that
    # [P    q] is positive semidefinite
    # [qᵀ 1-r]
    S = cp.Variable((dim+1, dim+1), PSD=True)
    constraints.append(S[:dim, :dim] == P)
    constraints.append(S[:dim, dim] == q)
    constraints.append(S[-1, -1] == 1-r)

    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    #prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    if prob.status == 'optimal':
        P_val = P.value
        q_val = q.value
        r_val = r.value
        return P_val, q_val, r_val
    else:
        raise Exception("optimization failed.")


def draw_ellipsoid(P, q, r, outside_pts, z):
    """
    Draw an ellipsoid defined as {x | xᵀPx + 2qᵀx ≤ r }
    This ellipsoid is equivalent to
    |Lx + L⁻¹q| ≤ √(r + qᵀP⁻¹q)
    where L is the symmetric matrix satisfying L * L = P
    """
    fig = plt.figure()
    dim = P.shape[0]
    L = scipy.linalg.sqrtm(P)
    print(f"P={P}")
    print(f"L={L}")
    radius = np.sqrt(r + q@(np.linalg.solve(P, q)))
    if dim == 2:
        # first compute the points on the unit sphere
        theta = np.linspace(0, 2 * np.pi, 200)
        sphere_pts = np.vstack((np.cos(theta), np.sin(theta)))
        ellipsoid_pts = np.linalg.solve(
            L, radius * sphere_pts - (np.linalg.solve(L, q)).reshape((2, -1)))
        ax = fig.add_subplot(111)
        ax.plot(ellipsoid_pts[0, :], ellipsoid_pts[1, :], c='blue')

        ax.scatter(outside_pts[:, 0], outside_pts[:, 1], c='red')
        ax.scatter(z[0], z[1], s=20, c='green')
        ax.axis('equal')
        plt.show()
    if dim == 3:
        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)

        sphere_pts_x = np.outer(np.sin(u), np.sin(v))
        sphere_pts_y = np.outer(np.sin(u), np.cos(v))
        sphere_pts_z = np.outer(np.cos(u), np.ones_like(v))
        sphere_pts = np.vstack((
            sphere_pts_x.reshape((1, -1)), sphere_pts_y.reshape((1, -1)),
            sphere_pts_z.reshape((1, -1))))
        ellipsoid_pts = np.linalg.solve(
            L, radius * sphere_pts - (np.linalg.solve(L, q)).reshape((3, -1)))
        ax = plt.axes(projection='3d')
        ellipsoid_pts_x = ellipsoid_pts[0,:].reshape(sphere_pts_x.shape)
        ellipsoid_pts_y = ellipsoid_pts[1,:].reshape(sphere_pts_y.shape)
        ellipsoid_pts_z = ellipsoid_pts[2,:].reshape(sphere_pts_z.shape)
        ax.plot_wireframe(ellipsoid_pts_x, ellipsoid_pts_y, ellipsoid_pts_z)
        ax.scatter(outside_pts[:, 0], outside_pts[:, 1], outside_pts[:, 2], c='red')
        ax.scatter(z[0], z[1], z[2], s=20, c='green')
        ax.axis('equal')
        plt.show()


if __name__ == "__main__":
    dim = 2
    # outside_pts = np.vstack((3 * np.cos(np.linspace(0, 2*np.pi, 10)), np.sin(np.linspace(0, 2*np.pi, 10)))).T
    outside_pts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.3, 0.8], [0.2, 0.1], [0.7, 0.4], [0.1, 0.4], [0.3, 0.5]])
    # outside_pts = np.array([[0., 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [0, 0.2, 0.9], [0.1, 0.3, 0.6]])
    z = get_random_z(outside_pts)
    # z = np.array([0.5, 0.5, 0.5])
    A, b, hull = get_hull(outside_pts)
    P, q, r = maximal_inner_ellipsoid(outside_pts, A, b, z)
    print(f"trace(P): {np.trace(P)}")
    print(f"volume: {(r + q @ (np.linalg.solve(P, q)))/np.power(np.linalg.det(P), 1.0/dim)}")
    draw_ellipsoid(P, q, r, outside_pts, z)


from scipy.spatial import ConvexHull, Delaunay
import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D  # noqa
# import warnings


def get_hull(pts):
    dim = pts.shape[1]
    hull = ConvexHull(pts)
    A = hull.equations[:, 0:dim]
    b = hull.equations[:, dim]
    return A, -b, hull


def compute_ellipsoid_volume(P, q, r):
    """
    The volume of the ellipsoid xᵀPx + 2qᵀx ≤ r is proportional to
    power(r + qᵀP⁻¹q, dim/2) / sqrt(det(P))
    We return this number.
    """
    dim = P.shape[0]
    return np.power((r + q @ np.linalg.solve(P, q)), dim / 2) /\
        np.sqrt(np.linalg.det(P))


def uniform_sample_from_convex_hull(deln, dim, n):
    """
    Uniformly sample n points in the convex hull Ax<=b
    This is copied from
    https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    @param deln Delaunay of the convex hull.
    """
    vols = np.abs(np.linalg.det(deln[:, :dim, :] - deln[:, dim:, :]))\
        / np.math.factorial(dim)
    sample = np.random.choice(len(vols), size=n, p=vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample],
                     dirichlet.rvs([1] * (dim + 1), size=n))


def centered_sample_from_convex_hull(pts):
    """
    Sample a random point z that is in the convex hull of the points
    v₁, ..., vₙ. z = (w₁v₁ + ... + wₙvₙ) / (w₁ + ... + wₙ) where wᵢ are all
    uniformly sampled from [0, 1]. Notice that by central limit theorem, the
    distribution of this sample is centered around the convex hull center, and
    also with small variance when the number of points are large.
    """
    num_pts = pts.shape[0]
    pts_weights = np.random.uniform(0, 1, num_pts)
    z = (pts_weights @ pts) / np.sum(pts_weights)
    return z


def find_ellipsoid(outside_pts, inside_pts, A, b, *, verbose=False):
    """
    For a given sets of points v₁, ..., vₙ, find the ellipsoid satisfying
    three constraints:
    1. The ellipsoid is within the convex hull of these points.
    2. The ellipsoid doesn't contain any of the points.
    3. The ellipsoid contains all the points in @p inside_pts
    This ellipsoid is parameterized as {x | xᵀPx + 2qᵀx ≤ r }.
    We find this ellipsoid by solving a semidefinite programming problem.
    @param outside_pts outside_pts[i, :] is the i'th point vᵢ. The point vᵢ
    must be outside of the ellipsoid.
    @param inside_pts inside_pts[i, :] is the i'th point that must be inside
    the ellipsoid.
    @param A, b The convex hull of v₁, ..., vₙ is Ax<=b
    @return (P, q, r, λ) P, q, r are the parameterization of this ellipsoid. λ
    is the slack variable used in constraining the ellipsoid inside the convex
    hull Ax <= b. If the problem is infeasible, then returns
    None, None, None, None
    """
    assert (isinstance(outside_pts, np.ndarray))
    (num_outside_pts, dim) = outside_pts.shape
    assert (isinstance(inside_pts, np.ndarray))
    assert (inside_pts.shape[1] == dim)
    num_inside_pts = inside_pts.shape[0]

    constraints = []
    P = cp.Variable((dim, dim), symmetric=True)
    q = cp.Variable(dim)
    r = cp.Variable()

    # Impose the constraint that v₁, ..., vₙ are all outside of the ellipsoid.
    for i in range(num_outside_pts):
        constraints.append(outside_pts[i, :] @ (P @ outside_pts[i, :]) +
                           2 * q @ outside_pts[i, :] >= r)
    # P is strictly positive definite.
    epsilon = 1e-6
    constraints.append(P - epsilon * np.eye(dim) >> 0)

    # Add the constraint that the ellipsoid contains @p inside_pts.
    for i in range(num_inside_pts):
        constraints.append(inside_pts[i, :] @ (P @ inside_pts[i, :]) +
                           2 * q @ inside_pts[i, :] <= r)

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
        Q[i] = cp.Variable((dim + 1, dim + 1), PSD=True)
        constraints.append(Q[i][:dim, :dim] == P)
        constraints.append(Q[i][:dim, dim] == q - lambda_var[i] * A[i, :] / 2)
        constraints.append(Q[i][-1, -1] == lambda_var[i] * b[i] - r)

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(verbose=verbose)
    except cp.error.SolverError:
        return None, None, None, None

    if prob.status == 'optimal':
        P_val = P.value
        q_val = q.value
        r_val = r.value
        lambda_val = lambda_var.value
        return P_val, q_val, r_val, lambda_val
    else:
        return None, None, None, None


def draw_ellipsoid(P, q, r, outside_pts, inside_pts):
    """
    Draw an ellipsoid defined as {x | xᵀPx + 2qᵀx ≤ r }
    This ellipsoid is equivalent to
    |Lx + L⁻¹q| ≤ √(r + qᵀP⁻¹q)
    where L is the symmetric matrix satisfying L * L = P
    """
    fig = plt.figure()
    dim = P.shape[0]
    L = scipy.linalg.sqrtm(P)
    radius = np.sqrt(r + q @ (np.linalg.solve(P, q)))
    if dim == 2:
        # first compute the points on the unit sphere
        theta = np.linspace(0, 2 * np.pi, 200)
        sphere_pts = np.vstack((np.cos(theta), np.sin(theta)))
        ellipsoid_pts = np.linalg.solve(
            L, radius * sphere_pts - (np.linalg.solve(L, q)).reshape((2, -1)))
        ax = fig.add_subplot(111)
        ax.plot(ellipsoid_pts[0, :], ellipsoid_pts[1, :], c='blue')

        ax.scatter(outside_pts[:, 0], outside_pts[:, 1], c='red')
        ax.scatter(inside_pts[:, 0], inside_pts[:, 1], s=20, c='green')
        ax.axis('equal')
        plt.show()
    if dim == 3:
        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)

        sphere_pts_x = np.outer(np.sin(u), np.sin(v))
        sphere_pts_y = np.outer(np.sin(u), np.cos(v))
        sphere_pts_z = np.outer(np.cos(u), np.ones_like(v))
        sphere_pts = np.vstack((sphere_pts_x.reshape(
            (1, -1)), sphere_pts_y.reshape(
                (1, -1)), sphere_pts_z.reshape((1, -1))))
        ellipsoid_pts = np.linalg.solve(
            L, radius * sphere_pts - (np.linalg.solve(L, q)).reshape((3, -1)))
        ax = plt.axes(projection='3d')
        ellipsoid_pts_x = ellipsoid_pts[0, :].reshape(sphere_pts_x.shape)
        ellipsoid_pts_y = ellipsoid_pts[1, :].reshape(sphere_pts_y.shape)
        ellipsoid_pts_z = ellipsoid_pts[2, :].reshape(sphere_pts_z.shape)
        ax.plot_wireframe(ellipsoid_pts_x, ellipsoid_pts_y, ellipsoid_pts_z)
        ax.scatter(outside_pts[:, 0],
                   outside_pts[:, 1],
                   outside_pts[:, 2],
                   c='red')
        ax.scatter(inside_pts[:, 0],
                   inside_pts[:, 1],
                   inside_pts[:, 2],
                   s=20,
                   c='green')
        if dim == 2:
            # 3D plot doesn't support equal axis yet. Only 2D plot can.
            ax.axis('equal')
        plt.show()


def inside_ellipsoid(pts, P, q, r):
    """
    For a batch of points, determine if they are inside the ellipsoid
    {x | xᵀPx + 2qᵀx ≤ r }

    Args:
      pts: pts is of size num_points x dim.

    Return:
      flag: flag is a numpy array of size num_points, where flag[i] is true if
      and only if pts[i] is inside the ellipsoid.
    """
    return np.sum(pts.T * (P @ pts.T), axis=0) + 2 * pts @ q <= r


class SearchLargeEllipsoid:
    """
    This class finds a large ellipsoid within the convex hull of @p pts but
    not containing any point in @p pts.
    It finds such ellipsoid through solving a sequence of semidefinite
    programming (SDP) problems.
    Mathematically we formulate the ellipsoid as
    {x | xᵀPx + 2qᵀx ≤ r}
    where P, q, r are parameters of the ellipsoid.
    If we denote the i'th point pts[i] as vᵢ, then the constraint that vᵢ
    is not in the ellipsoid is
    vᵢᵀPvᵢ+2qᵀvᵢ ≥ r
    If we denote the convex hull of @pts as the polytope
    ConvexHull(pts) = {x | Cx ≤ d},
    then the constraint that the ellipsoid is within the convex hull is
    ∃ λᵢ≥ 0, s.t ⌈ P          q−0.5λᵢcᵢ⌉ is positive semidefinite.
                 ⌊(q−0.5λᵢcᵢ)ᵀ   λᵢdᵢ−r⌋
    The volume of the ellipsoid is proportional to
    sqrt((r + qᵀP⁻¹q)ⁿ/det(P)). Maximizing this volume is equivalent to
    maximizing its logarithm n*log(r + qᵀP⁻¹q) - log(det(P))
    Hence we can formulate the following optimization problem
    max n*log(r + qᵀP⁻¹q) - log(det(P))
    s.t vᵢᵀPvᵢ+2qᵀvᵢ ≥ r
        ⌈ P          q−0.5λᵢcᵢ⌉ is positive semidefinite.
        ⌊(q−0.5λᵢcᵢ)ᵀ   λᵢdᵢ−r⌋
        λᵢ≥ 0

    All the constraints are convex in the decision variables P, q, r, λ.
    The cost function isn't a concave function, and we will linearize the cost
    function, and maximize this linearized cost within a trust region in each
    iteration.
    """
    def __init__(self, pts):
        """
        Args:
          pts: pts[i, :] is the i'th point that has to be outside of the
          ellipsoid.
        """
        self.pts = pts
        self.dim = pts.shape[1]
        # Compute the convex hull of pts.
        self.C, self.d, self.hull = get_hull(self.pts)
        hull_vertices = pts[self.hull.vertices]
        self.deln = hull_vertices[Delaunay(hull_vertices).simplices]

    def _find_initial_ellipsoid(self, pt: np.ndarray):
        """
        Find an ellipsoid around @p pt. This ellipsoid is contained within the
        convex hull of self.pts and do not contain any of self.pts.

        One way of finding such an ellipsoid is to first find define a polytope
        as 𝒫₀ = {x | (vᵢ − pt)ᵀx ≤ (vᵢ−pt)ᵀvᵢ}, namely for each vᵢ in self.pts,
        consider the plane that passes vᵢ whose normal points along the
        direction of vᵢ-pt. Then compute the maximal ellipsoid {Eu+f | |u|<=1}
        contained within 𝒫₀. We can compute this ellipsoid through the SDP
        max log det(E)
        s.t E is psd
            |(vᵢ − pt)ᵀE| ≤ (vᵢ−pt)ᵀ(vᵢ−f)
            |cᵢᵀE|≤ dᵢ − cᵢᵀf

        Args:
          pt: A seed point. The returned ellipsoid is contained in a polytope
          𝒫₀, this polytope contains @p pt. But the returned ellipsoid may not
          contain @p pt.
        Return:
          P0, q0, r0. The returned ellipsoid is parameterized as
          { x | xᵀP₀x + 2q₀ᵀx ≤ r₀}
        """
        assert (pt.shape == (self.dim, ))
        assert ((self.C @ pt <= self.d).all())
        E = cp.Variable((self.dim, self.dim), PSD=True)
        f = cp.Variable(self.dim)
        soc_constraints1 = [
            cp.SOC((self.pts[i] - pt) @ (self.pts[i] - f),
                   E @ (self.pts[i] - pt)) for i in range(self.pts.shape[0])
        ]
        soc_constraints2 = [
            cp.SOC(self.d[i] - self.C[i] @ f, self.C[i] @ E)
            for i in range(self.C.shape[0])
        ]
        prob = cp.Problem(cp.Maximize(cp.log_det(E)),
                          soc_constraints1 + soc_constraints2)
        prob.solve()
        E_val = E.value
        f_val = f.value
        P0 = np.linalg.inv(E_val.T @ E_val)
        q0 = -P0 @ f_val
        r0 = 1 - f_val.dot(P0 @ f_val)
        return P0, q0, r0

    def _search_within_region(self, P_curr, q_curr, r_curr, delta):
        """
        Solve the original optimization problem with a linear approximation of
        the objective (where we linearize the objective arround P_curr,
        q_curr, r_curr), and within a trust region of radius delta.
        """
        constraints = []
        P = cp.Variable((self.dim, self.dim), symmetric=True)
        q = cp.Variable(self.dim)
        r = cp.Variable()

        # Impose the constraint that v₁, ..., vₙ are all outside of the
        # ellipsoid.
        for i in range(self.pts.shape[0]):
            constraints.append(self.pts[i, :] @ (P @ self.pts[i, :]) +
                               2 * q @ self.pts[i, :] >= r)
        # P is strictly positive definite.
        epsilon = 1e-6
        constraints.append(P - epsilon * np.eye(self.dim) >> 0)
        # Impose the constraint that the ellipsoid is within the convex hull
        # of self.pts
        num_faces = self.C.shape[0]
        lambda_var = cp.Variable(num_faces)
        constraints.append(lambda_var >= 0)
        Q = [None] * num_faces
        for i in range(num_faces):
            Q[i] = cp.Variable((self.dim + 1, self.dim + 1), PSD=True)
            constraints.append(Q[i][:self.dim, :self.dim] == P)
            constraints.append(Q[i][:self.dim, self.dim] == q -
                               lambda_var[i] * self.C[i, :] / 2)
            constraints.append(Q[i][-1, -1] == lambda_var[i] * self.d[i] - r)

        # Impose the constraint that this new ellipsoid contains the center of
        # the previous ellipsoid.
        ellipsoid_center_curr = -np.linalg.solve(P_curr, q_curr)
        constraints.append(
            ellipsoid_center_curr @ (P @ ellipsoid_center_curr) +
            2 * q @ ellipsoid_center_curr >= r)

        # Impose the trust region constraint
        # |P - P_curr|² + |q - q_curr|² + |r-r_curr|² <= delta
        if (not np.isinf(delta)):
            assert (delta > 0)
            constraints.append(
                np.trace((P - P_curr).T @ (P - P_curr)) +
                (q - q_curr).dot(q - q_curr) + (r - r_curr)**2 <= delta)

        # Now add the linearized objective
        # n * trace([r_curr q_currᵀ]⁻¹ * [r qᵀ]) - (n+1) * trace(P_curr⁻¹*P)
        #           [q_curr -P_curr]     [q -P]
        # Denote X = [r_curr q_currᵀ]
        #            [q_curr -P_curr]
        X = np.empty((self.dim + 1, self.dim + 1))
        X[0, 0] = r_curr
        X[0, 1:] = q_curr.T
        X[1:, 0] = q_curr
        X[1:, 1:] = -P_curr
        X_inv = np.linalg.inv(X)
        objective = self.dim * (X_inv[0, 0] * r + 2 * X_inv[0, 1:].dot(q) +
                                np.trace(X_inv[1:, 1:] @ (-P))) -\
            (self.dim + 1) * np.trace(np.linalg.inv(P_curr)  @ P)
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve()


class FindLargeEllipsoid:
    """
    We find a large ellipsoid within the convex hull of @p pts but not
    containing any point in @p pts.
    The algorithm proceeds iteratively
    1. Start with outside_pts = pts, inside_pts = z where z is a random point
       in the convex hull of @p outside_pts.
    2. while num_iter < max_iterations
    3.   Solve an SDP to find an ellipsoid that is within the convex hull of
         @p pts, not containing any outside_pts, but contains all inside_pts.
    4.   If the SDP in the previous step is infeasible, then remove z from
         inside_pts, and append it to the outside_pts.
    5.   Randomly sample a point in the convex hull of @p pts, if this point is
         outside of the current ellipsoid, then append it to inside_pts.
    6.   num_iter += 1
    When the iterations limit is reached, we report the ellipsoid with the
    maximal volume.
    @param pts pts[i, :] is the i'th points that has to be outside of the
    ellipsoid.
    @param max_iterations The iterations limit.
    @param volume_increase_tol If the increase of the ellipsoid volume is
    no larger than this threshold, then stop.
    @return (P, q, r, inside_pts, outside_pts) The largest ellipsoid is
    parameterized as {x | xᵀPx + 2qᵀx ≤ r }
    """
    def __init__(self, pts):
        # warnings.warn("This class finds a large inscribed ellipsoid " +
        # "through a stochastic procedure. It is better to use the class " +
        # "SearchLargeEllipsoid which is deterministic")
        self.pts = pts
        self.dim = self.pts.shape[1]
        self.A, self.b, self.hull = get_hull(self.pts)
        hull_vertices = pts[self.hull.vertices]
        self.deln = hull_vertices[Delaunay(hull_vertices).simplices]
        # In each iteration, we randomly sample num_sample_pts inside the
        # convex hull self.hull. If all these sample points are inside the
        # ellipsoid, then we think the ellispoid is large enough, and
        # terminate the search.
        self.num_sample_pts = 20

    def search(self, max_iterations, volume_increase_tol, *, verbose=False):
        """
        Search the ellipsoid until either hitting the max_iterations, or the
        increase in the volume is smaller than volume_increase_tol.

        Return:
          P, q, r: The best ellipsoid found as {x | xᵀPx + 2qᵀx ≤ r}
          outside_pts, inside_pts: The points discovered during the search
          process. inside_pts are all inside the ellipsoid, outside_pts are
          all outside the ellipsoid.
        """
        # Grow the ellipsoid from a point sampled in the center of the convex
        # hull.
        candidate_pt = centered_sample_from_convex_hull(self.pts)
        inside_pts = candidate_pt.reshape((1, -1))
        P, q, r, lambda_val = find_ellipsoid(self.pts,
                                             inside_pts,
                                             self.A,
                                             self.b,
                                             verbose=verbose)
        if P is None:
            raise Exception("Failed in the first iteration. Check which " +
                            "solver is used. I highly recommend installing" +
                            " Mosek solver, as the default solver (SCS) " +
                            "coming with CVXPY often fails due to " +
                            "numerical issues.")
        return self.search_from(P, q, r, self.pts, inside_pts,
                                max_iterations - 1, volume_increase_tol)

    def search_from(self, P, q, r, outside_pts, inside_pts, max_iterations,
                    volume_increase_tol):
        """
        Start the search from an initial ellipsoid xᵀPx + 2qᵀx ≤ r, where
        inside_pts are all inside xᵀPx + 2qᵀx ≤ r, outside_pts are all outside
        xᵀPx + 2qᵀx ≤ r.
        A typical use of this function is that after calling search() and get
        the returned results, you still want to improve the returned results;
        then you can pass that result as the input P, q, r, inside_pts,
        outside_pts to this function.

        Args:
          inside_pts it contains the points inside the input ellipsoid.
          outide_pts it contains the points outside the input ellipsoid.
        """
        assert (np.all(inside_ellipsoid(inside_pts, P, q, r)))
        assert (not np.any(inside_ellipsoid(outside_pts, P, q, r)))

        num_iter = 0
        max_ellipsoid_volume = compute_ellipsoid_volume(P, q, r)
        P_best = P
        q_best = q
        r_best = r
        while num_iter < max_iterations:
            # Now take a new sample that is outside of the ellipsoid.
            sample_pts = uniform_sample_from_convex_hull(
                self.deln, self.dim, self.num_sample_pts)
            is_in_ellipsoid = inside_ellipsoid(sample_pts, P_best, q_best,
                                               r_best)
            if np.all(is_in_ellipsoid):
                return P_best, q_best, r_best, outside_pts, inside_pts
            else:
                # candidate_pt is the point outside of the current best
                # ellipsoid. Check if we can find a new ellipsoid that covers
                # inside_pts and z.
                candidate_pt = sample_pts[np.where(~is_in_ellipsoid)[0][0], :]
                P, q, r, lambda_val = find_ellipsoid(
                    outside_pts, np.vstack((inside_pts, candidate_pt)), self.A,
                    self.b)
                if P is None:
                    # Cannot find the ellipsoid that covers both inside_pts
                    # and candidate_pt. Add candidate_pt to outside_pts.
                    outside_pts = np.vstack((outside_pts, candidate_pt))
                else:
                    volume = compute_ellipsoid_volume(P, q, r)
                    if volume > max_ellipsoid_volume:
                        P_best = P
                        q_best = q
                        r_best = r
                        if volume - max_ellipsoid_volume <= \
                                volume_increase_tol:
                            return P_best, q_best, r_best, inside_pts,\
                                outside_pts
                        max_ellipsoid_volume = volume
                        inside_pts = np.vstack((inside_pts, candidate_pt))
                num_iter += 1
        return P_best, q_best, r_best, outside_pts, inside_pts


def find_large_ellipsoid(pts, max_iterations, volume_increase_tol):
    """
    We find a large ellipsoid within the convex hull of @p pts but not
    containing any point in @p pts.
    The algorithm proceeds iteratively
    1. Start with outside_pts = pts, inside_pts = z where z is a random point
       in the convex hull of @p outside_pts.
    2. while num_iter < max_iterations
    3.   Solve an SDP to find an ellipsoid that is within the convex hull of
         @p pts, not containing any outside_pts, but contains all inside_pts.
    4.   If the SDP in the previous step is infeasible, then remove z from
         inside_pts, and append it to the outside_pts.
    5.   Randomly sample a point in the convex hull of @p pts, if this point is
         outside of the current ellipsoid, then append it to inside_pts.
    6.   num_iter += 1
    When the iterations limit is reached, we report the ellipsoid with the
    maximal volume.
    @param pts pts[i, :] is the i'th points that has to be outside of the
    ellipsoid.
    @param max_iterations The iterations limit.
    @param volume_increase_tol If the increase of the ellipsoid volume is
    no larger than this threshold, then stop.
    @return (P, q, r, inside_pts, outside_pts) The largest ellipsoid is
    parameterized as {x | xᵀPx + 2qᵀx ≤ r }
    """
    raise Warning("This function is deprecated, please use " +
                  "FindLargeEllpsoid class using its method search()")
    return FindLargeEllipsoid(pts).search(max_iterations, volume_increase_tol)

# Find a large inscribed ellipsoid.

Given a point cloud containing points v₁, ..., vₙ, we want to find a large ellipsoid satisfying:

1. The ellipsoid is contained within the convex hull of the point cloud.
2. The ellipsoid doesn't contain any of the point vᵢ in the point cloud.

To find such an ellipsoid, we solve a sequence of semidefinite programming problems. The complete algorithm is explained in the [doc](https://github.com/hongkai-dai/large_inscribed_ellipsoid/blob/master/doc/formulation.pdf).

An outdated stochastic algorithm is given in my [stackoverflow answer](https://stackoverflow.com/a/61905793/1973861)

Here is a simple 2D example, where the point clouds contain these five red points.![simple example result](https://media.giphy.com/media/d2ZDDyyTwGIxSpVqom/giphy.gif)

To get started, run
```
python3 examples/demo_2d_toy.py
```

## Solvers
Since we need to solve a sequence of semidefinite programming (SDP) problems, the user should install SDP solvers. By default cvxpy will install and use the open-source SCS solver. My experience is that SCS doesn't solve this SDP problem reliably. I highly recommend [Mosek](https://www.mosek.com/downloads/) solver.

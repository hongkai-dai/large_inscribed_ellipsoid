# Find a large inscribed ellipsoid.

Given a point cloud containing points v₁, ..., vₙ, we want to find a large ellipsoid satisfying:

1. The ellipsoid is contained within the convex hull of the point cloud.
2. The ellipsoid doesn't contain any of the point vᵢ in the point cloud.

To find such an ellipsoid, we solve a sequence of semidefinite programming problems. The complete algorithm is explained in my [stackoverflow answer](https://stackoverflow.com/a/61905793/1973861)

Here is a simple 2D example, where the point clouds contain these five red points.![simple example result](https://media.giphy.com/media/QUMK4s9nzveUmHYxVe/giphy.gif)

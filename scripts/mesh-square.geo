// Generate square mesh on [0,1]^2.
// Usage:
//   gmsh mesh-square.geo -setnumber h 0.1

DefineConstant[ h = {0.1, Name "Mesh size"} ];

Point(1) = {0, 0, 0, h};

Extrude {1, 0, 0} { Point{1}; Layers{1.0 / h}; }
Extrude {0, 1, 0} { Line{1};  Layers{1.0 / h}; }

Physical Line(0) = {3};
Physical Line(1) = {4};
Physical Line(2) = {1};
Physical Line(3) = {2};

Physical Surface(10) = {5};

Mesh 2;

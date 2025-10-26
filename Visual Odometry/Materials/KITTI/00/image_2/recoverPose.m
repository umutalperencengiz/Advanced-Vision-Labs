% relativeCameraPose Compute relative up-to-scale pose of calibrated camera
%   [relativeOrientation, relativeLocation] = relativeCameraPose(M,
%   cameraParams, inlierPoints1, inlierPoints2) returns the orientation and
%   up-to-scale location of a calibrated camera relative to its previous
%   pose. relativeLocation is always a unit vector.
%
%   M is an essential or fundamental 3-by-3 matrix, or a projective2d
%   object containing a homography matrix. cameraParams is a
%   cameraParameters or cameraIntrinsics object. inlierPoints1 and
%   inlierPoints2 are matching inlier points from the two views
%   corresponding to the two poses. M, inlierPoints1, and inlierPoints2 are
%   returned by the estimateEssentialMatrix, or estimateFundamentalMatrix,
%   or estimateGeometricTransform functions. relativeOrientation is a
%   3-by-3-by-N rotation matrix. relativeLocation is a N-by-3 matrix with a
%   unit vector at each row. N is one if M is an essential or fundamental
%   matrix, and is up to two if M is a projective2d object.
%
%   [...] = relativeCameraPose(M, cameraParams1, cameraParams2, inlierPoints1, inlierPoints2)
%   returns the orientation and location of camera 2 relative to camera 1.
%   cameraParams1 and cameraParams2 are cameraParameters or
%   cameraIntrinsics objects containing the parameters of camera 1 and
%   camera 2 respectively.
%
%   [..., validPointsFraction] = relativeCameraPose(...) additionally
%   returns the fraction of the inlier points that project in front of both
%   cameras. If this fraction is too small (e. g. less than 0.9), that can
%   indicate that the input matrix M is unreliable.
%
%   Notes
%   -----
%   - You can compute the camera extrinsics as follows:
%     [rotationMatrix, translationVector] = cameraPoseToExtrinsics(
%       relativeOrientation, relativeLocation)
%
%   - The relativeCameraPose function uses inlierPoints1 and inlierPoints2 to
%     determine which one of the multiple possible solutions is physically
%     realizable. If the input M is a projective2d object, there could be
%     up to two solutions that are equally realizable.
%
%    Class Support
%    -------------
%    M must be double or single. cameraParams must be a cameraParameters or
%    cameraIntrinsics object. inlierPoints1 and inlierPoints2 can be
%    double, single, or any of the point feature types. location and
%    orientation are the same class as M.
%
%  Example: Structure from motion from two views
%  ---------------------------------------------
%  % This example shows you how to build a point cloud based on features
%  % matched between two images of an object.
%  % <a href="matlab:web(fullfile(matlabroot,'toolbox','vision','visiondemos','html','StructureFromMotionExample.html'))">View example</a>
%
%  See also estimateWorldCameraPose, cameraCalibrator, estimateCameraParameters, 
%    estimateEssentialMatrix, estimateFundamentalMatrix, cameraMatrix,
%    plotCamera, triangulate, triangulateMultiview, cameraPoseToExtrinsics,
%    extrinsics, estimateGeometricTransform

% Copyright 2015 The MathWorks, Inc.

% References:
% -----------
% [1] R. Hartley, A. Zisserman, "Multiple View Geometry in Computer
% Vision," Cambridge University Press, 2003.
%
% [2] R. Hartley, P. Sturm. "Triangulation." Computer Vision and
% Image Understanding. Vol 68, No. 2, November 1997, pp. 146-157
%
% [3] O. Faugeras and F. Lustman, �Motion and structure from motion in a
% piecewise planar environment�, in International Journal of Pattern
% Recognition and Artificial Intelligence, 2(3):485�508, 1988.

%#codegen

function [orientation, location, inliers] = ...
    recoverPose(F, varargin)

[cameraParams1, cameraParams2, inlierPoints1, inlierPoints2] = ...
    parseInputs(F, varargin{:});

if isa(cameraParams1, 'cameraIntrinsics')
    cameraParams1 = cameraParams1.CameraParameters;
end

if isa(cameraParams2, 'cameraIntrinsics')
    cameraParams2 = cameraParams2.CameraParameters;
end

K1 = cameraParams1.IntrinsicMatrix;
K2 = cameraParams2.IntrinsicMatrix;

if ~isa(F, 'projective2d')
    if isFundamentalMatrix(F, inlierPoints1, inlierPoints2, K1, K2)
        % Compute the essential matrix
        E = K2 * F * K1';
    else
        % We already have the essential matrix
        E = F;
    end
    [Rs, Ts] = vision.internal.calibration.decomposeEssentialMatrix(E);
else
     H = (K1 * F.T / K2)';
     [Rs, Ts] = vision.internal.calibration.decomposeHomographyMatrix(H);
end

[R, t, inliers] = chooseRealizableSolution(Rs, Ts, cameraParams1, cameraParams2, inlierPoints1, ...
    inlierPoints2);

% R and t are currently the transformation from camera1's coordinates into
% camera2's coordinates. To find the location and orientation of camera2 in
% camera1's coordinates we must take their inverse.
orientation = R;
location = t;
if size(location, 1) == 1
    orientation = R(:,:,1)';
    location = -t * orientation;    
else
    % Sort t to make consistent order of output
    [~, idx] = sort(t(:, 3));
    for n = 1:size(t, 1)
        orientation(:,:,n) = R(:,:,idx(n))';
        location(n, :) = -t(idx(n), :) * orientation(:,:,n);
    end
end

%--------------------------------------------------------------------------
function tf = isFundamentalMatrix(M, inlierPoints1, inlierPoints2, K1, K2)
% Assume M is F
numPoints = size(inlierPoints1, 1);
pts1h = [inlierPoints1, ones(numPoints, 1, 'like', inlierPoints1)];
pts2h = [inlierPoints2, ones(numPoints, 1, 'like', inlierPoints2)];
errorF = mean(abs(diag(pts2h * M * pts1h')));

% Assume M is E
F = K2 \ M / K1';
errorE = mean(abs(diag(pts2h * F * pts1h')));

tf = errorF < errorE;

%--------------------------------------------------------------------------
function [cameraParams1, cameraParams2, inlierPoints1, inlierPoints2] = ...
    parseInputs(F, varargin)
narginchk(4, 5);
if ~isa(F, 'projective2d')
    validateattributes(F, {'single', 'double'}, ...
        {'real', 'nonsparse', 'finite', '2d', 'size', [3,3]}, mfilename, 'F');
end

cameraParams1 = varargin{1};
if isa(varargin{2}, 'cameraParameters') || isa(varargin{2}, 'cameraIntrinsics')
    cameraParams2 = varargin{2};
    paramVarName = 'cameraParams';
    idx = 2;
else
    paramVarName = 'cameraParams1';
    cameraParams2 = cameraParams1;
    idx = 1;
end
validateattributes(cameraParams1, {'cameraParameters','cameraIntrinsics'},...
    {'scalar'}, mfilename, paramVarName);

points1 = varargin{idx + 1};
points2 = varargin{idx + 2};
[inlierPoints1, inlierPoints2] = ...
    vision.internal.inputValidation.checkAndConvertMatchedPoints(...
    points1, points2, mfilename, 'inlierPoints1', 'inlierPoints2');

coder.internal.errorIf(isempty(points1), 'vision:relativeCameraPose:emptyInlierPoints');

%--------------------------------------------------------------------------
% Determine which of the 4 possible solutions is physically realizable.
% A physically realizable solution is the one which puts reconstructed 3D
% points in front of both cameras. There could be two solutions if the R
% and t are extracted from homography matrix
function [R, t, inliers] = chooseRealizableSolution(Rs, Ts, cameraParams1, ...
    cameraParams2, points1, points2)
numNegatives = zeros(1, size(Ts, 1));

camMatrix1 = cameraMatrix(cameraParams1, eye(3), [0 0 0]);
inliersMat = zeros(size(points1, 1), size(Ts, 1));
for i = 1:size(Ts, 1)
    camMatrix2 = cameraMatrix(cameraParams2, Rs(:,:,i)', Ts(i, :));
    m1 = triangulateMidPoint(points1, points2, camMatrix1, camMatrix2);
    m2 = bsxfun(@plus, m1 * Rs(:,:,i)', Ts(i, :));
    inliersMat(:, i) = (m1(:,3) < 0) | (m2(:,3) < 0);
    numNegatives(i) = sum(inliersMat(:, i));
end

val = min(numNegatives);
idx = find(numNegatives == val);

idx = idx(1);

inliers = ~inliersMat(:, idx);

% validFraction = 1 - (val / size(points1, 1));

R = zeros(3, 3, length(idx), 'like', Rs);
t = zeros(length(idx), 3, 'like', Ts);

for n = 1 : length(idx)
    R0 = Rs(:,:,idx(n))';
    t0 = Ts(idx(n), :);

    tNorm = norm(t0);
    if tNorm ~= 0
        t0 = t0 ./ tNorm;
    end
    R(:,:,n) = R0;
    t(n, :) = t0;
end
%--------------------------------------------------------------------------
% Simple triangulation algorithm from
% Hartley, Richard and Peter Sturm. "Triangulation." Computer Vision and
% Image Understanding. Vol 68, No. 2, November 1997, pp. 146-157
function points3D = triangulateMidPoint(points1, points2, P1, P2)

numPoints = size(points1, 1);
points3D = zeros(numPoints, 3, 'like', points1);
P1 = P1';
P2 = P2';

M1 = P1(1:3, 1:3);
M2 = P2(1:3, 1:3);

c1 = -M1 \ P1(:,4);
c2 = -M2 \ P2(:,4);
y = c2 - c1;

u1 = [points1, ones(numPoints, 1, 'like', points1)]';
u2 = [points2, ones(numPoints, 1, 'like', points1)]';

a1 = M1 \ u1;
a2 = M2 \ u2;

isCodegen  = ~isempty(coder.target);
condThresh = eps(class(points1));

for i = 1:numPoints
    A   = [a1(:,i), -a2(:,i)];  
    AtA = A'*A;
    
    if rcond(AtA) < condThresh
        % Guard against matrix being singular or ill-conditioned
        p    = inf(3, 1, class(points1));
        p(3) = -p(3);
    else
        if isCodegen
            % mldivide on square matrix is faster in codegen mode.
            alpha = AtA \ A' * y;
        else
            alpha = A \ y;        
        end
        p = (c1 + alpha(1) * a1(:,i) + c2 + alpha(2) * a2(:,i)) / 2;
    end
    points3D(i, :) = p';

end

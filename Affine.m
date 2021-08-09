function [final_img] = Affine(img_left,img_right)

img_left_gray = rgb2gray(img_left);
img_right_gray = rgb2gray(img_right);

% Keypoints and descriptors
[f_left, d_left] = vl_sift(single(img_left_gray));
[f_right, d_right] = vl_sift(single(img_right_gray));
f_left = f_left(1:2,:);
f_right = f_right(1:2,:);

% Euclidian distance
[matches, scores] = vl_ubcmatch(d_left, d_right);

% Prune features
threshold = 100;
i = scores < threshold;
p_matches = matches(:,i);

N = 200;
threshold = 0.03;
most_inliers = 0;
inliers_matches = [];
for i = 1:N
    match_cnt = size(p_matches,2);
    %sample 3 random points
    r = randperm(match_cnt,3);
    r1 = r(1);
    r2 = r(2);
    r3 = r(3);
    
    %matches column at r
    m1 = p_matches(:,r1);
    m2 = p_matches(:,r2);
    m3 = p_matches(:,r3);
    
    % (x,y) coordinates at match index (column m(x))
    p1_left = f_left(:,m1(1));
    p1_right = f_right(:,m1(2));
    
    p2_left = f_left(:,m2(1));
    p2_right = f_right(:,m2(2));
    
    p3_left = f_left(:,m3(1));
    p3_right = f_right(:,m3(2));
    
    % create A and B matricies
    A = [p1_left(2),p1_left(1),0,0,1,0;
        0,0,p1_left(2),p1_left(1),0,1;
        p2_left(2),p2_left(1),0,0,1,0;
        0,0,p2_left(2),p2_left(1),0,1;
        p3_left(2),p3_left(1),0,0,1,0;
        0,0,p3_left(2),p3_left(1),0,1;];
    
    b = [p1_right(2); p1_right(1); p2_right(2); p2_right(1);p3_right(2); p3_right(1);]; 
    
    % least squares
    x = A\b;

    T = [x(1), x(2);
        x(3), x(4)];
    c = [x(5); x(6)];
    
    % filter features to match pruned matches
    f_match_left = f_left(:, p_matches(1,:));
    f_match_right = f_right(:, p_matches(2,:));
    temp = [f_match_left(2, :); f_match_left(1, :)];
    f_match_left = temp;

    % apply the transform
    estimated_points = T*f_match_left + c;
    temp = [estimated_points(2,:); estimated_points(1,:)];
    estimated_points = temp;
    
    % find error using euclidian distance between points
    error = sqrt(sum((estimated_points-f_match_right).^2));
    
    % compute inliers
    inliers_index = find(error < threshold);
    
    if size(inliers_index,2) > most_inliers
        most_inliers = size(inliers_index,2);
        inliers_matches = p_matches(:,inliers_index);
    end     
end

% Generate A matrix with all inliers 
A = [];
b = [];
for i = 1: most_inliers
    left_match_index = inliers_matches(1, i);
    right_match_index = inliers_matches(2, i);
    
    p1_left = f_left(:, left_match_index);
    p1_right = f_right(:, right_match_index);
    
    A = [A; 
        p1_left(2), p1_left(1), 0, 0, 1, 0;
        0, 0, p1_left(2), p1_left(1), 0, 1];
    
    b = [b; p1_right(2); p1_right(1)];
end

% compute and apply the transformation
x = A\b;

z = [x(1), x(2), 0;
    x(3), x(4), 0;
    x(5), x(6), 1];

tform = affine2d(z);
[img_transformed,~] = imwarp(img_left, tform);

%STITCHING
left_img_pad = img_transformed;
right_img_pad = img_right;

%Run SIFT again to get coordinates of matching features with the
%transformed image
[f_left_transformed, d_left_transformed] = vl_sift(single(rgb2gray(img_transformed)));
f_left_transformed = f_left_transformed(1:2,:);

% Euclidian distance
[matches, scores] = vl_ubcmatch(d_left_transformed, d_right);

% Prune features
threshold = 100;
i = find(scores < threshold);
p_matches_transformed = matches(:,i);

p_left = f_left_transformed(:,p_matches_transformed(1,:));
p_right = f_right(:,p_matches_transformed(2,:));

% median x and y feature shift
x_shift = round(median(p_right(1)-p_left(1)))
y_shift = round(median(p_right(2)-p_left(2)))

% pad left image
if x_shift > 0
    pad = zeros(size(left_img_pad,2), abs(x_shift), 3);
    left_img_pad = [pad, left_img_pad];
end

if y_shift > 0
    left_img_pad = padarray(left_img_pad,abs(y_shift),0,'pre');
end

% pad right image
if x_shift < 0
    pad = zeros(size(img_right,1), abs(x_shift), 3);
    right_img_pad = [pad, img_right];
end

if y_shift < 0
    right_img_pad = padarray(right_img_pad,abs(y_shift),0,'pre');
end

% feature should now be alligned, 
% find difference in row and column size to pad and create equal size imgs
row_diff = size(left_img_pad, 1) - size(right_img_pad, 1);
col_diff = size(left_img_pad, 2) - size(right_img_pad, 2);

% pad right image
if(row_diff >= 0)
    right_img_pad = padarray(right_img_pad,abs(row_diff),0,'post');
end

if col_diff >= 0
    pad = zeros(size(right_img_pad,1), abs(col_diff), 3);
    right_img_pad = [right_img_pad, pad];
end

% pad left image
if row_diff < 0
    left_img_pad = padarray(left_img_pad,abs(row_diff),0,'post');
end

if col_diff < 0
    pad = zeros(size(left_img_pad,1), abs(col_diff), 3);
    left_img_pad = [left_img_pad, pad];
end


left_img_pad_grey = rgb2gray(left_img_pad);
right_img_pad_grey = rgb2gray(right_img_pad);
final_img = zeros(size(left_img_pad, 1), size(left_img_pad, 2), 3);

% stitch image by laying images on top of each other and taking the highest
% value pixel for the final image
for i = 1:size(final_img,1)
    z = left_img_pad_grey(i,:)<=right_img_pad_grey(i,:);
    final_img(i,z,:) = right_img_pad(i,z,:);
    
    z = left_img_pad_grey(i,:)>right_img_pad_grey(i,:);
    final_img(i,z,:) = left_img_pad(i,z,:);
end

end


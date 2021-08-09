function [final_img] = Homo(img_left,img_right)

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
threshold = 1900;
i = scores < threshold;
p_matches = matches(:,i);

N = 500;
threshold = 0.2;
most_inliers = 0;
inliers_matches = [];
for i = 1:N
    match_cnt = size(p_matches,2);
    
    % sample 4 random points
    rn = randperm(match_cnt,4);
    
    % Generate A matrix with sampled points
    A = [];
    for j = 1:4
        r = rn(j);

        %matches column at r (x,y)
        m = p_matches(:,r);
    
        % (x,y) coordinates at match index (column m(x))
        p_left = f_left(:,m(1));
        p_right = f_right(:,m(2));

        A = [A;
            p_left(1) p_left(2) 1 0 0 0 -p_right(1)*p_left(1) -p_right(1)*p_left(2) -p_right(1);
            0 0 0 p_left(1) p_left(2) 1 -p_right(2)*p_left(1) -p_right(2)*p_left(2) -p_right(2);];
    end
    
    % compute homography
    [~, ~, V] = svd(A);
    HT = V(:,end);
    H = reshape(HT,[3,3]);
    
    % filter features to match pruned matches
    f_match_left = f_left(:, p_matches(1,:));
    f_match_right = f_right(:, p_matches(2,:));

    % get estimate
    z = [f_match_left(1,:); f_match_left(2,:); ones(1,size(f_match_left,2))];
    estimated_points = H'*z;
    
    % compute x and y error matricies 
    error_x = abs((estimated_points(1,:)./estimated_points(3,:))-f_match_right(1,:));
    error_y = abs((estimated_points(2,:)./estimated_points(3,:))-f_match_right(2,:));
    
    % compute inliers
    inliers_index = find((error_x < threshold) & (error_y < threshold));
    
    if size(inliers_index,2) > most_inliers
        most_inliers = size(inliers_index,2);
        inliers_matches = p_matches(:,inliers_index);
    end     
end

% generate A matrix using all inliers
A = [];
for i = 1: most_inliers
    left_match_index = inliers_matches(1, i);
    right_match_index = inliers_matches(2, i);
    
    p_left = f_left(:, left_match_index);
    p_right = f_right(:, right_match_index);
    
    A = [A;
        p_left(1) p_left(2) 1 0 0 0 -p_right(1)*p_left(1) -p_right(1)*p_left(2) -p_right(1);
        0 0 0 p_left(1) p_left(2) 1 -p_right(2)*p_left(1) -p_right(2)*p_left(2) -p_right(2);];
end
% compute and apply transfomation
[~, ~, V] = svd(A);
x = V(:,end);

H = reshape(x,3,3);
tform = projective2d(H);

[img_transformed,~] = imwarp(img_left, tform);

% STITCHING
left_img_pad = img_transformed;
right_img_pad = img_right;

[f_left_transformed, d_left_transformed] = vl_sift(single(rgb2gray(img_transformed)));
f_left_transformed = f_left_transformed(1:2,:);

% Euclidian distance
[matches, scores] = vl_ubcmatch(d_left_transformed, d_right);

% Prune features
threshold = 1900;
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

% figure();
% imshow(uint8(final_img));
end


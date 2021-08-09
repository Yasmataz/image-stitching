run('vlfeat-0.9.21/toolbox/vl_setup')


%% Part 1
fprintf("Running part1....\n");

leastSquares;
clear;

fprintf("Part1 complete, press enter to see final results for part 2\n");
pause();

%% Final Results
fprintf("\n\nDisplaying final results\n");
openfig("2A.fig");
openfig("2B1.fig");
openfig("2B2.fig");

fprintf("\n\nPress enter to run part 2 \n");
pause();

%% Part 2
fprintf("\n\nRunning part 2 .....\n");
tic;
img_left = imread("parliament-left.jpg");
img_right = imread("parliament-right.jpg");

final_img = Affine(img_left, img_right);
figure();
imshow(uint8(final_img));

img_left = imread("Ryerson-right.jpg");
img_right = imread("Ryerson-left.jpg");

final_img = Homo(img_left, img_right);
figure();
imshow(uint8(final_img));

img_left = imread("hill1.jpg");
img_right = imread("hill2.jpg");

final_img = Homo(img_left, img_right);
figure();
imshow(uint8(final_img));
toc;



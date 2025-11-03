import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import os
import time

class CameraCalibrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision and Deep Learning - Homework 1")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.images_folder = None
        self.images = []
        self.gray_images = []
        self.corners_list = []
        self.objpoints = []
        self.imgpoints = []
        
        # Calibration results
        self.mtx = None  # Intrinsic matrix
        self.dist = None  # Distortion coefficients
        self.rvecs = None  # Rotation vectors
        self.tvecs = None  # Translation vectors
        
        # Stereo images
        self.imgL = None
        self.imgR = None
        
        # SIFT images
        self.sift_img1 = None
        self.sift_img2 = None
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Computer Vision Homework 1")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Create sections
        self.create_section1(main_layout)
        self.create_section2(main_layout)
        self.create_section3(main_layout)
        self.create_section4(main_layout)
        
        # Console output
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        console_layout.addWidget(self.console)
        console_group.setLayout(console_layout)
        main_layout.addWidget(console_group)
        
    def create_section1(self, main_layout):
        """Section 1: Camera Calibration"""
        group = QGroupBox("1. Camera Calibration")
        layout = QVBoxLayout()
        
        # Load folder button
        btn_load = QPushButton("Load Folder")
        btn_load.clicked.connect(self.load_calibration_folder)
        layout.addWidget(btn_load)
        
        # Buttons row 1
        row1 = QHBoxLayout()
        btn_11 = QPushButton("1.1 Find Corners")
        btn_11.clicked.connect(self.find_corners)
        btn_12 = QPushButton("1.2 Find Intrinsic")
        btn_12.clicked.connect(self.find_intrinsic)
        row1.addWidget(btn_11)
        row1.addWidget(btn_12)
        layout.addLayout(row1)
        
        # Buttons row 2
        row2 = QHBoxLayout()
        btn_13 = QPushButton("1.3 Find Extrinsic")
        btn_13.clicked.connect(self.find_extrinsic)
        
        self.combo_image = QComboBox()
        for i in range(1, 16):
            self.combo_image.addItem(str(i))
        
        btn_14 = QPushButton("1.4 Find Distortion")
        btn_14.clicked.connect(self.find_distortion)
        btn_15 = QPushButton("1.5 Show Result")
        btn_15.clicked.connect(self.show_undistorted)
        
        row2.addWidget(btn_13)
        row2.addWidget(self.combo_image)
        row2.addWidget(btn_14)
        row2.addWidget(btn_15)
        layout.addLayout(row2)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
        
    def create_section2(self, main_layout):
        """Section 2: Augmented Reality"""
        group = QGroupBox("2. Augmented Reality")
        layout = QVBoxLayout()
        
        # Text input
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Input Word (max 6 chars):"))
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(30)
        row1.addWidget(self.text_input)
        layout.addLayout(row1)
        
        # Buttons
        row2 = QHBoxLayout()
        btn_21 = QPushButton("2.1 Show Words on Board")
        btn_21.clicked.connect(self.show_words_on_board)
        btn_22 = QPushButton("2.2 Show Words Vertically")
        btn_22.clicked.connect(self.show_words_vertically)
        row2.addWidget(btn_21)
        row2.addWidget(btn_22)
        layout.addLayout(row2)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
        
    def create_section3(self, main_layout):
        """Section 3: Stereo Disparity Map"""
        group = QGroupBox("3. Stereo Disparity Map")
        layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        btn_load_l = QPushButton("Load Image_L")
        btn_load_l.clicked.connect(self.load_image_l)
        btn_load_r = QPushButton("Load Image_R")
        btn_load_r.clicked.connect(self.load_image_r)
        btn_31 = QPushButton("3.1 Stereo Disparity Map")
        btn_31.clicked.connect(self.compute_disparity)
        
        row1.addWidget(btn_load_l)
        row1.addWidget(btn_load_r)
        row1.addWidget(btn_31)
        layout.addLayout(row1)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
        
    def create_section4(self, main_layout):
        """Section 4: SIFT"""
        group = QGroupBox("4. SIFT")
        layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        btn_load1 = QPushButton("Load Image 1")
        btn_load1.clicked.connect(self.load_sift_image1)
        btn_load2 = QPushButton("Load Image 2")
        btn_load2.clicked.connect(self.load_sift_image2)
        row1.addWidget(btn_load1)
        row1.addWidget(btn_load2)
        layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        btn_41 = QPushButton("4.1 Keypoints")
        btn_41.clicked.connect(self.show_keypoints)
        btn_42 = QPushButton("4.2 Matched Keypoints")
        btn_42.clicked.connect(self.show_matched_keypoints)
        row2.addWidget(btn_41)
        row2.addWidget(btn_42)
        layout.addLayout(row2)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
        
    def log(self, message):
        """Log message to console"""
        self.console.append(message)
        QApplication.processEvents()
        
    def load_calibration_folder(self):
        """Load calibration images folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Images Folder")
        if folder:
            self.images_folder = folder
            self.images = []
            self.gray_images = []
            
            # Load images 1.bmp to 15.bmp
            for i in range(1, 16):
                img_path = os.path.join(folder, f"{i}.bmp")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    self.images.append(img)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.gray_images.append(gray)
            
            self.log(f"Loaded {len(self.images)} images from {folder}")
            
    def find_corners(self):
        if not self.gray_images:
            self.log("Please load calibration folder first!")
            return

        self.corners_list, self.objpoints, self.imgpoints = [], [], []

        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        screen = QApplication.primaryScreen()
        sw, sh = screen.size().width(), screen.size().height()

        for idx, gray in enumerate(self.gray_images):
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                self.corners_list.append(corners2)
                self.objpoints.append(objp)
                self.imgpoints.append(corners2)

                img_show = self.images[idx].copy()
                cv2.drawChessboardCorners(img_show, (11, 8), corners2, ret)

                h, w = img_show.shape[:2]
                scale = min(sw / (w + 0.0), sh / (h + 0.0), 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                if scale < 1.0:
                    img_show = cv2.resize(img_show, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    new_w, new_h = w, h

                canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
                x0 = (sw - new_w) // 2
                y0 = (sh - new_h) // 2
                canvas[y0:y0+new_h, x0:x0+new_w] = img_show

                win = f"Corners - Image {idx+1}"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(win, canvas)

                cv2.waitKey(500)


                try:
                    cv2.destroyWindow(win)
                except cv2.error:
                    pass
            else:
                self.log(f"[Q1] Corner not found in image {idx+1}")

        cv2.destroyAllWindows()
        self.log(f"Found corners in {len(self.corners_list)} images")

        
    def find_intrinsic(self):
        """1.2 Find intrinsic matrix"""
        if not self.objpoints or not self.imgpoints:
            self.log("Please find corners first!")
            return
            
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, 
            self.gray_images[0].shape[::-1], None, None)
        
        self.log("Intrinsic Matrix:")
        self.log(str(self.mtx))
        
    def find_extrinsic(self):
        """1.3 Find extrinsic matrix"""
        if self.rvecs is None or self.tvecs is None:
            self.log("Please find intrinsic matrix first!")
            return
            
        idx = self.combo_image.currentIndex()
        
        if idx < len(self.rvecs):
            rvec = self.rvecs[idx]
            tvec = self.tvecs[idx]
            
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Combine R and T to form extrinsic matrix [R|T]
            extrinsic = np.hstack((R, tvec))
            
            self.log(f"Extrinsic Matrix for Image {idx+1}:")
            self.log(str(extrinsic))
            
    def find_distortion(self):
        """1.4 Find distortion matrix"""
        if self.dist is None:
            self.log("Please find intrinsic matrix first!")
            return
            
        self.log("Distortion Matrix:")
        self.log(str(self.dist))
        
    def show_undistorted(self):
        if self.mtx is None or self.dist is None:
            self.log("Please calibrate camera first!")
            return

        if not self.images:
            self.log("Please load calibration folder first!")
            return

        for idx, img in enumerate(self.images):
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h))
            
            # Undistort
            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            
            # Add text labels
            distorted_labeled = img.copy()
            undistorted_labeled = dst.copy()
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 3
            text_color = (0, 0, 255)  # Red
            
            cv2.putText(distorted_labeled, 'Distorted', (50, 100),
                    font, font_scale, text_color, font_thickness)
            cv2.putText(undistorted_labeled, 'Undistorted', (50, 100),
                    font, font_scale, text_color, font_thickness)
            
            # Combine images side by side
            combined = np.hstack((distorted_labeled, undistorted_labeled))
            
            # Display fullscreen
            window_name = f'Distorted vs Undistorted - Image {idx+1}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, combined)
            
            self.log(f"Displayed undistorted image {idx+1}")
            
            key = cv2.waitKey(500)
            if key == 27:  # 按 ESC 可提早結束
                break

            cv2.destroyWindow(window_name)

        cv2.destroyAllWindows()
        self.log("Finished showing all undistorted images.")

            
    def show_words_on_board(self):
        if self.mtx is None:
            self.log("Please calibrate camera first!")
            return

        word = self.text_input.toPlainText().strip().upper()[:6]
        reversed(word)
        print(word)
        if not word:
            self.log("Please input a word!")
            return

        # Load alphabet database
        db_path = QFileDialog.getOpenFileName(
            self, "Select alphabet_db_onboard.txt", "", "Text Files (*.txt)")[0]
        if not db_path:
            return

        fs = cv2.FileStorage(db_path, cv2.FILE_STORAGE_READ)

        positions = [(1, 2), (4, 2), (7, 2), (1, 5), (4, 5), (7, 5)]

        screen = QApplication.primaryScreen()
        sw, sh = screen.size().width(), screen.size().height()

        for img_idx in range(min(5, len(self.images))):
            img = self.images[img_idx].copy()

            for char_idx, char in enumerate(reversed(word)):
                if char_idx >= 6:
                    break

                node = fs.getNode(char)
                if not node.isNone():
                    char_points = node.mat()
                else:
                    continue

                # Translate to position
                offset_x, offset_y = positions[char_idx]
                translated_points = char_points.copy()
                translated_points[:, :, 0] += offset_x
                translated_points[:, :, 1] += offset_y

                # Project to 2D and draw lines
                for line in translated_points:
                    points_3d = line.reshape(-1, 1, 3).astype(np.float32)
                    points_2d, _ = cv2.projectPoints(
                        points_3d, self.rvecs[img_idx], self.tvecs[img_idx],
                        self.mtx, self.dist)

                    pt1 = tuple(points_2d[0][0].astype(int))
                    pt2 = tuple(points_2d[1][0].astype(int))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 5)

            h, w = img.shape[:2]
            scale = min(sw / w, sh / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else img

            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            x0 = (sw - new_w) // 2
            y0 = (sh - new_h) // 2
            canvas[y0:y0+new_h, x0:x0+new_w] = resized

            win = f"AR Result {img_idx+1} - '{word}'"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(win, canvas)

            cv2.waitKey(1000)
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass

        fs.release()
        cv2.destroyAllWindows()
        self.log(f"Displayed word '{word}' on board.")

        
    def show_words_vertically(self):
        if self.mtx is None:
            self.log("Please calibrate camera first!")
            return

        word = self.text_input.toPlainText().strip().upper()[:6]
        if not word:
            self.log("Please input a word!")
            return

        db_path = QFileDialog.getOpenFileName(
            self, "Select alphabet_db_vertical.txt", "", "Text Files (*.txt)")[0]
        if not db_path:
            return

        fs = cv2.FileStorage(db_path, cv2.FILE_STORAGE_READ)

        positions = [(1, 2), (4, 2), (7, 2), (1, 5), (4, 5), (7, 5)]

        screen = QApplication.primaryScreen()
        sw, sh = screen.size().width(), screen.size().height()

        for img_idx in range(min(5, len(self.images))):
            img = self.images[img_idx].copy()

            for char_idx, ch in enumerate(reversed(word)):
                if char_idx >= 6:
                    break

                node = fs.getNode(ch)
                if node.isNone():
                    continue

                char_pts = np.array(node.mat(), dtype=np.float32)  # (N,2,3)
                px, py = positions[char_idx]

                char_pts[:, :, 0] += float(px)  # X shift
                char_pts[:, :, 1] += float(py)  # Y shift
                # char_pts[:, :, 2] 保持原來的 Z（字體的高度由資料庫決定）

                for line3d in char_pts:
                    pts2d, _ = cv2.projectPoints(
                        line3d.reshape(-1, 3), self.rvecs[img_idx], self.tvecs[img_idx],
                        self.mtx, self.dist)
                    pts2d = np.asarray(pts2d, dtype=np.float32).reshape(-1, 2)
                    if pts2d.shape[0] < 2:
                        continue
                    p1 = (int(round(pts2d[0, 0])), int(round(pts2d[0, 1])))
                    p2 = (int(round(pts2d[1, 0])), int(round(pts2d[1, 1])))
                    cv2.line(img, p1, p2, (0, 0, 255), 5)

            h, w = img.shape[:2]
            scale = min(sw / w, sh / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            disp = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            x0, y0 = (sw - nw) // 2, (sh - nh) // 2
            canvas[y0:y0+nh, x0:x0+nw] = disp

            win = f"AR Vertical Result {img_idx+1} - '{word}'"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(win, canvas)
            cv2.waitKey(1000)
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass

        fs.release()
        cv2.destroyAllWindows()
        self.log(f"Displayed word '{word}' vertically (fixed XY translation).")


        
    def load_image_l(self):
        """Load left stereo image"""
        filename = QFileDialog.getOpenFileName(self, "Select Left Image", 
                                               "", "Images (*.png *.jpg *.bmp)")[0]
        if filename:
            self.imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.log(f"Loaded left image: {filename}")
            
    def load_image_r(self):
        """Load right stereo image"""
        filename = QFileDialog.getOpenFileName(self, "Select Right Image",
                                               "", "Images (*.png *.jpg *.bmp)")[0]
        if filename:
            self.imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.log(f"Loaded right image: {filename}")
            
    def compute_disparity(self):
        """3.1 Compute stereo disparity map (全螢幕黑底、只縮小不放大)"""
        if self.imgL is None or self.imgR is None:
            self.log("Please load both left and right images!")
            return

        stereo = cv2.StereoBM_create(numDisparities=432, blockSize=25)

        # Compute disparity
        disp = stereo.compute(self.imgL, self.imgR)

        # Normalize for display (8-bit)
        disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        screen = QApplication.primaryScreen()
        sw, sh = screen.size().width(), screen.size().height()

        h, w = disp_norm.shape[:2]
        scale = min(sw / w, sh / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if scale < 1.0:
            disp_show = cv2.resize(disp_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp_show = disp_norm
            new_w, new_h = w, h

        canvas = np.zeros((sh, sw), dtype=np.uint8)  # 灰階即可
        x0 = (sw - new_w) // 2
        y0 = (sh - new_h) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = disp_show

        win = "Disparity Map"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(win, canvas)

        cv2.waitKey(0)
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
        self.log("Computed disparity map")

        
    def load_sift_image1(self):
        """Load first SIFT image"""
        filename = QFileDialog.getOpenFileName(self, "Select Image 1",
                                               "", "Images (*.png *.jpg *.bmp)")[0]
        if filename:
            self.sift_img1 = cv2.imread(filename)
            self.log(f"Loaded SIFT image 1: {filename}")
            
    def load_sift_image2(self):
        """Load second SIFT image"""
        filename = QFileDialog.getOpenFileName(self, "Select Image 2",
                                               "", "Images (*.png *.jpg *.bmp)")[0]
        if filename:
            self.sift_img2 = cv2.imread(filename)
            self.log(f"Loaded SIFT image 2: {filename}")
            
    def show_keypoints(self):
        if self.sift_img1 is None:
            self.log("Please load image 1 first!")
            return

        gray = cv2.cvtColor(self.sift_img1, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        img_kp = cv2.drawKeypoints(
            gray, keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        screen = QApplication.primaryScreen()
        sw, sh = screen.size().width(), screen.size().height()

        h, w = img_kp.shape[:2]
        scale = min(sw / w, sh / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if scale < 1.0:
            img_disp = cv2.resize(img_kp, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_disp = img_kp
            new_w, new_h = w, h

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        x0 = (sw - new_w) // 2
        y0 = (sh - new_h) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = img_disp

        win = "SIFT Keypoints"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(win, canvas)

        cv2.waitKey(0)
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
        self.log(f"Found {len(keypoints)} keypoints")

        
    def show_matched_keypoints(self):
        if self.sift_img1 is None or self.sift_img2 is None:
            self.log("Please load both images first!")
            return

        gray1 = cv2.cvtColor(self.sift_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.sift_img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])

        img_matches = cv2.drawMatchesKnn(
            gray1, kp1, gray2, kp2, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        if len(img_matches.shape) == 2:  # 若是灰階，轉成 BGR 方便貼到彩色 canvas
            img_matches = cv2.cvtColor(img_matches, cv2.COLOR_GRAY2BGR)

        screen = QApplication.primaryScreen()
        sw, sh = screen.size().width(), screen.size().height()

        h, w = img_matches.shape[:2]
        scale = min(sw / w, sh / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if scale < 1.0:
            disp = cv2.resize(img_matches, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp = img_matches
            new_w, new_h = w, h

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        x0 = (sw - new_w) // 2
        y0 = (sh - new_h) // 2

        canvas[y0:y0+new_h, x0:x0+new_w] = disp

        win = "Matched Keypoints"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(win, canvas)

        cv2.waitKey(0)
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
        self.log(f"Found {len(good_matches)} good matches")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraCalibrationApp()
    window.show()
    sys.exit(app.exec_())
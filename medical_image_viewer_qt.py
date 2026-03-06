import os
import sys
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPolygon, QColor
from PyQt5.QtCore import QPoint
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, skeletonize_3d


class MedicalImageViewer(QMainWindow):
    def __init__(self, folder_paths, folder_labels):
        super().__init__()
        self.folder_paths = folder_paths
        self.folder_labels = folder_labels
        self.num_folders = len(folder_paths)
        
        # 加载第一个文件
        self.load_first_file()
        
        # 创建UI
        self.init_ui()
        
    def load_first_file(self):
        """加载第一个文件"""
        image_folder = self.folder_paths[0]
        nii_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.nii.gz')])
        
        if not nii_files:
            raise ValueError(f"在 {image_folder} 中没有找到.nii.gz文件")
        
        self.current_file = nii_files[0]
        self.all_files = nii_files
        self.current_file_index = 0
        
        self.load_data()
        
    def load_data(self):
        """加载当前文件的所有数据"""
        self.data_list = []
        
        for folder_path in self.folder_paths:
            file_path = os.path.join(folder_path, self.current_file)
            if os.path.exists(file_path):
                nii_img = nib.load(file_path)
                data = nii_img.get_fdata()
                self.data_list.append(data)
            else:
                print(f"警告: 文件不存在 {file_path}")
                self.data_list.append(None)
        
        if self.data_list[0] is not None:
            self.num_slices = self.data_list[0].shape[2]
            
            # 找出GT中有掩膜的切片索引
            self.valid_slices = []
            gt_data = self.data_list[1]  # GT数据
            for i in range(self.num_slices):
                gt_slice = gt_data[:, :, i]
                if np.sum(gt_slice > 0) > 0:  # 如果GT切片有掩膜
                    self.valid_slices.append(i)
            
            if len(self.valid_slices) == 0:
                raise ValueError("GT中没有找到任何掩膜")
            
            # 设置当前切片为第一个有效切片
            self.current_valid_index = len(self.valid_slices) // 2
            self.current_slice = self.valid_slices[self.current_valid_index]
            
            # 计算3D指标（只计算一次）
            self.calculate_3d_metrics()
        else:
            raise ValueError("无法加载图像数据")
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('医学图像分割结果对比查看器')
        
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 文件信息标签
        self.file_label = QLabel()
        self.file_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.file_label)
        
        # 图像显示区域 - 使用网格布局
        image_layout = QHBoxLayout()
        self.image_2d_labels = []  # 2D切片标签
        self.image_3d_labels = []  # 3D可视化标签
        self.title_labels = []
        self.metrics_labels = []  # 指标显示标签
        
        # 显示所有文件夹（包括Image）
        for i in range(self.num_folders):
            container = QVBoxLayout()
            
            # 标题标签
            title_label = QLabel(self.folder_labels[i])
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 5px;")
            container.addWidget(title_label)
            self.title_labels.append(title_label)
            
            # 2D切片图像标签
            img_2d_label = QLabel()
            img_2d_label.setAlignment(Qt.AlignCenter)
            img_2d_label.setMinimumSize(300, 300)
            img_2d_label.setStyleSheet("border: 2px solid gray; background-color: black;")
            container.addWidget(img_2d_label)
            self.image_2d_labels.append(img_2d_label)
            
            # 3D可视化标签或指标显示
            if i == 0:
                # Image列：显示所有模型的指标汇总
                metrics_summary_label = QLabel()
                metrics_summary_label.setAlignment(Qt.AlignCenter)
                metrics_summary_label.setMinimumSize(300, 300)
                metrics_summary_label.setStyleSheet(
                    "border: 2px solid gray; background-color: #f5f5f5; "
                    "font-size: 11pt; color: #333333; padding: 20px;"
                )
                metrics_summary_label.setWordWrap(True)
                container.addWidget(metrics_summary_label)
                self.image_3d_labels.append(metrics_summary_label)
                self.metrics_summary_label = metrics_summary_label
            else:
                # GT和模型列：显示3D可视化
                img_3d_label = QLabel()
                img_3d_label.setAlignment(Qt.AlignCenter)
                img_3d_label.setMinimumSize(300, 300)
                img_3d_label.setStyleSheet("border: 2px solid gray; background-color: white;")
                container.addWidget(img_3d_label)
                self.image_3d_labels.append(img_3d_label)
            
            image_layout.addLayout(container)
        
        main_layout.addLayout(image_layout)
        
        # 控制区域
        control_layout = QHBoxLayout()
        
        # 上一个文件按钮
        self.prev_file_btn = QPushButton('上一个文件 (←)')
        self.prev_file_btn.clicked.connect(self.prev_file)
        control_layout.addWidget(self.prev_file_btn)
        
        # 切片滑动条
        self.slice_label = QLabel(f'切片: {self.current_valid_index + 1}/{len(self.valid_slices)}')
        control_layout.addWidget(self.slice_label)
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(len(self.valid_slices) - 1)
        self.slice_slider.setValue(self.current_valid_index)
        self.slice_slider.valueChanged.connect(self.update_slice)
        control_layout.addWidget(self.slice_slider)
        
        # 下一个文件按钮
        self.next_file_btn = QPushButton('下一个文件 (→)')
        self.next_file_btn.clicked.connect(self.next_file)
        control_layout.addWidget(self.next_file_btn)
        
        main_layout.addLayout(control_layout)
        
        # 初始显示
        self.update_display()
        
        # 设置窗口大小
        self.resize(300 * self.num_folders, 650)
    
    def normalize_image(self, img):
        """归一化图像到0-255"""
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        return img.astype(np.uint8)
    
    def calculate_dice(self, pred, gt):
        """计算Dice系数"""
        pred_binary = (pred > 0).astype(np.float32)
        gt_binary = (gt > 0).astype(np.float32)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = 2.0 * intersection / union
        return dice
    
    def calculate_cldice(self, pred, gt, is_3d=False):
        """计算clDice系数（centerline Dice）"""
        # 确保数组是C连续的
        pred = np.ascontiguousarray(pred)
        gt = np.ascontiguousarray(gt)
        
        pred_binary = (pred > 0).astype(bool)
        gt_binary = (gt > 0).astype(bool)
        
        if not pred_binary.any() or not gt_binary.any():
            return 0.0
        
        # 骨架化
        try:
            if is_3d:
                pred_skel = skeletonize_3d(pred_binary)
                gt_skel = skeletonize_3d(gt_binary)
            else:
                pred_skel = skeletonize(pred_binary)
                gt_skel = skeletonize(gt_binary)
        except:
            # 如果骨架化失败，返回0
            return 0.0
        
        if not pred_skel.any() or not gt_skel.any():
            return 0.0
        
        # 计算距离变换
        pred_dt = distance_transform_edt(~pred_binary)
        gt_dt = distance_transform_edt(~gt_binary)
        
        # 计算tprec和tsens
        tprec = np.sum(pred_skel * (gt_dt == 0)) / np.sum(pred_skel)
        tsens = np.sum(gt_skel * (pred_dt == 0)) / np.sum(gt_skel)
        
        # clDice
        if tprec + tsens == 0:
            return 0.0
        
        cldice = 2.0 * tprec * tsens / (tprec + tsens)
        return cldice
    
    def calculate_3d_metrics(self):
        """计算3D指标（只计算一次）"""
        self.metrics_3d = {}
        
        # GT数据
        gt_data = self.data_list[1]  # 第二个文件夹是GT
        
        # 对每个模型计算3D指标
        for idx in range(2, self.num_folders):  # 从第三个文件夹开始是模型
            pred_data = self.data_list[idx]
            
            if pred_data is not None and gt_data is not None:
                dice_3d = self.calculate_dice(pred_data, gt_data)
                cldice_3d = self.calculate_cldice(pred_data, gt_data, is_3d=True)
                
                self.metrics_3d[idx] = {
                    'dice_3d': dice_3d,
                    'cldice_3d': cldice_3d
                }
            else:
                self.metrics_3d[idx] = {
                    'dice_3d': 0.0,
                    'cldice_3d': 0.0
                }
    
    def calculate_2d_metrics(self, slice_idx):
        """计算当前2D切片的指标"""
        metrics_2d = {}
        
        # GT切片
        gt_slice = self.data_list[1][:, :, slice_idx]
        
        # 对每个模型计算2D指标
        for idx in range(2, self.num_folders):
            pred_slice = self.data_list[idx][:, :, slice_idx]
            
            if pred_slice is not None and gt_slice is not None:
                dice_2d = self.calculate_dice(pred_slice, gt_slice)
                cldice_2d = self.calculate_cldice(pred_slice, gt_slice, is_3d=False)
                
                metrics_2d[idx] = {
                    'dice_2d': dice_2d,
                    'cldice_2d': cldice_2d
                }
            else:
                metrics_2d[idx] = {
                    'dice_2d': 0.0,
                    'cldice_2d': 0.0
                }
        
        return metrics_2d
    
    def create_overlay_image(self, image_data, mask_data):
        """创建叠加图像"""
        # 归一化图像
        image_norm = self.normalize_image(image_data)
        
        # 创建RGB图像
        h, w = image_norm.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = image_norm
        rgb_image[:, :, 1] = image_norm
        rgb_image[:, :, 2] = image_norm
        
        # 创建淡红色掩膜叠加
        mask_binary = (mask_data > 0.1).astype(np.float32)
        
        # 在有掩膜的地方叠加淡红色
        # 使用alpha混合
        alpha = 0.4  # 透明度
        red_color = np.array([255, 100, 100], dtype=np.float32)
        
        for i in range(3):
            rgb_image[:, :, i] = np.where(
                mask_binary > 0,
                rgb_image[:, :, i] * (1 - alpha) + red_color[i] * alpha,
                rgb_image[:, :, i]
            ).astype(np.uint8)
        
        return rgb_image
    
    def numpy_to_qimage(self, img):
        """将numpy数组转换为QImage"""
        if len(img.shape) == 2:
            # 灰度图
            img_norm = self.normalize_image(img)
            h, w = img_norm.shape
            qimg = QImage(img_norm.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # RGB图
            h, w, c = img.shape
            bytes_per_line = 3 * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        return qimg
    
    def create_3d_visualization(self, mask_data_3d):
        """创建3D可视化效果（最大强度投影 MIP）"""
        # 获取3D掩膜的尺寸
        depth, height, width = mask_data_3d.shape
        
        # 创建画布
        canvas_size = 300
        
        # 使用最大强度投影（MIP）从不同角度
        # 旋转角度以获得更好的3D视角
        angle = 15  # 旋转角度
        
        # 创建投影图像
        from scipy.ndimage import rotate
        
        # 对3D数据进行轻微旋转以获得透视效果
        rotated = rotate(mask_data_3d, angle, axes=(0, 1), reshape=False, order=1)
        
        # 最大强度投影（沿深度方向）
        mip = np.max(rotated, axis=0)
        
        # 归一化
        if mip.max() > 0:
            mip = mip / mip.max()
        
        # 创建彩色图像（淡红色）
        h, w = mip.shape
        colored = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 应用淡红色映射
        mask = mip > 0.1
        intensity = mip[mask]
        
        # 淡红色渐变
        colored[mask, 0] = 255  # 红色通道
        colored[mask, 1] = (255 - intensity * 100).astype(np.uint8)  # 绿色通道
        colored[mask, 2] = (255 - intensity * 100).astype(np.uint8)  # 蓝色通道
        
        # 添加深度阴影效果
        # 计算每个像素的深度（第一个非零切片的位置）
        depth_map = np.zeros_like(mip)
        for i in range(depth):
            slice_mask = rotated[i, :, :] > 0.1
            depth_map[slice_mask & (depth_map == 0)] = i / depth
        
        # 根据深度调整亮度
        depth_shading = (1 - depth_map * 0.3)
        for c in range(3):
            colored[:, :, c] = (colored[:, :, c] * depth_shading).astype(np.uint8)
        
        # 调整大小以适应画布
        from scipy.ndimage import zoom
        scale = min(canvas_size * 0.8 / h, canvas_size * 0.8 / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        if new_h > 0 and new_w > 0:
            zoom_factors = (new_h / h, new_w / w, 1)
            resized = zoom(colored, zoom_factors, order=1)
        else:
            resized = colored
        
        # 创建最终画布并居中放置
        final_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        start_y = (canvas_size - resized.shape[0]) // 2
        start_x = (canvas_size - resized.shape[1]) // 2
        end_y = start_y + resized.shape[0]
        end_x = start_x + resized.shape[1]
        
        final_canvas[start_y:end_y, start_x:end_x] = resized
        
        # 转换为QPixmap
        qimg = QImage(final_canvas.data, canvas_size, canvas_size, 
                     canvas_size * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        return pixmap
    
    def update_display(self):
        """更新显示"""
        # 计算当前切片的2D指标
        metrics_2d = self.calculate_2d_metrics(self.current_slice)
        
        # 准备指标汇总文本
        metrics_summary_lines = []
        
        # 遍历所有文件夹（包括Image）
        for idx in range(self.num_folders):
            
            if self.data_list[idx] is not None:
                # 收集模型指标（跳过Image和GT）
                if idx > 1:  # 跳过Image和GT
                    dice_3d = self.metrics_3d[idx]['dice_3d']
                    cldice_3d = self.metrics_3d[idx]['cldice_3d']
                    dice_2d = metrics_2d[idx]['dice_2d']
                    cldice_2d = metrics_2d[idx]['cldice_2d']
                    
                    # 添加到汇总
                    model_name = self.folder_labels[idx]
                    metrics_summary_lines.append(f"{model_name}:")
                    metrics_summary_lines.append(f"  3D: Dice={dice_3d:.3f}, clDice={cldice_3d:.3f}")
                    metrics_summary_lines.append(f"  2D: Dice={dice_2d:.3f}, clDice={cldice_2d:.3f}")
                    metrics_summary_lines.append("")
                
                # 2D切片显示
                slice_data = self.data_list[idx][:, :, self.current_slice]
                image_data = self.data_list[0][:, :, self.current_slice].T
                
                if idx == 0:
                    # Image列：只显示原始图像，不叠加掩膜
                    img_array = self.create_overlay_image(image_data, np.zeros_like(image_data))
                else:
                    # GT和模型列：显示图像+掩膜
                    mask_data = slice_data.T
                    img_array = self.create_overlay_image(image_data, mask_data)
                
                qimg = self.numpy_to_qimage(img_array)
                
                pixmap_2d = QPixmap.fromImage(qimg)
                scaled_pixmap_2d = pixmap_2d.scaled(
                    self.image_2d_labels[idx].size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_2d_labels[idx].setPixmap(scaled_pixmap_2d)
                
                # 3D可视化显示（跳过Image列）
                if idx > 0:
                    mask_data_3d = self.data_list[idx].transpose(2, 0, 1)  # 转换为 (depth, height, width)
                    pixmap_3d = self.create_3d_visualization(mask_data_3d)
                    self.image_3d_labels[idx].setPixmap(pixmap_3d)
        
        # 更新Image列的指标汇总
        if hasattr(self, 'metrics_summary_label'):
            summary_text = "模型指标汇总\n" + "="*30 + "\n\n" + "\n".join(metrics_summary_lines)
            self.metrics_summary_label.setText(summary_text)
        
        # 更新文件信息
        self.file_label.setText(f'文件: {self.current_file} ({self.current_file_index + 1}/{len(self.all_files)}) | '
                               f'有效切片: {len(self.valid_slices)}/{self.num_slices}')
        self.slice_label.setText(f'切片: {self.current_valid_index + 1}/{len(self.valid_slices)} (实际: {self.current_slice + 1})')
        
        # 更新按钮状态
        self.prev_file_btn.setEnabled(self.current_file_index > 0)
        self.next_file_btn.setEnabled(self.current_file_index < len(self.all_files) - 1)
    
    def update_slice(self, value):
        """滑动条回调"""
        self.current_valid_index = int(value)
        self.current_slice = self.valid_slices[self.current_valid_index]
        self.update_display()
    
    def prev_file(self):
        """上一个文件"""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.current_file = self.all_files[self.current_file_index]
            self.load_data()
            self.slice_slider.setMaximum(len(self.valid_slices) - 1)
            self.slice_slider.setValue(len(self.valid_slices) // 2)
            self.update_display()
    
    def next_file(self):
        """下一个文件"""
        if self.current_file_index < len(self.all_files) - 1:
            self.current_file_index += 1
            self.current_file = self.all_files[self.current_file_index]
            self.load_data()
            self.slice_slider.setMaximum(len(self.valid_slices) - 1)
            self.slice_slider.setValue(len(self.valid_slices) // 2)
            self.update_display()
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Left:
            self.prev_file()
        elif event.key() == Qt.Key_Right:
            self.next_file()
    
    def wheelEvent(self, event):
        """鼠标滚轮事件"""
        # 获取滚轮滚动方向
        delta = event.angleDelta().y()
        
        if delta > 0:
            # 向上滚动 - 上一个有效切片
            if self.current_valid_index > 0:
                self.current_valid_index -= 1
                self.current_slice = self.valid_slices[self.current_valid_index]
                self.slice_slider.setValue(self.current_valid_index)
        else:
            # 向下滚动 - 下一个有效切片
            if self.current_valid_index < len(self.valid_slices) - 1:
                self.current_valid_index += 1
                self.current_slice = self.valid_slices[self.current_valid_index]
                self.slice_slider.setValue(self.current_valid_index)


def main():
    """主函数"""
    # 配置文件夹路径和标签
    folder_paths = [
        r"E:\Project\MAISI\CoronaryDataset\preprocessed\test\imagesTr",           # 原始图像文件夹
        r"E:\Project\MAISI\CoronaryDataset\preprocessed\test\labelsTr",     # 真实掩膜文件夹
        r"E:\Project\MAISI\CoronaryDataset\preprocessed\test\labelsTr",     # 真实掩膜文件夹
        # 可以继续添加更多文件夹
    ]
    
    folder_labels = [
        "Image",      # 原始图像标签
        "GT",         # 真实掩膜标签
        "Unet"
        # 对应添加更多标签
    ]
    
    # 验证输入
    if len(folder_paths) != len(folder_labels):
        raise ValueError("文件夹路径数量必须与标签数量相同")
    
    if len(folder_paths) < 2:
        raise ValueError("至少需要2个文件夹（图像和GT）")
    
    if len(folder_paths) > 10:
        raise ValueError("最多支持10个文件夹")
    
    # 检查文件夹是否存在
    for path in folder_paths:
        if not os.path.exists(path):
            raise ValueError(f"文件夹不存在: {path}")
    
    # 创建应用
    app = QApplication(sys.argv)
    viewer = MedicalImageViewer(folder_paths, folder_labels)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

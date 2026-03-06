import os
import sys
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QGridLayout,
                             QProgressDialog, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QThreadPool, QRunnable, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPolygon, QColor
from PyQt5.QtCore import QPoint
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, skeletonize_3d
from concurrent.futures import ThreadPoolExecutor
import threading


class PrecomputeThread(QThread):
    """预计算线程 - 在后台计算当前文件所有切片的2D指标"""
    progress = pyqtSignal(str)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.running = True
    
    def run(self):
        """预计算当前文件的所有2D指标"""
        try:
            if not hasattr(self.viewer, 'valid_slices') or not self.viewer.valid_slices:
                return
            
            self.progress.emit("正在预计算当前文件的2D指标...")
            
            # 使用更多线程池并发计算（8个工作线程）
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for slice_idx in self.viewer.valid_slices:
                    if not self.running:
                        break
                    future = executor.submit(self.compute_slice_metrics, slice_idx)
                    futures.append((slice_idx, future))
                
                # 等待所有任务完成
                for slice_idx, future in futures:
                    if not self.running:
                        break
                    try:
                        result = future.result()
                        self.viewer.metrics_2d_cache[self.viewer.current_file][slice_idx] = result
                    except Exception as e:
                        print(f"计算切片 {slice_idx} 指标时出错: {e}")
            
            self.progress.emit("当前文件2D指标预计算完成")
            
        except Exception as e:
            print(f"预计算线程出错: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_slice_metrics(self, slice_idx):
        """计算单个切片的指标"""
        metrics = {}
        
        try:
            # 检查GT数据是否存在
            if self.viewer.data_list[1] is None:
                return metrics
            
            gt_slice = self.viewer.data_list[1][:, :, slice_idx]
            
            for idx in range(2, self.viewer.num_folders):
                try:
                    # 检查模型数据是否存在
                    if self.viewer.data_list[idx] is None:
                        metrics[idx] = {
                            'dice_2d': 0.0,
                            'cldice_2d': 0.0
                        }
                        continue
                    
                    pred_slice = self.viewer.data_list[idx][:, :, slice_idx]
                    
                    if pred_slice is not None and gt_slice is not None:
                        dice_2d = self.viewer.calculate_dice(pred_slice, gt_slice)
                        cldice_2d = self.viewer.calculate_cldice(pred_slice, gt_slice, is_3d=False)
                        
                        metrics[idx] = {
                            'dice_2d': dice_2d,
                            'cldice_2d': cldice_2d
                        }
                    else:
                        metrics[idx] = {
                            'dice_2d': 0.0,
                            'cldice_2d': 0.0
                        }
                except Exception as e:
                    print(f"计算模型 {self.viewer.folder_labels[idx]} 切片 {slice_idx} 指标时出错: {e}")
                    metrics[idx] = {
                        'dice_2d': 0.0,
                        'cldice_2d': 0.0
                    }
        except Exception as e:
            print(f"计算切片 {slice_idx} 指标时出错: {e}")
        
        return metrics
    
    def stop(self):
        """停止线程"""
        self.running = False


class FilePreloadThread(QThread):
    """文件预加载线程 - 预加载前后文件的数据和指标"""
    progress = pyqtSignal(str)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.running = True
        self.lock = threading.Lock()
    
    def run(self):
        """预加载前后2个文件（进一步减少以避免内存溢出）"""
        try:
            if not hasattr(self.viewer, 'all_files'):
                return
            
            current_idx = self.viewer.current_file_index
            total_files = len(self.viewer.all_files)
            
            # 计算需要预加载的文件索引（减少到前后各2个）
            preload_indices = []
            for offset in range(1, 3):  # 前后各2个（原来是3个）
                if current_idx + offset < total_files:  # 优先加载后面的文件
                    preload_indices.append(current_idx + offset)
                if current_idx - offset >= 0:
                    preload_indices.append(current_idx - offset)
            
            for file_idx in preload_indices:
                if not self.running:
                    break
                
                filename = self.viewer.all_files[file_idx]
                
                # 检查是否已经缓存
                with self.lock:
                    if filename in self.viewer.file_cache:
                        continue
                
                self.progress.emit(f"正在预加载文件 {filename}...")
                
                try:
                    # 加载文件数据
                    data_list = []
                    for folder_path in self.viewer.folder_paths:
                        file_path = os.path.join(folder_path, filename)
                        if os.path.exists(file_path):
                            nii_img = nib.load(file_path)
                            data = nii_img.get_fdata()
                            data_list.append(data)
                        else:
                            data_list.append(None)
                    
                    # 找出有效切片
                    valid_slices = []
                    if data_list[1] is not None:
                        num_slices = data_list[1].shape[2]
                        for i in range(num_slices):
                            gt_slice = data_list[1][:, :, i]
                            if np.sum(gt_slice > 0) > 0:
                                valid_slices.append(i)
                    
                    # 计算3D指标（仅计算Dice，跳过clDice以节省内存和时间）
                    metrics_3d = {}
                    gt_data = data_list[1]
                    for model_idx in range(2, self.viewer.num_folders):
                        pred_data = data_list[model_idx]
                        if pred_data is not None and gt_data is not None:
                            dice_3d = self.viewer.calculate_dice(pred_data, gt_data)
                            # 跳过clDice计算以节省内存
                            metrics_3d[model_idx] = {
                                'dice_3d': dice_3d,
                                'cldice_3d': 0.0  # 延迟计算
                            }
                        else:
                            metrics_3d[model_idx] = {
                                'dice_3d': 0.0,
                                'cldice_3d': 0.0
                            }
                    
                    # 缓存数据
                    with self.lock:
                        self.viewer.file_cache[filename] = {
                            'data_list': data_list,
                            'valid_slices': valid_slices,
                            'metrics_3d': metrics_3d
                        }
                        self.viewer.metrics_2d_cache[filename] = {}
                    
                    self.progress.emit(f"文件 {filename} 预加载完成")
                    
                except MemoryError:
                    print(f"内存不足，停止预加载")
                    # 内存不足时停止预加载
                    break
                except Exception as e:
                    print(f"预加载文件 {filename} 时出错: {e}")
                    continue
            
            self.progress.emit("文件预加载完成")
            
        except Exception as e:
            print(f"文件预加载线程出错: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """停止线程"""
        self.running = False


class DataLoadThread(QThread):
    """数据加载线程"""
    progress = pyqtSignal(int, str)  # 进度值和消息
    finished = pyqtSignal()
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
    
    def run(self):
        """在后台线程中加载数据和计算指标"""
        try:
            # 加载第一个文件
            self.progress.emit(5, "正在扫描文件...")
            image_folder = self.viewer.folder_paths[0]
            nii_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.nii.gz')])
            
            if not nii_files:
                raise ValueError(f"在 {image_folder} 中没有找到.nii.gz文件")
            
            self.viewer.current_file = nii_files[0]
            self.viewer.all_files = nii_files
            self.viewer.current_file_index = 0
            
            # 加载数据
            self.progress.emit(10, "正在加载图像数据...")
            self.viewer.data_list = []
            
            for idx, folder_path in enumerate(self.viewer.folder_paths):
                progress_val = 10 + int((idx + 1) / len(self.viewer.folder_paths) * 20)
                self.progress.emit(progress_val, f"正在加载 {self.viewer.folder_labels[idx]} 数据...")
                
                file_path = os.path.join(folder_path, self.viewer.current_file)
                if os.path.exists(file_path):
                    nii_img = nib.load(file_path)
                    data = nii_img.get_fdata()
                    self.viewer.data_list.append(data)
                else:
                    print(f"警告: 文件不存在 {file_path}")
                    self.viewer.data_list.append(None)
            
            # 找出有效切片
            self.progress.emit(30, "正在分析有效切片...")
            if self.viewer.data_list[0] is not None:
                self.viewer.num_slices = self.viewer.data_list[0].shape[2]
                
                self.viewer.valid_slices = []
                gt_data = self.viewer.data_list[1]
                for i in range(self.viewer.num_slices):
                    gt_slice = gt_data[:, :, i]
                    if np.sum(gt_slice > 0) > 0:
                        self.viewer.valid_slices.append(i)
                
                if len(self.viewer.valid_slices) == 0:
                    raise ValueError("GT中没有找到任何掩膜")
                
                self.viewer.current_valid_index = len(self.viewer.valid_slices) // 2
                self.viewer.current_slice = self.viewer.valid_slices[self.viewer.current_valid_index]
            
            # 计算3D指标（使用多进程加速）
            self.progress.emit(35, "正在计算3D指标...")
            self.viewer.metrics_3d = {}
            
            gt_data = self.viewer.data_list[1]
            total_models = self.viewer.num_folders - 2
            
            for idx, model_idx in enumerate(range(2, self.viewer.num_folders)):
                progress_val = 35 + int((idx + 1) / total_models * 60)
                model_name = self.viewer.folder_labels[model_idx]
                self.progress.emit(progress_val, f"正在计算 {model_name} 的3D指标... ({idx+1}/{total_models})")
                
                pred_data = self.viewer.data_list[model_idx]
                
                if pred_data is not None and gt_data is not None:
                    try:
                        # 计算Dice
                        dice_3d = self.viewer.calculate_dice(pred_data, gt_data)
                        
                        # 计算clDice（耗时操作，可能内存不足）
                        self.progress.emit(progress_val, f"正在计算 {model_name} 的clDice... ({idx+1}/{total_models})")
                        try:
                            cldice_3d = self.viewer.calculate_cldice(pred_data, gt_data, is_3d=True)
                        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
                            print(f"内存不足，跳过 {model_name} 的clDice计算: {e}")
                            cldice_3d = 0.0
                        
                        self.viewer.metrics_3d[model_idx] = {
                            'dice_3d': dice_3d,
                            'cldice_3d': cldice_3d
                        }
                    except Exception as e:
                        print(f"计算 {model_name} 的3D指标时出错: {e}")
                        self.viewer.metrics_3d[model_idx] = {
                            'dice_3d': 0.0,
                            'cldice_3d': 0.0
                        }
                else:
                    self.viewer.metrics_3d[model_idx] = {
                        'dice_3d': 0.0,
                        'cldice_3d': 0.0
                    }
            
            self.progress.emit(95, "正在准备显示...")
            self.progress.emit(100, "加载完成！")
            self.finished.emit()
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit()


class MedicalImageViewer(QMainWindow):
    def __init__(self, root_folder):
        super().__init__()
        
        # 自动扫描子文件夹
        self.scan_folders(root_folder)
        
        # 初始化缓存
        self.metrics_3d = {}
        self.metrics_2d_cache = {}  # {filename: {slice_idx: {model_idx: {dice_2d, cldice_2d}}}}
        self.file_cache = {}  # {filename: {data_list, valid_slices, metrics_3d}}
        self.viz_3d_cache = {}  # 3D可视化缓存 {model_idx: pixmap}
        
        # 初始化线程
        self.precompute_thread = None
        self.preload_thread = None
        
        # 先创建UI（空白状态）
        self.init_ui()
        
        # 显示窗口
        self.show()
        
        # 在后台线程加载数据
        self.load_data_with_progress()
    
    def scan_folders(self, root_folder):
        """扫描根文件夹，自动识别Images、Labels和模型文件夹"""
        if not os.path.exists(root_folder):
            raise ValueError(f"根文件夹不存在: {root_folder}")
        
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(root_folder) 
                     if os.path.isdir(os.path.join(root_folder, f))]
        
        if len(subfolders) < 2:
            raise ValueError(f"根文件夹中至少需要2个子文件夹（Images和Labels）")
        
        # 初始化
        self.folder_paths = []
        self.folder_labels = []
        
        # 查找Images文件夹
        images_folder = None
        for folder in subfolders:
            if folder.lower() == 'images':
                images_folder = os.path.join(root_folder, folder)
                self.folder_paths.append(images_folder)
                self.folder_labels.append("Image")
                break
        
        if images_folder is None:
            raise ValueError("未找到Images文件夹")
        
        # 查找Labels文件夹（GT）
        labels_folder = None
        for folder in subfolders:
            if folder.lower() == 'labels':
                labels_folder = os.path.join(root_folder, folder)
                self.folder_paths.append(labels_folder)
                self.folder_labels.append("GT")
                break
        
        if labels_folder is None:
            raise ValueError("未找到Labels文件夹")
        
        # 添加其他模型文件夹（按字母顺序排序）
        model_folders = []
        for folder in subfolders:
            # 排除Images、Labels和svg文件夹
            if folder.lower() not in ['images', 'labels', 'svg']:
                model_folders.append(folder)
        
        model_folders.sort()  # 按字母顺序排序
        
        for folder in model_folders:
            folder_path = os.path.join(root_folder, folder)
            self.folder_paths.append(folder_path)
            self.folder_labels.append(folder)  # 使用文件夹名作为模型名
        
        self.num_folders = len(self.folder_paths)
        
        if self.num_folders > 10:
            print(f"警告: 检测到{self.num_folders}个文件夹，只显示前10个")
            self.folder_paths = self.folder_paths[:10]
            self.folder_labels = self.folder_labels[:10]
            self.num_folders = 10
        
        print(f"检测到{self.num_folders}个文件夹:")
        for i, (path, label) in enumerate(zip(self.folder_paths, self.folder_labels)):
            print(f"  {i+1}. {label}: {path}")
        
    def load_data_with_progress(self):
        """显示进度对话框并在后台加载数据"""
        # 创建进度对话框
        self.progress_dialog = QProgressDialog("正在初始化...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("加载数据")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        
        # 创建并启动加载线程
        self.load_thread = DataLoadThread(self)
        self.load_thread.progress.connect(self.update_progress)
        self.load_thread.finished.connect(self.on_load_finished)
        self.load_thread.start()
    
    def update_progress(self, value, message):
        """更新进度对话框"""
        self.progress_dialog.setValue(value)
        self.progress_dialog.setLabelText(message)
    
    def on_load_finished(self):
        """数据加载完成"""
        self.progress_dialog.close()
        
        # 缓存当前文件
        self.file_cache[self.current_file] = {
            'data_list': self.data_list,
            'valid_slices': self.valid_slices,
            'metrics_3d': self.metrics_3d
        }
        self.metrics_2d_cache[self.current_file] = {}
        
        # 更新滑动条范围
        if hasattr(self, 'valid_slices') and len(self.valid_slices) > 0:
            self.slice_slider.setMaximum(len(self.valid_slices) - 1)
            self.slice_slider.setValue(self.current_valid_index)
            self.slice_label.setText(f'切片: {self.current_valid_index + 1}/{len(self.valid_slices)}')
        
        # 更新显示
        self.update_display()
        
        # 启动预计算线程 - 计算当前文件所有切片的2D指标
        self.start_precompute()
        
        # 启动预加载线程 - 预加载前后文件
        self.start_preload()
    
    def start_precompute(self):
        """启动预计算线程"""
        if self.precompute_thread is not None and self.precompute_thread.isRunning():
            self.precompute_thread.stop()
            self.precompute_thread.wait()
        
        self.precompute_thread = PrecomputeThread(self)
        self.precompute_thread.progress.connect(self.on_precompute_progress)
        self.precompute_thread.start()
    
    def start_preload(self):
        """启动预加载线程"""
        if self.preload_thread is not None and self.preload_thread.isRunning():
            self.preload_thread.stop()
            self.preload_thread.wait()
        
        self.preload_thread = FilePreloadThread(self)
        self.preload_thread.progress.connect(self.on_preload_progress)
        self.preload_thread.start()
    
    def on_precompute_progress(self, message):
        """预计算进度回调"""
        # 可以在状态栏显示
        pass
    
    def on_preload_progress(self, message):
        """预加载进度回调"""
        # 可以在状态栏显示
        pass
    
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
        # 计算3D指标
        self.calculate_3d_metrics()
        
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
        
        # 获取屏幕尺寸
        screen = QApplication.desktop().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 文件信息标签
        self.file_label = QLabel("正在加载...")
        self.file_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.file_label)
        
        # 创建主滚动区域
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        main_scroll.setWidget(scroll_widget)
        
        self.image_2d_labels = []
        self.image_3d_labels = []
        self.title_labels = []
        
        # 每行最多显示5个
        max_per_row = 5
        num_rows = (self.num_folders + max_per_row - 1) // max_per_row
        
        # 根据屏幕大小计算图像尺寸
        available_width = screen_width * 0.9  # 使用90%的屏幕宽度
        cols_in_first_row = min(max_per_row, self.num_folders)
        img_size = min(300, int(available_width / cols_in_first_row) - 20)
        
        for row in range(num_rows):
            # 计算当前行的起始和结束索引
            start_idx = row * max_per_row
            end_idx = min(start_idx + max_per_row, self.num_folders)
            
            # 标题行
            title_layout = QHBoxLayout()
            for i in range(start_idx, end_idx):
                title_label = QLabel(self.folder_labels[i])
                title_label.setAlignment(Qt.AlignCenter)
                title_label.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 5px;")
                title_label.setFixedWidth(img_size)
                title_layout.addWidget(title_label)
                self.title_labels.append(title_label)
            scroll_layout.addLayout(title_layout)
            
            # 2D图像行
            image_2d_layout = QHBoxLayout()
            for i in range(start_idx, end_idx):
                img_2d_label = QLabel()
                img_2d_label.setAlignment(Qt.AlignCenter)
                img_2d_label.setFixedSize(img_size, img_size)
                img_2d_label.setStyleSheet("border: 2px solid gray; background-color: black;")
                img_2d_label.setScaledContents(False)
                image_2d_layout.addWidget(img_2d_label)
                self.image_2d_labels.append(img_2d_label)
            scroll_layout.addLayout(image_2d_layout)
            
            # 3D可视化或指标汇总行
            image_3d_layout = QHBoxLayout()
            for i in range(start_idx, end_idx):
                if i == 0:
                    # Image列：显示指标汇总（带滚动条）
                    metrics_scroll = QScrollArea()
                    metrics_scroll.setFixedSize(img_size, img_size)
                    metrics_scroll.setStyleSheet("border: 2px solid gray; background-color: #f5f5f5;")
                    metrics_scroll.setWidgetResizable(True)
                    
                    metrics_summary_label = QLabel()
                    metrics_summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
                    metrics_summary_label.setStyleSheet(
                        "font-size: 9pt; color: #333333; padding: 10px; background-color: #f5f5f5;"
                    )
                    metrics_summary_label.setWordWrap(True)
                    metrics_summary_label.setTextFormat(Qt.RichText)  # 支持HTML格式
                    metrics_scroll.setWidget(metrics_summary_label)
                    
                    image_3d_layout.addWidget(metrics_scroll)
                    self.image_3d_labels.append(metrics_summary_label)
                    self.metrics_summary_label = metrics_summary_label
                else:
                    # GT和模型列：显示3D可视化
                    img_3d_label = QLabel()
                    img_3d_label.setAlignment(Qt.AlignCenter)
                    img_3d_label.setFixedSize(img_size, img_size)
                    img_3d_label.setStyleSheet("border: 2px solid gray; background-color: white;")
                    img_3d_label.setScaledContents(False)
                    image_3d_layout.addWidget(img_3d_label)
                    self.image_3d_labels.append(img_3d_label)
            scroll_layout.addLayout(image_3d_layout)
            
            # 添加行间距
            if row < num_rows - 1:
                spacer = QLabel()
                spacer.setFixedHeight(15)
                scroll_layout.addWidget(spacer)
        
        main_layout.addWidget(main_scroll)
        
        # 控制区域
        control_layout = QHBoxLayout()
        
        # 上一个文件按钮
        self.prev_file_btn = QPushButton('上一个文件 (←)')
        self.prev_file_btn.clicked.connect(self.prev_file)
        control_layout.addWidget(self.prev_file_btn)
        
        # 切片滑动条
        self.slice_label = QLabel('切片: 0/0')
        control_layout.addWidget(self.slice_label)
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        control_layout.addWidget(self.slice_slider)
        
        # 下一个文件按钮
        self.next_file_btn = QPushButton('下一个文件 (→)')
        self.next_file_btn.clicked.connect(self.next_file)
        control_layout.addWidget(self.next_file_btn)
        
        main_layout.addLayout(control_layout)
        
        # 排序控制区域
        sort_layout = QHBoxLayout()
        
        # 2D切片排序按钮
        self.sort_2d_btn = QPushButton('按Ours的2D Dice排序切片')
        self.sort_2d_btn.clicked.connect(self.toggle_sort_2d_mode)
        self.sort_2d_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        sort_layout.addWidget(self.sort_2d_btn)
        
        # 3D文件排序按钮
        self.sort_3d_btn = QPushButton('按Ours的3D Dice排序文件')
        self.sort_3d_btn.clicked.connect(self.toggle_sort_3d_mode)
        self.sort_3d_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        sort_layout.addWidget(self.sort_3d_btn)
        
        # 恢复原序按钮
        self.reset_sort_btn = QPushButton('恢复原始顺序')
        self.reset_sort_btn.clicked.connect(self.reset_sort_mode)
        self.reset_sort_btn.setEnabled(False)
        sort_layout.addWidget(self.reset_sort_btn)
        
        # 保存SVG按钮
        self.save_svg_btn = QPushButton('保存当前视图为SVG')
        self.save_svg_btn.clicked.connect(self.save_current_view_as_svg)
        self.save_svg_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 5px;")
        sort_layout.addWidget(self.save_svg_btn)
        
        # 排序状态标签
        self.sort_status_label = QLabel('')
        self.sort_status_label.setStyleSheet("color: blue; font-weight: bold;")
        sort_layout.addWidget(self.sort_status_label)
        
        main_layout.addLayout(sort_layout)
        
        # 初始化排序状态
        self.is_sorted_2d = False  # 2D切片排序
        self.is_sorted_3d = False  # 3D文件排序
        self.sorted_slices = []
        self.sorted_files = []
        self.original_valid_slices = []
        self.original_all_files = []
        
        # 初始化SVG保存计数器
        self.svg_save_counter = 1
        
        # 设置窗口大小（根据屏幕大小自适应）
        window_width = min(screen_width * 0.95, img_size * cols_in_first_row + 50)
        window_height = min(screen_height * 0.9, 800)
        self.resize(int(window_width), int(window_height))
    
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
        """计算当前2D切片的指标（使用缓存）"""
        # 检查缓存
        if (self.current_file in self.metrics_2d_cache and 
            slice_idx in self.metrics_2d_cache[self.current_file]):
            return self.metrics_2d_cache[self.current_file][slice_idx]
        
        # 缓存未命中，实时计算
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
        
        # 缓存结果
        if self.current_file not in self.metrics_2d_cache:
            self.metrics_2d_cache[self.current_file] = {}
        self.metrics_2d_cache[self.current_file][slice_idx] = metrics_2d
        
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
        
        # 逆时针旋转90度以匹配2D切片的方向
        resized = np.rot90(resized, k=1)  # k=1表示逆时针旋转90度
        
        # 水平翻转（镜像）
        resized = np.fliplr(resized)
        
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
        # 检查数据是否已加载
        if not hasattr(self, 'data_list') or not self.data_list:
            return
        
        if not hasattr(self, 'valid_slices') or not self.valid_slices:
            return
        
        # 快速获取2D指标（如果缓存未命中，先显示图像，后台计算指标）
        metrics_2d = None
        if (self.current_file in self.metrics_2d_cache and 
            self.current_slice in self.metrics_2d_cache[self.current_file]):
            metrics_2d = self.metrics_2d_cache[self.current_file][self.current_slice]
        
        # 准备指标汇总文本
        metrics_summary_lines = []
        
        # 遍历所有文件夹（包括Image）
        for idx in range(self.num_folders):
            
            if self.data_list[idx] is not None:
                # 收集模型指标（跳过Image和GT）
                if idx > 1 and idx in self.metrics_3d:
                    dice_3d = self.metrics_3d[idx]['dice_3d']
                    cldice_3d = self.metrics_3d[idx]['cldice_3d']
                    
                    # 2D指标
                    if metrics_2d and idx in metrics_2d:
                        dice_2d = metrics_2d[idx]['dice_2d']
                        cldice_2d = metrics_2d[idx]['cldice_2d']
                    else:
                        dice_2d = 0.0
                        cldice_2d = 0.0
                    
                    # 添加到汇总
                    model_name = self.folder_labels[idx]
                    metrics_summary_lines.append(f"<b>{model_name}:</b>")
                    metrics_summary_lines.append(f"  3D: Dice={dice_3d:.3f}, clDice={cldice_3d:.3f}")
                    if metrics_2d and idx in metrics_2d:
                        metrics_summary_lines.append(f"  2D: Dice={dice_2d:.3f}, clDice={cldice_2d:.3f}")
                    else:
                        metrics_summary_lines.append(f"  2D: 计算中...")
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
                
                # 3D可视化显示（跳过Image列，使用缓存）
                if idx > 0:
                    # 检查缓存
                    cache_key = f"{self.current_file}_{idx}"
                    if cache_key in self.viz_3d_cache:
                        pixmap_3d = self.viz_3d_cache[cache_key]
                    else:
                        mask_data_3d = self.data_list[idx].transpose(2, 0, 1)
                        pixmap_3d = self.create_3d_visualization(mask_data_3d)
                        self.viz_3d_cache[cache_key] = pixmap_3d
                    
                    self.image_3d_labels[idx].setPixmap(pixmap_3d)
        
        # 更新Image列的指标汇总
        if hasattr(self, 'metrics_summary_label') and metrics_summary_lines:
            summary_text = "<b>模型指标汇总</b><br>" + "="*30 + "<br><br>" + "<br>".join(metrics_summary_lines)
            self.metrics_summary_label.setText(summary_text)
        
        # 更新文件信息
        if hasattr(self, 'current_file'):
            self.file_label.setText(f'文件: {self.current_file} ({self.current_file_index + 1}/{len(self.all_files)}) | '
                                   f'有效切片: {len(self.valid_slices)}/{self.num_slices}')
            self.slice_label.setText(f'切片: {self.current_valid_index + 1}/{len(self.valid_slices)} (实际: {self.current_slice + 1})')
        
        # 更新按钮状态
        if hasattr(self, 'all_files'):
            self.prev_file_btn.setEnabled(self.current_file_index > 0)
            self.next_file_btn.setEnabled(self.current_file_index < len(self.all_files) - 1)
        
        # 如果2D指标未缓存，在后台计算（不阻塞UI）
        if metrics_2d is None:
            self.compute_current_slice_metrics_async()
    
    def toggle_sort_2d_mode(self):
        """切换2D切片排序模式"""
        if self.is_sorted_2d:
            return
        
        # 找到Ours模型的索引
        ours_idx = None
        for idx, label in enumerate(self.folder_labels):
            if label.lower() == 'ours':
                ours_idx = idx
                break
        
        if ours_idx is None:
            self.sort_status_label.setText("未找到'Ours'模型！")
            return
        
        # 保存原始顺序
        self.original_valid_slices = self.valid_slices.copy()
        
        # 显示进度对话框
        total_slices = len(self.valid_slices)
        progress = QProgressDialog("正在初始化排序...", "取消", 0, total_slices, self)
        progress.setWindowTitle("按2D Dice排序切片")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        
        # 强制显示进度对话框
        QApplication.processEvents()
        
        # 使用多线程并发计算
        slice_dice_pairs = []
        completed = [0]  # 使用列表以便在闭包中修改
        lock = threading.Lock()
        
        def compute_slice_dice(slice_idx):
            """计算单个切片的Dice（仅计算Dice，不计算clDice以加速）"""
            # 获取或计算2D指标
            if (self.current_file in self.metrics_2d_cache and 
                slice_idx in self.metrics_2d_cache[self.current_file] and
                ours_idx in self.metrics_2d_cache[self.current_file][slice_idx]):
                # 从缓存获取
                metrics = self.metrics_2d_cache[self.current_file][slice_idx]
                dice_2d = metrics[ours_idx]['dice_2d']
            else:
                # 仅计算Dice（不计算clDice以加速）
                gt_slice = self.data_list[1][:, :, slice_idx]
                pred_slice = self.data_list[ours_idx][:, :, slice_idx]
                
                if pred_slice is not None and gt_slice is not None:
                    dice_2d = self.calculate_dice(pred_slice, gt_slice)
                else:
                    dice_2d = 0.0
            
            return (slice_idx, dice_2d)
        
        # 使用线程池并发计算（增加到12个工作线程）
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(compute_slice_dice, slice_idx): slice_idx 
                      for slice_idx in self.valid_slices}
            
            # 批量处理结果以减少UI更新频率
            update_interval = max(1, total_slices // 20)  # 每5%更新一次
            
            for future in futures:
                if progress.wasCanceled():
                    # 取消所有未完成的任务
                    executor.shutdown(wait=False)
                    self.valid_slices = self.original_valid_slices
                    return
                
                try:
                    result = future.result()
                    with lock:
                        slice_dice_pairs.append(result)
                        completed[0] += 1
                    
                    # 减少UI更新频率以加速
                    if completed[0] % update_interval == 0 or completed[0] == total_slices:
                        progress.setValue(completed[0])
                        progress.setLabelText(f"正在计算切片 Dice 指标...\n\n"
                                             f"进度: {completed[0]}/{total_slices} ({completed[0]*100//total_slices}%)\n"
                                             f"当前切片: {result[0]}, Dice: {result[1]:.3f}")
                        QApplication.processEvents()
                except Exception as e:
                    print(f"计算切片Dice时出错: {e}")
        
        # 完成所有切片的处理
        progress.setLabelText(f"正在排序 {total_slices} 个切片...")
        progress.setValue(total_slices)
        QApplication.processEvents()
        
        # 按Dice从高到低排序
        slice_dice_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 更新valid_slices为排序后的顺序
        self.sorted_slices = [pair[0] for pair in slice_dice_pairs]
        self.valid_slices = self.sorted_slices
        
        # 重置到第一个切片
        self.current_valid_index = 0
        self.current_slice = self.valid_slices[0]
        
        # 关闭进度对话框
        progress.close()
        
        # 更新UI
        self.is_sorted_2d = True
        self.sort_2d_btn.setEnabled(False)
        self.reset_sort_btn.setEnabled(True)
        self.sort_status_label.setText(f"已按Ours的2D Dice排序切片 (最高: {slice_dice_pairs[0][1]:.3f}, 最低: {slice_dice_pairs[-1][1]:.3f})")
        
        # 更新滑动条和显示
        self.slice_slider.setValue(0)
        self.update_display()
    
    def toggle_sort_3d_mode(self):
        """切换3D文件排序模式"""
        if self.is_sorted_3d:
            return
        
        # 找到Ours模型的索引
        ours_idx = None
        for idx, label in enumerate(self.folder_labels):
            if label.lower() == 'ours':
                ours_idx = idx
                break
        
        if ours_idx is None:
            self.sort_status_label.setText("未找到'Ours'模型！")
            return
        
        # 保存原始文件列表
        self.original_all_files = self.all_files.copy()
        
        # 显示进度对话框
        total_files = len(self.all_files)
        progress = QProgressDialog("正在初始化排序...", "取消", 0, total_files, self)
        progress.setWindowTitle("按3D Dice排序文件")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        
        # 强制显示进度对话框
        QApplication.processEvents()
        
        # 使用多线程并发计算
        file_dice_pairs = []
        completed = [0]
        lock = threading.Lock()
        
        def compute_file_dice(filename):
            """计算单个文件的3D Dice（轻量级版本，不缓存数据）"""
            try:
                # 检查缓存
                if filename in self.file_cache and ours_idx in self.file_cache[filename]['metrics_3d']:
                    metrics_3d = self.file_cache[filename]['metrics_3d']
                    dice_3d = metrics_3d[ours_idx]['dice_3d']
                    return (filename, dice_3d)
                
                # 仅加载GT和Ours模型的数据以节省内存
                gt_path = os.path.join(self.folder_paths[1], filename)
                ours_path = os.path.join(self.folder_paths[ours_idx], filename)
                
                if os.path.exists(gt_path) and os.path.exists(ours_path):
                    # 加载GT数据
                    gt_nii = nib.load(gt_path)
                    gt_data = gt_nii.get_fdata()
                    
                    # 加载Ours数据
                    ours_nii = nib.load(ours_path)
                    pred_data = ours_nii.get_fdata()
                    
                    # 计算Dice
                    dice_3d = self.calculate_dice(pred_data, gt_data)
                    
                    # 立即释放内存，不缓存数据
                    del pred_data
                    del ours_nii
                    del gt_data
                    del gt_nii
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                else:
                    dice_3d = 0.0
                
                return (filename, dice_3d)
            
            except MemoryError as e:
                print(f"内存不足，无法加载文件 {filename}: {e}")
                # 强制垃圾回收
                import gc
                gc.collect()
                return (filename, 0.0)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                return (filename, 0.0)
        
        # 使用线程池串行计算（max_workers=1以避免内存溢出）
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(compute_file_dice, filename): filename 
                      for filename in self.all_files}
            
            # 批量处理结果以减少UI更新频率
            update_interval = max(1, total_files // 20)  # 每5%更新一次
            
            for future in futures:
                if progress.wasCanceled():
                    # 取消所有未完成的任务
                    executor.shutdown(wait=False)
                    self.all_files = self.original_all_files
                    return
                
                try:
                    result = future.result()
                    with lock:
                        file_dice_pairs.append(result)
                        completed[0] += 1
                    
                    # 减少UI更新频率以加速
                    if completed[0] % update_interval == 0 or completed[0] == total_files:
                        progress.setValue(completed[0])
                        progress.setLabelText(f"正在计算文件 3D Dice 指标...\n\n"
                                             f"进度: {completed[0]}/{total_files} ({completed[0]*100//total_files}%)\n"
                                             f"当前文件: {result[0]}\n"
                                             f"Dice: {result[1]:.3f}")
                        QApplication.processEvents()
                except Exception as e:
                    print(f"计算文件3D Dice时出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 完成所有文件的处理
        progress.setLabelText(f"正在排序 {total_files} 个文件...")
        progress.setValue(total_files)
        QApplication.processEvents()
        
        # 按3D Dice从高到低排序
        file_dice_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 保存排序结果到临时字典（用于后续加载）
        self.sorted_dice_scores = {filename: dice for filename, dice in file_dice_pairs}
        
        # 更新文件列表为排序后的顺序
        self.sorted_files = [pair[0] for pair in file_dice_pairs]
        self.all_files = self.sorted_files
        
        # 重置到第一个文件
        self.current_file_index = 0
        self.current_file = self.all_files[0]
        
        # 加载第一个文件
        if self.current_file in self.file_cache:
            # 从缓存加载
            cached = self.file_cache[self.current_file]
            self.data_list = cached['data_list']
            self.valid_slices = cached['valid_slices']
            self.metrics_3d = cached['metrics_3d']
            self.num_slices = self.data_list[0].shape[2]
            
            self.current_valid_index = len(self.valid_slices) // 2
            self.current_slice = self.valid_slices[self.current_valid_index]
            
            # 关闭进度对话框
            progress.close()
            
            # 更新UI
            self.is_sorted_3d = True
            self.sort_3d_btn.setEnabled(False)
            self.reset_sort_btn.setEnabled(True)
            self.sort_status_label.setText(f"已按Ours的3D Dice排序文件 (最高: {file_dice_pairs[0][1]:.3f}, 最低: {file_dice_pairs[-1][1]:.3f})")
            
            # 更新滑动条和显示
            self.slice_slider.setMaximum(len(self.valid_slices) - 1)
            self.slice_slider.setValue(self.current_valid_index)
            
            # 更新文件切换按钮状态
            self.prev_file_btn.setEnabled(self.current_file_index > 0)
            self.next_file_btn.setEnabled(self.current_file_index < len(self.all_files) - 1)
            
            self.update_display()
            
            # 重新启动预计算和预加载
            self.start_precompute()
            self.start_preload()
        else:
            # 需要加载文件
            progress.close()
            
            # 更新UI状态
            self.is_sorted_3d = True
            self.sort_3d_btn.setEnabled(False)
            self.reset_sort_btn.setEnabled(True)
            self.sort_status_label.setText(f"已按Ours的3D Dice排序文件 (最高: {file_dice_pairs[0][1]:.3f}, 最低: {file_dice_pairs[-1][1]:.3f})")
            
            # 更新文件切换按钮状态（先禁用，加载完成后再启用）
            self.prev_file_btn.setEnabled(False)
            self.next_file_btn.setEnabled(False)
            
            # 异步加载第一个文件
            self.load_file_async(self.current_file)
    
    def reset_sort_mode(self):
        """恢复原始顺序"""
        if self.is_sorted_2d:
            # 恢复2D切片顺序
            self.valid_slices = self.original_valid_slices.copy()
            self.current_valid_index = len(self.valid_slices) // 2
            self.current_slice = self.valid_slices[self.current_valid_index]
            
            self.is_sorted_2d = False
            self.sort_2d_btn.setEnabled(True)
            
            self.slice_slider.setValue(self.current_valid_index)
            self.update_display()
        
        if self.is_sorted_3d:
            # 恢复3D文件顺序
            self.all_files = self.original_all_files.copy()
            
            # 找到当前文件在原始列表中的位置
            try:
                self.current_file_index = self.all_files.index(self.current_file)
            except ValueError:
                self.current_file_index = 0
                self.current_file = self.all_files[0]
                
                # 重新加载文件
                if self.current_file in self.file_cache:
                    cached = self.file_cache[self.current_file]
                    self.data_list = cached['data_list']
                    self.valid_slices = cached['valid_slices']
                    self.metrics_3d = cached['metrics_3d']
                    self.num_slices = self.data_list[0].shape[2]
                    
                    self.current_valid_index = len(self.valid_slices) // 2
                    self.current_slice = self.valid_slices[self.current_valid_index]
                    self.slice_slider.setMaximum(len(self.valid_slices) - 1)
                    self.slice_slider.setValue(self.current_valid_index)
            
            self.is_sorted_3d = False
            self.sort_3d_btn.setEnabled(True)
            
            self.update_display()
            self.start_precompute()
            self.start_preload()
        
        # 如果两种排序都已恢复，禁用恢复按钮
        if not self.is_sorted_2d and not self.is_sorted_3d:
            self.reset_sort_btn.setEnabled(False)
            self.sort_status_label.setText("")
    
    def calculate_2d_metrics_sync(self, slice_idx):
        """同步计算2D指标（用于排序）"""
        metrics = {}
        
        try:
            # 检查GT数据是否存在
            if self.data_list[1] is None:
                return metrics
            
            gt_slice = self.data_list[1][:, :, slice_idx]
            
            for idx in range(2, self.num_folders):
                try:
                    # 检查模型数据是否存在
                    if self.data_list[idx] is None:
                        metrics[idx] = {
                            'dice_2d': 0.0,
                            'cldice_2d': 0.0
                        }
                        continue
                    
                    pred_slice = self.data_list[idx][:, :, slice_idx]
                    
                    if pred_slice is not None and gt_slice is not None:
                        dice_2d = self.calculate_dice(pred_slice, gt_slice)
                        cldice_2d = self.calculate_cldice(pred_slice, gt_slice, is_3d=False)
                        
                        metrics[idx] = {
                            'dice_2d': dice_2d,
                            'cldice_2d': cldice_2d
                        }
                    else:
                        metrics[idx] = {
                            'dice_2d': 0.0,
                            'cldice_2d': 0.0
                        }
                except Exception as e:
                    print(f"计算模型 {self.folder_labels[idx]} 切片 {slice_idx} 指标时出错: {e}")
                    metrics[idx] = {
                        'dice_2d': 0.0,
                        'cldice_2d': 0.0
                    }
        except Exception as e:
            print(f"计算切片 {slice_idx} 指标时出错: {e}")
        
        # 缓存结果
        if self.current_file not in self.metrics_2d_cache:
            self.metrics_2d_cache[self.current_file] = {}
        self.metrics_2d_cache[self.current_file][slice_idx] = metrics
        
        return metrics
    
    def compute_current_slice_metrics_async(self):
        """异步计算当前切片的指标（不阻塞UI）"""
        # 使用QThread在后台计算
        class QuickMetricsThread(QThread):
            finished_signal = pyqtSignal(dict)
            
            def __init__(self, viewer, slice_idx):
                super().__init__()
                self.viewer = viewer
                self.slice_idx = slice_idx
            
            def run(self):
                try:
                    metrics = {}
                    gt_slice = self.viewer.data_list[1][:, :, self.slice_idx]
                    
                    for idx in range(2, self.viewer.num_folders):
                        pred_slice = self.viewer.data_list[idx][:, :, self.slice_idx]
                        
                        if pred_slice is not None and gt_slice is not None:
                            dice_2d = self.viewer.calculate_dice(pred_slice, gt_slice)
                            cldice_2d = self.viewer.calculate_cldice(pred_slice, gt_slice, is_3d=False)
                            
                            metrics[idx] = {
                                'dice_2d': dice_2d,
                                'cldice_2d': cldice_2d
                            }
                        else:
                            metrics[idx] = {
                                'dice_2d': 0.0,
                                'cldice_2d': 0.0
                            }
                    
                    self.finished_signal.emit(metrics)
                except Exception as e:
                    print(f"异步计算指标出错: {e}")
        
        # 创建并启动线程
        thread = QuickMetricsThread(self, self.current_slice)
        thread.finished_signal.connect(lambda m: self.on_metrics_computed(self.current_slice, m))
        thread.start()
        
        # 保存线程引用，防止被垃圾回收
        if not hasattr(self, '_metric_threads'):
            self._metric_threads = []
        self._metric_threads.append(thread)
    
    def on_metrics_computed(self, slice_idx, metrics):
        """指标计算完成回调"""
        # 缓存结果
        if self.current_file not in self.metrics_2d_cache:
            self.metrics_2d_cache[self.current_file] = {}
        self.metrics_2d_cache[self.current_file][slice_idx] = metrics
        
        # 如果当前还在这个切片，更新显示
        if self.current_slice == slice_idx:
            self.update_display()
    
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
            
            # 重置2D切片排序状态（但保持3D文件排序）
            if self.is_sorted_2d:
                self.is_sorted_2d = False
                self.sort_2d_btn.setEnabled(True)
                if not self.is_sorted_3d:
                    self.reset_sort_btn.setEnabled(False)
                    self.sort_status_label.setText("")
            
            # 检查缓存
            if self.current_file in self.file_cache:
                # 从缓存加载（即时）
                cached = self.file_cache[self.current_file]
                self.data_list = cached['data_list']
                self.valid_slices = cached['valid_slices']
                self.metrics_3d = cached['metrics_3d']
                self.num_slices = self.data_list[0].shape[2]
                
                self.current_valid_index = len(self.valid_slices) // 2
                self.current_slice = self.valid_slices[self.current_valid_index]
                self.slice_slider.setMaximum(len(self.valid_slices) - 1)
                self.slice_slider.setValue(self.current_valid_index)
                self.update_display()
                
                # 重新启动预计算和预加载
                self.start_precompute()
                self.start_preload()
            else:
                # 需要加载，使用后台线程
                self.load_file_async(self.current_file)
    
    def next_file(self):
        """下一个文件"""
        if self.current_file_index < len(self.all_files) - 1:
            self.current_file_index += 1
            self.current_file = self.all_files[self.current_file_index]
            
            # 重置2D切片排序状态（但保持3D文件排序）
            if self.is_sorted_2d:
                self.is_sorted_2d = False
                self.sort_2d_btn.setEnabled(True)
                if not self.is_sorted_3d:
                    self.reset_sort_btn.setEnabled(False)
                    self.sort_status_label.setText("")
            
            # 检查缓存
            if self.current_file in self.file_cache:
                # 从缓存加载（即时）
                cached = self.file_cache[self.current_file]
                self.data_list = cached['data_list']
                self.valid_slices = cached['valid_slices']
                self.metrics_3d = cached['metrics_3d']
                self.num_slices = self.data_list[0].shape[2]
                
                self.current_valid_index = len(self.valid_slices) // 2
                self.current_slice = self.valid_slices[self.current_valid_index]
                self.slice_slider.setMaximum(len(self.valid_slices) - 1)
                self.slice_slider.setValue(self.current_valid_index)
                self.update_display()
                
                # 重新启动预计算和预加载
                self.start_precompute()
                self.start_preload()
            else:
                # 需要加载，使用后台线程
                self.load_file_async(self.current_file)
    
    def load_file_async(self, filename):
        """异步加载文件"""
        # 显示加载提示
        self.file_label.setText(f"正在加载文件: {filename}...")
        
        # 禁用控制按钮
        self.prev_file_btn.setEnabled(False)
        self.next_file_btn.setEnabled(False)
        self.slice_slider.setEnabled(False)
        
        # 创建加载线程
        class FileLoadThread(QThread):
            finished_signal = pyqtSignal(dict)
            
            def __init__(self, viewer, filename):
                super().__init__()
                self.viewer = viewer
                self.filename = filename
            
            def run(self):
                try:
                    # 加载数据
                    data_list = []
                    for folder_path in self.viewer.folder_paths:
                        file_path = os.path.join(folder_path, self.filename)
                        if os.path.exists(file_path):
                            nii_img = nib.load(file_path)
                            data = nii_img.get_fdata()
                            data_list.append(data)
                        else:
                            data_list.append(None)
                    
                    # 找出有效切片
                    valid_slices = []
                    if data_list[1] is not None:
                        num_slices = data_list[1].shape[2]
                        for i in range(num_slices):
                            gt_slice = data_list[1][:, :, i]
                            if np.sum(gt_slice > 0) > 0:
                                valid_slices.append(i)
                    
                    # 计算3D指标
                    metrics_3d = {}
                    gt_data = data_list[1]
                    
                    # 检查是否有排序时保存的Dice分数
                    has_sorted_scores = hasattr(self.viewer, 'sorted_dice_scores') and self.filename in self.viewer.sorted_dice_scores
                    
                    for model_idx in range(2, self.viewer.num_folders):
                        pred_data = data_list[model_idx]
                        if pred_data is not None and gt_data is not None:
                            # 如果是Ours模型且有排序分数，使用排序时的分数
                            ours_idx = None
                            for idx, label in enumerate(self.viewer.folder_labels):
                                if label.lower() == 'ours':
                                    ours_idx = idx
                                    break
                            
                            if model_idx == ours_idx and has_sorted_scores:
                                dice_3d = self.viewer.sorted_dice_scores[self.filename]
                            else:
                                dice_3d = self.viewer.calculate_dice(pred_data, gt_data)
                            
                            # 尝试计算clDice，如果内存不足则跳过
                            try:
                                cldice_3d = self.viewer.calculate_cldice(pred_data, gt_data, is_3d=True)
                            except (MemoryError, np.core._exceptions._ArrayMemoryError):
                                print(f"内存不足，跳过 {self.viewer.folder_labels[model_idx]} 的clDice计算")
                                cldice_3d = 0.0
                            
                            metrics_3d[model_idx] = {
                                'dice_3d': dice_3d,
                                'cldice_3d': cldice_3d
                            }
                        else:
                            metrics_3d[model_idx] = {
                                'dice_3d': 0.0,
                                'cldice_3d': 0.0
                            }
                    
                    result = {
                        'data_list': data_list,
                        'valid_slices': valid_slices,
                        'metrics_3d': metrics_3d,
                        'num_slices': data_list[0].shape[2] if data_list[0] is not None else 0
                    }
                    
                    self.finished_signal.emit(result)
                    
                except Exception as e:
                    print(f"加载文件出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 创建并启动线程
        thread = FileLoadThread(self, filename)
        thread.finished_signal.connect(lambda r: self.on_file_loaded(filename, r))
        thread.start()
        
        # 保存线程引用
        if not hasattr(self, '_load_threads'):
            self._load_threads = []
        self._load_threads.append(thread)
    
    def on_file_loaded(self, filename, result):
        """文件加载完成回调"""
        # 更新数据
        self.data_list = result['data_list']
        self.valid_slices = result['valid_slices']
        self.metrics_3d = result['metrics_3d']
        self.num_slices = result['num_slices']
        
        # 缓存数据
        self.file_cache[filename] = {
            'data_list': self.data_list,
            'valid_slices': self.valid_slices,
            'metrics_3d': self.metrics_3d
        }
        self.metrics_2d_cache[filename] = {}
        
        # 更新UI
        self.current_valid_index = len(self.valid_slices) // 2
        self.current_slice = self.valid_slices[self.current_valid_index]
        self.slice_slider.setMaximum(len(self.valid_slices) - 1)
        self.slice_slider.setValue(self.current_valid_index)
        
        # 启用控制按钮
        self.prev_file_btn.setEnabled(True)
        self.next_file_btn.setEnabled(True)
        self.slice_slider.setEnabled(True)
        
        # 更新显示
        self.update_display()
        
        # 重新启动预计算和预加载
        self.start_precompute()
        self.start_preload()
    
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
    
    def save_current_view_as_svg(self):
        """保存当前视图为SVG格式"""
        try:
            # 检查数据是否已加载
            if not hasattr(self, 'data_list') or not self.data_list:
                print("没有数据可保存")
                return
            
            # 找到Ours模型的索引
            ours_idx = None
            for idx, label in enumerate(self.folder_labels):
                if label.lower() == 'ours':
                    ours_idx = idx
                    break
            
            # 创建新的显示顺序：Image, GT, Ours, 其他模型
            display_order = [0, 1]  # Image, GT
            if ours_idx is not None:
                display_order.append(ours_idx)  # Ours
            
            # 添加其他模型（排除Image, GT, Ours）
            for idx in range(2, self.num_folders):
                if idx != ours_idx:
                    display_order.append(idx)
            
            # 获取根文件夹名称
            root_folder_name = os.path.basename(os.path.dirname(self.folder_paths[0]))
            
            # 创建svg文件夹
            svg_folder = os.path.join(os.path.dirname(self.folder_paths[0]), 'svg')
            os.makedirs(svg_folder, exist_ok=True)
            
            # 生成文件名
            svg_filename = f"{root_folder_name}_{self.svg_save_counter}.svg"
            svg_path = os.path.join(svg_folder, svg_filename)
            
            # 图像尺寸设置
            img_width = 200  # 每个图像的宽度
            img_height = 200  # 每个图像的高度
            gap = 5  # 图像之间的间隔（减小到5px）
            title_height = 30  # 标题高度
            
            # 计算SVG总尺寸
            num_cols = len(display_order)
            total_width = num_cols * img_width + (num_cols - 1) * gap + 40  # 左右各20边距
            total_height = title_height + img_height * 2 + gap * 2 + 40  # 上下各20边距
            
            # 创建SVG文件
            import svgwrite
            from io import BytesIO
            import base64
            
            dwg = svgwrite.Drawing(svg_path, size=(total_width, total_height))
            
            # 添加白色背景
            dwg.add(dwg.rect(insert=(0, 0), size=(total_width, total_height), fill='white'))
            
            # 当前Y位置
            current_y = 20
            
            # 第一行：标题
            for col_idx, idx in enumerate(display_order):
                x_pos = 20 + col_idx * (img_width + gap)
                title_text = self.folder_labels[idx]
                dwg.add(dwg.text(title_text, 
                               insert=(x_pos + img_width / 2, current_y + 20),
                               text_anchor='middle',
                               font_size='16px',
                               font_weight='bold',
                               fill='black'))
            
            current_y += title_height
            
            # 第二行：2D切片图像（带掩膜）
            for col_idx, idx in enumerate(display_order):
                if self.data_list[idx] is not None:
                    # 获取当前切片数据
                    slice_data = self.data_list[idx][:, :, self.current_slice]
                    image_data = self.data_list[0][:, :, self.current_slice].T
                    
                    if idx == 0:
                        # Image列：只显示原始图像
                        img_array = self.create_overlay_image(image_data, np.zeros_like(image_data))
                    else:
                        # GT和模型列：显示图像+掩膜
                        mask_data = slice_data.T
                        img_array = self.create_overlay_image(image_data, mask_data)
                    
                    # 转换为PNG并编码为base64
                    from PIL import Image
                    pil_img = Image.fromarray(img_array)
                    buffer = BytesIO()
                    pil_img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # 添加到SVG
                    x_pos = 20 + col_idx * (img_width + gap)
                    dwg.add(dwg.image(href=f'data:image/png;base64,{img_base64}',
                                    insert=(x_pos, current_y),
                                    size=(img_width, img_height)))
            
            current_y += img_height + gap
            
            # 第三行：MIP 3D可视化
            for col_idx, idx in enumerate(display_order):
                if idx > 0 and self.data_list[idx] is not None:  # 跳过Image列
                    # 生成MIP图像
                    mask_data_3d = self.data_list[idx].transpose(2, 0, 1)
                    
                    # 创建MIP可视化
                    pixmap = self.create_3d_visualization(mask_data_3d)
                    
                    # 将QPixmap转换为numpy数组
                    qimg = pixmap.toImage()
                    
                    # 转换为RGB格式
                    qimg = qimg.convertToFormat(QImage.Format_RGB888)
                    
                    width = qimg.width()
                    height = qimg.height()
                    ptr = qimg.bits()
                    ptr.setsize(qimg.byteCount())
                    
                    # QImage的RGB888格式是RGB顺序
                    arr_rgb = np.array(ptr).reshape(height, width, 3)
                    
                    # 转换为PNG并编码为base64
                    pil_img = Image.fromarray(arr_rgb)
                    buffer = BytesIO()
                    pil_img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # 添加到SVG
                    x_pos = 20 + col_idx * (img_width + gap)
                    dwg.add(dwg.image(href=f'data:image/png;base64,{img_base64}',
                                    insert=(x_pos, current_y),
                                    size=(img_width, img_height)))
            
            # 保存SVG文件
            dwg.save()
            
            # 更新计数器
            self.svg_save_counter += 1
            
            print(f"SVG已保存到: {svg_path}")
            self.sort_status_label.setText(f"已保存: {svg_filename}")
            
        except Exception as e:
            print(f"保存SVG时出错: {e}")
            import traceback
            traceback.print_exc()
            self.sort_status_label.setText(f"保存失败: {str(e)}")


def main():
    """
    主函数 - 只需配置根文件夹路径
    """
    # 配置根文件夹路径
    # 根文件夹下应包含：
    #   - Images/     (原始图像)
    #   - Labels/     (真实标注GT)
    #   - 其他文件夹   (模型预测结果，文件夹名即为模型名)
    
    root_folder = r"E:\WebDownLoad\VisualCompareSeg\Brain"  # 修改为你的根文件夹路径
    
    # 检查根文件夹是否存在
    if not os.path.exists(root_folder):
        raise ValueError(f"根文件夹不存在: {root_folder}")
    
    # 创建应用
    app = QApplication(sys.argv)
    viewer = MedicalImageViewer(root_folder)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


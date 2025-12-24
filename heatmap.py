import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # pyplot import 전에 백엔드 설정 필요
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QSplitter, QComboBox,
    QLineEdit, QFormLayout, QHeaderView, QCheckBox, QTabWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class GraphGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph Generator - Heatmap & Boxplot")
        self.setGeometry(50, 50, 1800, 1000)

        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 스플리터로 좌우 분할
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 왼쪽: 탭 위젯으로 히트맵/박스플롯 설정 분리
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 공통: 데이터 테이블 영역
        self._setup_data_table(left_layout)

        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        left_layout.addWidget(self.tab_widget)

        # 히트맵 탭
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        self._setup_heatmap_tab(heatmap_layout)
        self.tab_widget.addTab(heatmap_tab, "히트맵")

        # 박스플롯 탭
        boxplot_tab = QWidget()
        boxplot_layout = QVBoxLayout(boxplot_tab)
        self._setup_boxplot_tab(boxplot_layout)
        self.tab_widget.addTab(boxplot_tab, "박스플롯")

        # 오른쪽: 미리보기 영역
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        preview_group = QGroupBox("미리보기")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel("그래프가 여기에 표시됩니다")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 500)
        self.preview_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        preview_layout.addWidget(self.preview_label)

        right_layout.addWidget(preview_group)

        # 스플리터에 추가
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([750, 850])

        # 초기 테이블 설정
        self.apply_table_size()

        # 현재 Figure 저장용
        self.current_fig = None
        self.current_graph_type = None

    def _setup_data_table(self, parent_layout):
        """공통 데이터 테이블 설정"""
        # 테이블 크기 설정
        size_group = QGroupBox("테이블 크기 설정")
        size_layout = QHBoxLayout(size_group)

        size_layout.addWidget(QLabel("행(지역) 수:"))
        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 50)
        self.row_spin.setValue(12)
        size_layout.addWidget(self.row_spin)

        size_layout.addWidget(QLabel("열(약제) 수:"))
        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 50)
        self.col_spin.setValue(9)
        size_layout.addWidget(self.col_spin)

        self.apply_size_btn = QPushButton("테이블 크기 적용")
        self.apply_size_btn.clicked.connect(self.apply_table_size)
        size_layout.addWidget(self.apply_size_btn)

        parent_layout.addWidget(size_group)

        # 데이터 테이블
        table_group = QGroupBox("데이터 입력 (첫 행: 약제명, 첫 열: 지역명, 나머지: 값)")
        table_layout = QVBoxLayout(table_group)

        self.data_table = QTableWidget()
        self.data_table.setRowCount(13)
        self.data_table.setColumnCount(10)
        self.data_table.setMinimumHeight(400)
        table_layout.addWidget(self.data_table)

        # 테이블 버튼들
        table_btn_layout = QHBoxLayout()

        self.load_csv_btn = QPushButton("CSV 불러오기")
        self.load_csv_btn.clicked.connect(self.load_csv)
        table_btn_layout.addWidget(self.load_csv_btn)

        self.load_sample_btn = QPushButton("샘플 데이터 불러오기")
        self.load_sample_btn.clicked.connect(self.load_sample_data)
        table_btn_layout.addWidget(self.load_sample_btn)

        self.clear_btn = QPushButton("테이블 초기화")
        self.clear_btn.clicked.connect(self.clear_table)
        table_btn_layout.addWidget(self.clear_btn)

        table_layout.addLayout(table_btn_layout)
        parent_layout.addWidget(table_group)

    def _setup_heatmap_tab(self, layout):
        """히트맵 탭 설정"""
        # 히트맵 설정
        settings_group = QGroupBox("히트맵 설정")
        settings_layout = QFormLayout(settings_group)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['RdYlGn', 'YlGnBu', 'YlOrRd', 'Blues', 'Greens', 'Reds', 'coolwarm', 'viridis'])
        settings_layout.addRow("색상 맵:", self.cmap_combo)

        self.vmin_spin = QSpinBox()
        self.vmin_spin.setRange(0, 100)
        self.vmin_spin.setValue(0)
        settings_layout.addRow("최소값:", self.vmin_spin)

        self.vmax_spin = QSpinBox()
        self.vmax_spin.setRange(0, 100)
        self.vmax_spin.setValue(100)
        settings_layout.addRow("최대값:", self.vmax_spin)

        self.cbar_label = QLineEdit("Mortality Rate (%)")
        settings_layout.addRow("컬러바 라벨:", self.cbar_label)

        layout.addWidget(settings_group)

        # 제목 및 글꼴 설정
        font_group = QGroupBox("제목 및 글꼴 설정")
        font_layout = QFormLayout(font_group)

        self.heatmap_title_edit = QLineEdit("")
        self.heatmap_title_edit.setPlaceholderText("히트맵 제목 입력 (선택사항)")
        font_layout.addRow("제목:", self.heatmap_title_edit)

        self.heatmap_title_size_spin = QSpinBox()
        self.heatmap_title_size_spin.setRange(8, 36)
        self.heatmap_title_size_spin.setValue(16)
        font_layout.addRow("제목 크기:", self.heatmap_title_size_spin)

        title_style_layout = QHBoxLayout()
        self.heatmap_title_bold_check = QCheckBox("볼드")
        self.heatmap_title_bold_check.setChecked(True)
        self.heatmap_title_italic_check = QCheckBox("이탤릭")
        title_style_layout.addWidget(self.heatmap_title_bold_check)
        title_style_layout.addWidget(self.heatmap_title_italic_check)
        title_style_layout.addStretch()
        font_layout.addRow("제목 스타일:", title_style_layout)

        self.axis_size_spin = QSpinBox()
        self.axis_size_spin.setRange(6, 24)
        self.axis_size_spin.setValue(11)
        font_layout.addRow("축 라벨 크기:", self.axis_size_spin)

        axis_style_layout = QHBoxLayout()
        self.axis_bold_check = QCheckBox("볼드")
        self.axis_bold_check.setChecked(True)
        self.axis_italic_check = QCheckBox("이탤릭")
        axis_style_layout.addWidget(self.axis_bold_check)
        axis_style_layout.addWidget(self.axis_italic_check)
        axis_style_layout.addStretch()
        font_layout.addRow("축 라벨 스타일:", axis_style_layout)

        self.annot_size_spin = QSpinBox()
        self.annot_size_spin.setRange(6, 20)
        self.annot_size_spin.setValue(11)
        font_layout.addRow("셀 값 크기:", self.annot_size_spin)

        annot_style_layout = QHBoxLayout()
        self.annot_bold_check = QCheckBox("볼드")
        self.annot_bold_check.setChecked(True)
        self.annot_italic_check = QCheckBox("이탤릭")
        annot_style_layout.addWidget(self.annot_bold_check)
        annot_style_layout.addWidget(self.annot_italic_check)
        annot_style_layout.addStretch()
        font_layout.addRow("셀 값 스타일:", annot_style_layout)

        self.cbar_size_spin = QSpinBox()
        self.cbar_size_spin.setRange(6, 20)
        self.cbar_size_spin.setValue(12)
        font_layout.addRow("컬러바 라벨 크기:", self.cbar_size_spin)

        self.heatmap_italic_texts_edit = QLineEdit("")
        self.heatmap_italic_texts_edit.setPlaceholderText("예: Spodoptera frugiperda (쉼표로 구분)")
        font_layout.addRow("이탤릭 텍스트:", self.heatmap_italic_texts_edit)

        layout.addWidget(font_group)

        # 생성 버튼
        btn_layout = QHBoxLayout()

        self.generate_heatmap_btn = QPushButton("히트맵 생성 / 미리보기")
        self.generate_heatmap_btn.clicked.connect(self.generate_heatmap)
        self.generate_heatmap_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.generate_heatmap_btn)

        self.save_heatmap_btn = QPushButton("이미지 저장")
        self.save_heatmap_btn.clicked.connect(self.save_current_graph)
        self.save_heatmap_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.save_heatmap_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def _setup_boxplot_tab(self, layout):
        """박스플롯 탭 설정"""
        # 박스플롯 설정
        settings_group = QGroupBox("박스플롯 설정")
        settings_layout = QFormLayout(settings_group)

        # 정렬 기준
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(['평균값 내림차순', '평균값 오름차순', '중앙값 내림차순', '중앙값 오름차순', '원본 순서'])
        settings_layout.addRow("정렬 기준:", self.sort_combo)

        # Y축 범위
        self.box_ymin_spin = QSpinBox()
        self.box_ymin_spin.setRange(0, 100)
        self.box_ymin_spin.setValue(0)
        settings_layout.addRow("Y축 최소값:", self.box_ymin_spin)

        self.box_ymax_spin = QSpinBox()
        self.box_ymax_spin.setRange(0, 100)
        self.box_ymax_spin.setValue(100)
        settings_layout.addRow("Y축 최대값:", self.box_ymax_spin)

        # 평균값 표시
        self.show_mean_check = QCheckBox("평균값 마커 표시")
        self.show_mean_check.setChecked(True)
        settings_layout.addRow("", self.show_mean_check)

        # 평균값 라벨 표시
        self.show_mean_label_check = QCheckBox("평균값 숫자 표시")
        self.show_mean_label_check.setChecked(True)
        settings_layout.addRow("", self.show_mean_label_check)

        # 박스 색상 팔레트
        self.box_palette_combo = QComboBox()
        self.box_palette_combo.addItems(['Set2', 'Set3', 'Pastel1', 'Pastel2', 'Paired', 'husl', 'tab10', 'tab20'])
        settings_layout.addRow("색상 팔레트:", self.box_palette_combo)

        layout.addWidget(settings_group)

        # 제목 및 라벨 설정
        label_group = QGroupBox("제목 및 라벨 설정")
        label_layout = QFormLayout(label_group)

        self.boxplot_title_edit = QLineEdit("")
        self.boxplot_title_edit.setPlaceholderText("박스플롯 제목 입력")
        label_layout.addRow("제목:", self.boxplot_title_edit)

        self.boxplot_subtitle_edit = QLineEdit("")
        self.boxplot_subtitle_edit.setPlaceholderText("부제목 입력 (선택사항)")
        label_layout.addRow("부제목:", self.boxplot_subtitle_edit)

        self.boxplot_xlabel_edit = QLineEdit("Active Ingredient")
        label_layout.addRow("X축 라벨:", self.boxplot_xlabel_edit)

        self.boxplot_ylabel_edit = QLineEdit("Control Efficacy (%)")
        label_layout.addRow("Y축 라벨:", self.boxplot_ylabel_edit)

        layout.addWidget(label_group)

        # 글꼴 설정
        font_group = QGroupBox("글꼴 설정")
        font_layout = QFormLayout(font_group)

        self.boxplot_title_size_spin = QSpinBox()
        self.boxplot_title_size_spin.setRange(8, 36)
        self.boxplot_title_size_spin.setValue(14)
        font_layout.addRow("제목 크기:", self.boxplot_title_size_spin)

        title_style_layout = QHBoxLayout()
        self.boxplot_title_bold_check = QCheckBox("볼드")
        self.boxplot_title_bold_check.setChecked(True)
        self.boxplot_title_italic_check = QCheckBox("이탤릭")
        title_style_layout.addWidget(self.boxplot_title_bold_check)
        title_style_layout.addWidget(self.boxplot_title_italic_check)
        title_style_layout.addStretch()
        font_layout.addRow("제목 스타일:", title_style_layout)

        self.boxplot_axis_size_spin = QSpinBox()
        self.boxplot_axis_size_spin.setRange(6, 24)
        self.boxplot_axis_size_spin.setValue(10)
        font_layout.addRow("축 라벨 크기:", self.boxplot_axis_size_spin)

        self.boxplot_tick_size_spin = QSpinBox()
        self.boxplot_tick_size_spin.setRange(6, 20)
        self.boxplot_tick_size_spin.setValue(9)
        font_layout.addRow("눈금 라벨 크기:", self.boxplot_tick_size_spin)

        # 눈금 라벨 스타일
        tick_style_layout = QHBoxLayout()
        self.boxplot_tick_bold_check = QCheckBox("볼드")
        self.boxplot_tick_bold_check.setChecked(True)
        self.boxplot_tick_italic_check = QCheckBox("이탤릭")
        tick_style_layout.addWidget(self.boxplot_tick_bold_check)
        tick_style_layout.addWidget(self.boxplot_tick_italic_check)
        tick_style_layout.addStretch()
        font_layout.addRow("눈금 라벨 스타일:", tick_style_layout)

        self.boxplot_italic_texts_edit = QLineEdit("")
        self.boxplot_italic_texts_edit.setPlaceholderText("예: Spodoptera frugiperda (쉼표로 구분)")
        font_layout.addRow("이탤릭 텍스트:", self.boxplot_italic_texts_edit)

        layout.addWidget(font_group)

        # 생성 버튼
        btn_layout = QHBoxLayout()

        self.generate_boxplot_btn = QPushButton("박스플롯 생성 / 미리보기")
        self.generate_boxplot_btn.clicked.connect(self.generate_boxplot)
        self.generate_boxplot_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.generate_boxplot_btn)

        self.save_boxplot_btn = QPushButton("이미지 저장")
        self.save_boxplot_btn.clicked.connect(self.save_current_graph)
        self.save_boxplot_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.save_boxplot_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def apply_table_size(self):
        """테이블 크기 적용"""
        rows = self.row_spin.value() + 1
        cols = self.col_spin.value() + 1

        self.data_table.setRowCount(rows)
        self.data_table.setColumnCount(cols)

        self.data_table.setItem(0, 0, QTableWidgetItem(""))

        self.data_table.horizontalHeader().setVisible(False)
        self.data_table.verticalHeader().setVisible(False)

        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)

        for i in range(cols):
            self.data_table.setColumnWidth(i, 100)
        for i in range(rows):
            self.data_table.setRowHeight(i, 30)

    def clear_table(self):
        """테이블 초기화"""
        for i in range(self.data_table.rowCount()):
            for j in range(self.data_table.columnCount()):
                self.data_table.setItem(i, j, QTableWidgetItem(""))

    def load_sample_data(self):
        """샘플 데이터 불러오기"""
        columns = ['', 'Acrinathrin', 'Acetamiprid', 'Dinotefuran', 'Cyclaniliprole',
                   'Chlorfluazuron', 'Chlorfenapyr', 'Abamectin', 'Spinetoram', 'Emamectin benzoate']

        data = [
            ['Gochang', 13, 11, 37, 63, 42, 96, 89, 98, 76],
            ['Gwangju', 14, 11, 34, 39, 64, 78, 57, 92, 72],
            ['Namwon', 28, 27, 49, 32, 66, 85, 72, 97, 85],
            ['Damyang', 9, 20, 36, 54, 70, 93, 63, 97, 71],
            ['Boseong', 35, 31, 46, 48, 97, 77, 59, 81, 74],
            ['Suncheon', 51, 66, 85, 57, 77, 82, 97, 99, 99],
            ['Yeonggwang', 7, 6, 22, 36, 78, 75, 50, 81, 66],
            ['Yeongam', 21, 27, 53, 29, 79, 77, 51, 76, 82],
            ['Iksan', 31, 46, 84, 64, 75, 96, 85, 87, 88],
            ['Jeongeup', 15, 2, 37, 35, 78, 82, 59, 97, 88],
            ['Haenam', 54, 45, 83, 62, 77, 91, 71, 99, 91],
            ['Hwasun', 8, 13, 26, 38, 48, 83, 88, 96, 86],
        ]

        self.row_spin.setValue(12)
        self.col_spin.setValue(9)
        self.apply_table_size()

        for j, col in enumerate(columns):
            self.data_table.setItem(0, j, QTableWidgetItem(col))

        for i, row in enumerate(data):
            for j, val in enumerate(row):
                self.data_table.setItem(i + 1, j, QTableWidgetItem(str(val)))

    def load_csv(self):
        """CSV 파일 불러오기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 선택", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            reply = QMessageBox.question(
                self, "CSV 불러오기 옵션",
                "CSV 파일의 첫 번째 열을 행 이름(인덱스)으로 사용하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            use_index_col = (reply == QMessageBox.Yes)

            try:
                if use_index_col:
                    df = pd.read_csv(file_path, index_col=0)
                else:
                    df = pd.read_csv(file_path)
                    df.index = [f"Row{i+1}" for i in range(len(df))]

                self.row_spin.setValue(len(df))
                self.col_spin.setValue(len(df.columns))
                self.apply_table_size()

                self.data_table.setItem(0, 0, QTableWidgetItem(""))
                for j, col in enumerate(df.columns):
                    self.data_table.setItem(0, j + 1, QTableWidgetItem(str(col)))

                for i in range(len(df)):
                    self.data_table.setItem(i + 1, 0, QTableWidgetItem(str(df.index[i])))
                    for j in range(len(df.columns)):
                        self.data_table.setItem(i + 1, j + 1, QTableWidgetItem(str(df.iloc[i, j])))

                QMessageBox.information(self, "성공", "CSV 파일을 불러왔습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일을 불러오는 중 오류 발생:\n{str(e)}")

    def get_table_data(self):
        """테이블에서 데이터 추출 및 검증"""
        rows = self.data_table.rowCount()
        cols = self.data_table.columnCount()

        col_names = []
        for j in range(1, cols):
            item = self.data_table.item(0, j)
            col_names.append(item.text().strip() if item and item.text().strip() else f"Col{j}")

        row_names = []
        data = []
        invalid_cells = []

        for i in range(1, rows):
            item = self.data_table.item(i, 0)
            row_names.append(item.text().strip() if item and item.text().strip() else f"Row{i}")

            row_data = []
            for j in range(1, cols):
                item = self.data_table.item(i, j)
                cell_text = item.text().strip() if item else ""

                if not cell_text:
                    val = np.nan
                else:
                    try:
                        val = float(cell_text)
                    except ValueError:
                        val = np.nan
                        invalid_cells.append(f"({i}, {j}): '{cell_text}'")
                row_data.append(val)
            data.append(row_data)

        df = pd.DataFrame(data, index=row_names, columns=col_names)

        if invalid_cells:
            warning_msg = f"다음 셀의 값이 숫자가 아니어서 NaN으로 처리됩니다:\n"
            warning_msg += "\n".join(invalid_cells[:10])
            if len(invalid_cells) > 10:
                warning_msg += f"\n... 외 {len(invalid_cells) - 10}개"
            QMessageBox.warning(self, "데이터 경고", warning_msg)

        return df

    def _get_font_weight(self, is_bold):
        return 'bold' if is_bold else 'normal'

    def _get_font_style(self, is_italic):
        return 'italic' if is_italic else 'normal'

    def _get_italic_texts(self, source='heatmap'):
        if source == 'heatmap':
            text = self.heatmap_italic_texts_edit.text().strip()
        else:
            text = self.boxplot_italic_texts_edit.text().strip()
        if not text:
            return []
        return [t.strip() for t in text.split(',') if t.strip()]

    def _apply_italic_to_label(self, label, italic_texts, use_bold=False):
        result = label
        for italic_text in italic_texts:
            if italic_text in result:
                escaped_text = italic_text.replace(' ', '\\ ')
                if use_bold:
                    result = result.replace(italic_text, f'$\\bf{{\\it{{{escaped_text}}}}}$')
                else:
                    result = result.replace(italic_text, f'$\\it{{{escaped_text}}}$')
        return result

    def _create_heatmap_figure(self, df):
        """히트맵 Figure 생성"""
        fig, ax = plt.subplots(figsize=(14, 10))

        italic_texts = self._get_italic_texts('heatmap')

        if italic_texts:
            new_columns = [self._apply_italic_to_label(col, italic_texts) for col in df.columns]
            new_index = [self._apply_italic_to_label(idx, italic_texts) for idx in df.index]
            df = df.copy()
            df.columns = new_columns
            df.index = new_index

        annot_weight = self._get_font_weight(self.annot_bold_check.isChecked())
        annot_style = self._get_font_style(self.annot_italic_check.isChecked())

        sns.heatmap(
            df,
            annot=True,
            fmt='.0f',
            cmap=self.cmap_combo.currentText(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': self.cbar_label.text(), 'shrink': 0.8},
            annot_kws={
                'size': self.annot_size_spin.value(),
                'weight': annot_weight,
                'style': annot_style
            },
            ax=ax
        )

        axis_weight = self._get_font_weight(self.axis_bold_check.isChecked())
        axis_style = self._get_font_style(self.axis_italic_check.isChecked())
        axis_size = self.axis_size_spin.value()

        plt.xticks(rotation=45, ha='right', fontsize=axis_size,
                   fontweight=axis_weight, fontstyle=axis_style)
        plt.yticks(rotation=0, fontsize=axis_size,
                   fontweight=axis_weight, fontstyle=axis_style)

        ax.set_xlabel('', fontsize=axis_size)
        ax.set_ylabel('', fontsize=axis_size)

        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(self.cbar_label.text(), fontsize=self.cbar_size_spin.value())

        title_text = self.heatmap_title_edit.text().strip()
        if title_text:
            title_is_bold = self.heatmap_title_bold_check.isChecked()
            title_is_italic = self.heatmap_title_italic_check.isChecked()
            title_weight = self._get_font_weight(title_is_bold)
            title_style = self._get_font_style(title_is_italic)

            if italic_texts:
                title_text = self._apply_italic_to_label(title_text, italic_texts, use_bold=title_is_bold)

            ax.set_title(
                title_text,
                fontsize=self.heatmap_title_size_spin.value(),
                fontweight=title_weight,
                fontstyle=title_style,
                pad=20
            )

        plt.tight_layout()
        return fig

    def _create_boxplot_figure(self, df):
        """박스플롯 Figure 생성"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # 데이터를 long format으로 변환
        df_melted = df.melt(var_name='Category', value_name='Value')
        df_melted = df_melted.dropna()

        # 정렬 기준에 따라 순서 결정
        sort_option = self.sort_combo.currentText()
        if sort_option == '평균값 내림차순':
            order = df.mean().sort_values(ascending=False).index.tolist()
        elif sort_option == '평균값 오름차순':
            order = df.mean().sort_values(ascending=True).index.tolist()
        elif sort_option == '중앙값 내림차순':
            order = df.median().sort_values(ascending=False).index.tolist()
        elif sort_option == '중앙값 오름차순':
            order = df.median().sort_values(ascending=True).index.tolist()
        else:
            order = df.columns.tolist()

        # 이탤릭 텍스트 처리
        italic_texts = self._get_italic_texts('boxplot')
        title_is_bold = self.boxplot_title_bold_check.isChecked()

        # 박스플롯 그리기
        palette = self.box_palette_combo.currentText()
        colors = sns.color_palette(palette, len(order))

        bp = ax.boxplot(
            [df[col].dropna().values for col in order],
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='gray', markersize=6)
        )

        # 박스 색상 설정
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('darkgray')
            patch.set_linewidth(1.5)

        # 평균값 마커 표시
        if self.show_mean_check.isChecked():
            means = [df[col].mean() for col in order]
            ax.scatter(range(1, len(order) + 1), means, color='red', marker='D', s=50, zorder=5, label='Mean')

            # 평균값 라벨 표시
            if self.show_mean_label_check.isChecked():
                for i, mean in enumerate(means):
                    ax.annotate(
                        f'{mean:.1f}',
                        (i + 1, mean),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
                    )

        # 눈금 라벨 스타일 설정
        tick_weight = self._get_font_weight(self.boxplot_tick_bold_check.isChecked())
        tick_style = self._get_font_style(self.boxplot_tick_italic_check.isChecked())
        tick_size = self.boxplot_tick_size_spin.value()

        # X축 라벨 설정
        x_labels = order
        if italic_texts:
            x_labels = [self._apply_italic_to_label(label, italic_texts, use_bold=self.boxplot_tick_bold_check.isChecked()) for label in order]
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=tick_size, fontweight=tick_weight, fontstyle=tick_style)

        # Y축 범위 설정
        ax.set_ylim(self.box_ymin_spin.value(), self.box_ymax_spin.value())

        # Y축 눈금 라벨 스타일 적용
        ax.tick_params(axis='y', labelsize=tick_size)
        for label in ax.get_yticklabels():
            label.set_fontweight(tick_weight)
            label.set_fontstyle(tick_style)

        # 축 라벨 설정
        ax.set_xlabel(self.boxplot_xlabel_edit.text(), fontsize=self.boxplot_axis_size_spin.value(), fontweight='bold')
        ax.set_ylabel(self.boxplot_ylabel_edit.text(), fontsize=self.boxplot_axis_size_spin.value(), fontweight='bold')

        # 제목 설정
        title_text = self.boxplot_title_edit.text().strip()
        subtitle_text = self.boxplot_subtitle_edit.text().strip()

        if title_text:
            title_weight = self._get_font_weight(title_is_bold)
            title_style = self._get_font_style(self.boxplot_title_italic_check.isChecked())

            if italic_texts:
                title_text = self._apply_italic_to_label(title_text, italic_texts, use_bold=title_is_bold)

            full_title = title_text
            if subtitle_text:
                if italic_texts:
                    subtitle_text = self._apply_italic_to_label(subtitle_text, italic_texts, use_bold=title_is_bold)
                full_title = f"{title_text}\n({subtitle_text})"

            ax.set_title(
                full_title,
                fontsize=self.boxplot_title_size_spin.value(),
                fontweight=title_weight,
                fontstyle=title_style,
                pad=15
            )

        # 범례 추가
        if self.show_mean_check.isChecked():
            ax.legend(loc='upper right', fontsize=10)

        # 그리드 추가
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        return fig

    def generate_heatmap(self):
        """히트맵 생성 및 미리보기"""
        try:
            df = self.get_table_data()

            if df.empty or df.isna().all().all():
                QMessageBox.warning(self, "경고", "유효한 데이터를 입력해주세요.")
                return

            if self.current_fig:
                plt.close(self.current_fig)

            self.current_fig = self._create_heatmap_figure(df)
            self.current_graph_type = 'heatmap'

            self._show_preview(self.current_fig)

        except Exception as e:
            QMessageBox.critical(self, "오류", f"히트맵 생성 중 오류 발생:\n{str(e)}")

    def generate_boxplot(self):
        """박스플롯 생성 및 미리보기"""
        try:
            df = self.get_table_data()

            if df.empty or df.isna().all().all():
                QMessageBox.warning(self, "경고", "유효한 데이터를 입력해주세요.")
                return

            if self.current_fig:
                plt.close(self.current_fig)

            self.current_fig = self._create_boxplot_figure(df)
            self.current_graph_type = 'boxplot'

            self._show_preview(self.current_fig)

        except Exception as e:
            QMessageBox.critical(self, "오류", f"박스플롯 생성 중 오류 발생:\n{str(e)}")

    def _show_preview(self, fig):
        """미리보기 표시"""
        with BytesIO() as buf:
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())

        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)

    def save_current_graph(self):
        """현재 그래프 저장"""
        if self.current_fig is None:
            QMessageBox.warning(self, "경고", "먼저 그래프를 생성해주세요.")
            return

        default_name = f"{self.current_graph_type}.png" if self.current_graph_type else "graph.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "이미지 저장", default_name, "PNG Files (*.png);;All Files (*)"
        )

        if file_path:
            try:
                df = self.get_table_data()
                if self.current_graph_type == 'heatmap':
                    fig = self._create_heatmap_figure(df)
                else:
                    fig = self._create_boxplot_figure(df)

                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                QMessageBox.information(self, "성공", f"이미지가 저장되었습니다:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 중 오류 발생:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = GraphGenerator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

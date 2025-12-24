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
    QLineEdit, QFormLayout, QHeaderView, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class HeatmapGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heatmap Generator")
        self.setGeometry(100, 100, 1400, 800)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 스플리터로 좌우 분할
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 왼쪽: 데이터 입력 영역
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
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
        
        left_layout.addWidget(size_group)
        
        # 데이터 테이블
        table_group = QGroupBox("데이터 입력 (첫 행: 약제명, 첫 열: 지역명, 나머지: 값)")
        table_layout = QVBoxLayout(table_group)
        
        self.data_table = QTableWidget()
        self.data_table.setRowCount(13)  # 12 지역 + 1 헤더
        self.data_table.setColumnCount(10)  # 9 약제 + 1 헤더
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
        left_layout.addWidget(table_group)
        
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

        left_layout.addWidget(settings_group)

        # 제목 및 글꼴 설정
        font_group = QGroupBox("제목 및 글꼴 설정")
        font_layout = QFormLayout(font_group)

        # 제목 입력
        self.title_edit = QLineEdit("")
        self.title_edit.setPlaceholderText("히트맵 제목 입력 (선택사항)")
        font_layout.addRow("제목:", self.title_edit)

        # 제목 글꼴 크기
        self.title_size_spin = QSpinBox()
        self.title_size_spin.setRange(8, 36)
        self.title_size_spin.setValue(16)
        font_layout.addRow("제목 크기:", self.title_size_spin)

        # 제목 스타일
        title_style_layout = QHBoxLayout()
        self.title_bold_check = QCheckBox("볼드")
        self.title_bold_check.setChecked(True)
        self.title_italic_check = QCheckBox("이탤릭")
        title_style_layout.addWidget(self.title_bold_check)
        title_style_layout.addWidget(self.title_italic_check)
        title_style_layout.addStretch()
        font_layout.addRow("제목 스타일:", title_style_layout)

        # 축 라벨 글꼴 크기
        self.axis_size_spin = QSpinBox()
        self.axis_size_spin.setRange(6, 24)
        self.axis_size_spin.setValue(11)
        font_layout.addRow("축 라벨 크기:", self.axis_size_spin)

        # 축 라벨 스타일
        axis_style_layout = QHBoxLayout()
        self.axis_bold_check = QCheckBox("볼드")
        self.axis_bold_check.setChecked(True)
        self.axis_italic_check = QCheckBox("이탤릭")
        axis_style_layout.addWidget(self.axis_bold_check)
        axis_style_layout.addWidget(self.axis_italic_check)
        axis_style_layout.addStretch()
        font_layout.addRow("축 라벨 스타일:", axis_style_layout)

        # 셀 값 글꼴 크기
        self.annot_size_spin = QSpinBox()
        self.annot_size_spin.setRange(6, 20)
        self.annot_size_spin.setValue(11)
        font_layout.addRow("셀 값 크기:", self.annot_size_spin)

        # 셀 값 스타일
        annot_style_layout = QHBoxLayout()
        self.annot_bold_check = QCheckBox("볼드")
        self.annot_bold_check.setChecked(True)
        self.annot_italic_check = QCheckBox("이탤릭")
        annot_style_layout.addWidget(self.annot_bold_check)
        annot_style_layout.addWidget(self.annot_italic_check)
        annot_style_layout.addStretch()
        font_layout.addRow("셀 값 스타일:", annot_style_layout)

        # 컬러바 라벨 크기
        self.cbar_size_spin = QSpinBox()
        self.cbar_size_spin.setRange(6, 20)
        self.cbar_size_spin.setValue(12)
        font_layout.addRow("컬러바 라벨 크기:", self.cbar_size_spin)

        # 특정 텍스트 이탤릭 설정
        self.italic_texts_edit = QLineEdit("")
        self.italic_texts_edit.setPlaceholderText("예: Spodoptera, frugiperda (쉼표로 구분)")
        font_layout.addRow("이탤릭 텍스트:", self.italic_texts_edit)

        left_layout.addWidget(font_group)
        
        # 생성 버튼
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("히트맵 생성 / 미리보기")
        self.generate_btn.clicked.connect(self.generate_heatmap)
        self.generate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.generate_btn)
        
        self.save_btn = QPushButton("이미지 저장")
        self.save_btn.clicked.connect(self.save_heatmap)
        self.save_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.save_btn)
        
        left_layout.addLayout(btn_layout)
        
        # 오른쪽: 미리보기 영역
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        preview_group = QGroupBox("미리보기")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("히트맵이 여기에 표시됩니다")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 500)
        self.preview_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        preview_layout.addWidget(self.preview_label)
        
        right_layout.addWidget(preview_group)
        
        # 스플리터에 추가
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 800])
        
        # 초기 테이블 설정
        self.apply_table_size()
        
    def apply_table_size(self):
        """테이블 크기 적용"""
        rows = self.row_spin.value() + 1  # +1 for header
        cols = self.col_spin.value() + 1  # +1 for row labels
        
        self.data_table.setRowCount(rows)
        self.data_table.setColumnCount(cols)
        
        # 첫 번째 셀 비우기
        self.data_table.setItem(0, 0, QTableWidgetItem(""))
        
        # 헤더 스타일 설정
        self.data_table.horizontalHeader().setVisible(False)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
    def clear_table(self):
        """테이블 초기화"""
        for i in range(self.data_table.rowCount()):
            for j in range(self.data_table.columnCount()):
                self.data_table.setItem(i, j, QTableWidgetItem(""))
                
    def load_sample_data(self):
        """샘플 데이터 불러오기"""
        # 샘플 데이터
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
        
        # 테이블 크기 조정
        self.row_spin.setValue(12)
        self.col_spin.setValue(9)
        self.apply_table_size()
        
        # 헤더 입력
        for j, col in enumerate(columns):
            self.data_table.setItem(0, j, QTableWidgetItem(col))
            
        # 데이터 입력
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                self.data_table.setItem(i + 1, j, QTableWidgetItem(str(val)))
                
    def load_csv(self):
        """CSV 파일 불러오기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 선택", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            # 첫 열을 인덱스로 사용할지 물어보기
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

                # 테이블 크기 조정
                self.row_spin.setValue(len(df))
                self.col_spin.setValue(len(df.columns))
                self.apply_table_size()

                # 헤더 입력
                self.data_table.setItem(0, 0, QTableWidgetItem(""))
                for j, col in enumerate(df.columns):
                    self.data_table.setItem(0, j + 1, QTableWidgetItem(str(col)))

                # 데이터 입력
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

        # 열 이름 (첫 번째 행)
        col_names = []
        for j in range(1, cols):
            item = self.data_table.item(0, j)
            col_names.append(item.text().strip() if item and item.text().strip() else f"Col{j}")

        # 행 이름과 데이터
        row_names = []
        data = []
        invalid_cells = []  # 유효하지 않은 셀 추적

        for i in range(1, rows):
            item = self.data_table.item(i, 0)
            row_names.append(item.text().strip() if item and item.text().strip() else f"Row{i}")

            row_data = []
            for j in range(1, cols):
                item = self.data_table.item(i, j)
                cell_text = item.text().strip() if item else ""

                if not cell_text:
                    val = np.nan  # 빈 셀은 NaN으로 처리
                else:
                    try:
                        val = float(cell_text)
                    except ValueError:
                        val = np.nan
                        invalid_cells.append(f"({i}, {j}): '{cell_text}'")
                row_data.append(val)
            data.append(row_data)

        df = pd.DataFrame(data, index=row_names, columns=col_names)

        # 유효하지 않은 셀이 있으면 경고
        if invalid_cells:
            warning_msg = f"다음 셀의 값이 숫자가 아니어서 NaN으로 처리됩니다:\n"
            warning_msg += "\n".join(invalid_cells[:10])  # 최대 10개만 표시
            if len(invalid_cells) > 10:
                warning_msg += f"\n... 외 {len(invalid_cells) - 10}개"
            QMessageBox.warning(self, "데이터 경고", warning_msg)

        return df

    def _get_font_weight(self, is_bold):
        """볼드 체크박스 값에 따른 폰트 weight 반환"""
        return 'bold' if is_bold else 'normal'

    def _get_font_style(self, is_italic):
        """이탤릭 체크박스 값에 따른 폰트 style 반환"""
        return 'italic' if is_italic else 'normal'

    def _get_italic_texts(self):
        """이탤릭 처리할 텍스트 목록 반환"""
        text = self.italic_texts_edit.text().strip()
        if not text:
            return []
        return [t.strip() for t in text.split(',') if t.strip()]

    def _apply_italic_to_label(self, label, italic_texts):
        """라벨 텍스트에서 특정 문자열을 이탤릭으로 변환 (matplotlib mathtext 사용)"""
        result = label
        for italic_text in italic_texts:
            if italic_text in result:
                # mathtext 형식으로 이탤릭 적용
                result = result.replace(italic_text, f'$\\it{{{italic_text}}}$')
        return result

    def _create_heatmap_figure(self, df):
        """히트맵 Figure 생성 (공통 로직)"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # 이탤릭 처리할 텍스트 목록
        italic_texts = self._get_italic_texts()

        # 라벨에 이탤릭 적용
        if italic_texts:
            new_columns = [self._apply_italic_to_label(col, italic_texts) for col in df.columns]
            new_index = [self._apply_italic_to_label(idx, italic_texts) for idx in df.index]
            df = df.copy()
            df.columns = new_columns
            df.index = new_index

        # 셀 값 스타일 설정
        annot_weight = self._get_font_weight(self.annot_bold_check.isChecked())
        annot_style = self._get_font_style(self.annot_italic_check.isChecked())

        # 히트맵 그리기
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

        # 축 라벨 스타일 설정
        axis_weight = self._get_font_weight(self.axis_bold_check.isChecked())
        axis_style = self._get_font_style(self.axis_italic_check.isChecked())
        axis_size = self.axis_size_spin.value()

        plt.xticks(rotation=45, ha='right', fontsize=axis_size,
                   fontweight=axis_weight, fontstyle=axis_style)
        plt.yticks(rotation=0, fontsize=axis_size,
                   fontweight=axis_weight, fontstyle=axis_style)

        ax.set_xlabel('', fontsize=axis_size)
        ax.set_ylabel('', fontsize=axis_size)

        # 컬러바 라벨 크기 설정
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(self.cbar_label.text(), fontsize=self.cbar_size_spin.value())

        # 제목 설정
        title_text = self.title_edit.text().strip()
        if title_text:
            title_weight = self._get_font_weight(self.title_bold_check.isChecked())
            title_style = self._get_font_style(self.title_italic_check.isChecked())
            ax.set_title(
                title_text,
                fontsize=self.title_size_spin.value(),
                fontweight=title_weight,
                fontstyle=title_style,
                pad=20
            )

        plt.tight_layout()
        return fig

    def generate_heatmap(self):
        """히트맵 생성 및 미리보기"""
        try:
            df = self.get_table_data()

            if df.empty or df.isna().all().all():
                QMessageBox.warning(self, "경고", "유효한 데이터를 입력해주세요.")
                return

            # Figure 생성
            fig = self._create_heatmap_figure(df)

            # 이미지를 바이트로 저장 (with 문으로 리소스 관리)
            with BytesIO() as buf:
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                buf.seek(0)

                # QPixmap으로 변환하여 표시
                pixmap = QPixmap()
                pixmap.loadFromData(buf.getvalue())

            # 미리보기 라벨 크기에 맞게 조정
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)

            plt.close(fig)

        except Exception as e:
            QMessageBox.critical(self, "오류", f"히트맵 생성 중 오류 발생:\n{str(e)}")
            
    def save_heatmap(self):
        """히트맵 이미지 저장"""
        df = self.get_table_data()

        if df.empty or df.isna().all().all():
            QMessageBox.warning(self, "경고", "먼저 유효한 데이터를 입력하고 히트맵을 생성해주세요.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "이미지 저장", "heatmap.png", "PNG Files (*.png);;All Files (*)"
        )

        if file_path:
            fig = None
            try:
                # 고해상도로 저장
                fig = self._create_heatmap_figure(df)
                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                QMessageBox.information(self, "성공", f"이미지가 저장되었습니다:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 중 오류 발생:\n{str(e)}")
            finally:
                if fig is not None:
                    plt.close(fig)


def main():
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    window = HeatmapGenerator()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
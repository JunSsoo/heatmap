import sys
import os
from io import BytesIO

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QComboBox,
    QLineEdit, QFormLayout, QHeaderView, QCheckBox, QTabWidget, QPlainTextEdit,
    QScrollArea, QInputDialog, QColorDialog,
    QListWidget, QListWidgetItem, QStackedWidget, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QColor

import presets
import charts
import stats_utils as stx
import data_model as dm
import curve_models as cm
import project as prj


# 한글을 모두 표현할 수 있는 폰트 후보 (라틴+한글 둘 다 렌더링 가능)
KOREAN_CAPABLE_FONTS = ('Malgun Gothic', 'Nanum Gothic', 'NanumGothic',
                        'AppleGothic', 'Noto Sans CJK KR')


def _is_korean_capable(family):
    return family in KOREAN_CAPABLE_FONTS


def set_global_font(family):
    """matplotlib 전역 폰트 설정.

    주의: matplotlib 은 font.family 한 개 폰트만 사용하고 글자별 폴백이 없으므로,
    사용자가 Arial/Helvetica 같은 **한글 없는 폰트**를 선택하고 한글을 넣으면
    글리프가 깨짐. 그 경우 자동으로 한글 폰트로 대체하는 스위치를 둠.
    """
    plt.rcParams['font.family'] = family
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'


def choose_render_font(user_family, text_samples):
    """실제 차트 렌더 직전에 호출: 텍스트에 한글이 있으면 한글 폰트로 전환.

    user_family: 사용자가 고른 폰트
    text_samples: 렌더링 대상 텍스트 리스트 (제목·축라벨·그룹명 등)
    """
    has_korean = any(
        any('가' <= c <= '힯' or 'ㄱ' <= c <= 'ㅣ' for c in (s or ''))
        for s in text_samples)
    if has_korean and not _is_korean_capable(user_family):
        return 'Malgun Gothic'
    return user_family


# 기본값은 한글·라틴 모두 렌더 가능한 Malgun Gothic
# (사용자가 영어 논문용 Arial/Helvetica 선택 가능)
set_global_font('Malgun Gothic')


# 데이터 테이블 유형 → UI 표시 이름
TABLE_TYPE_LABELS = {
    dm.DataTableType.COLUMN: '① Column (열 = 그룹)',
    dm.DataTableType.XY: '② XY (X → Y 시리즈)',
    dm.DataTableType.GROUPED: '③ Grouped (요인1 × 요인2)',
    dm.DataTableType.DOSE_RESPONSE: '④ Dose-Response (용량-반응)',
    dm.DataTableType.SURVIVAL: '⑤ Survival (생존분석)',
}
LABEL_TO_TABLE_TYPE = {v: k for k, v in TABLE_TYPE_LABELS.items()}

# 각 테이블 유형에 대응되는 신규 차트 탭 이름
CHART_TAB_NAMES = {
    'heatmap': '히트맵',
    'boxplot': '박스플롯',
    'bar': '막대그래프',
    'violin': '바이올린',
    'scatter': '산점도',
    'grouped_bar': '그룹 막대',
    'interaction': '상호작용',
    'dose_response': '용량-반응',
    'xy_error': 'XY+에러바',
}

# 차트 탭에서 허용되는 테이블 유형 매핑
CHART_ALLOWED_TYPES = {
    'heatmap': {dm.DataTableType.COLUMN},
    'boxplot': {dm.DataTableType.COLUMN},
    'bar': {dm.DataTableType.COLUMN},
    'violin': {dm.DataTableType.COLUMN},
    'scatter': {dm.DataTableType.COLUMN, dm.DataTableType.XY},
    'grouped_bar': {dm.DataTableType.GROUPED},
    'interaction': {dm.DataTableType.GROUPED},
    'dose_response': {dm.DataTableType.DOSE_RESPONSE},
    'xy_error': {dm.DataTableType.XY},
}


# 각 차트의 사용 가이드 (사용자가 처음 써도 알 수 있도록 각 탭 상단에 표시)
CHART_GUIDES = {
    'heatmap': {
        'when': "여러 범주(예: 지역) × 여러 범주(예: 약제)의 수치를 "
                "색 강도로 한눈에 비교할 때.",
        'format': [
            "1행: X축 범주 이름 (예: 약제명)",
            "1열: Y축 범주 이름 (예: 지역명)",
            "나머지 셀: 숫자 값 (0 ~ 100 같은 수치)",
        ],
        'example_key': 'sample_heatmap',
    },
    'boxplot': {
        'when': "여러 그룹의 분포를 비교할 때 (중앙값·사분위·이상치).",
        'format': [
            "1행: 그룹 이름 (각 열이 하나의 그룹)",
            "1열: 행 번호 또는 반복 ID",
            "나머지 셀: 각 그룹의 반복 측정값",
        ],
        'example_key': 'column_groups',
    },
    'bar': {
        'when': "그룹별 평균 ± 오차바(SEM/SD/95%CI)를 비교할 때.",
        'format': [
            "1행: 그룹 이름",
            "1열: 행 번호",
            "나머지 셀: 각 그룹의 반복 측정값",
        ],
        'example_key': 'column_groups',
    },
    'violin': {
        'when': "분포의 형태(밀도)까지 보고 싶을 때. Boxplot보다 풍부.",
        'format': [
            "1행: 그룹 이름",
            "1열: 행 번호",
            "나머지 셀: 각 그룹의 반복 측정값",
        ],
        'example_key': 'column_groups',
    },
    'scatter': {
        'when': "두 변수의 관계와 선형회귀를 동시에 보고 싶을 때.",
        'format': [
            "1열: X 값 (숫자, 연속형)",
            "2열 이상: Y 시리즈 (여러 개 가능)",
            "각 행: 하나의 관측점",
        ],
        'example_key': 'xy_series',
    },
    'grouped_bar': {
        'when': "두 요인이 교차된 설계(예: 지역 × 약제 농도)의 효과 비교. "
                "Two-way ANOVA와 짝.",
        'format': [
            "1열: 요인1 수준 (예: 지역)",
            "2열 이상: 요인2 수준별 값 (예: 저·중·고 농도)",
            "각 행: 한 번의 관측 (반복 측정)",
        ],
        'example_key': 'grouped',
    },
    'interaction': {
        'when': "Two-way ANOVA에서 두 요인 간 상호작용을 시각적으로 해석할 때. "
                "선이 평행하면 상호작용 없음, 교차하면 상호작용 있음.",
        'format': [
            "1열: 요인1 수준",
            "2열 이상: 요인2 수준별 값",
            "(Grouped Bar와 동일한 데이터)",
        ],
        'example_key': 'grouped',
    },
    'dose_response': {
        'when': "농약 LC50/EC50 계산. 농도가 증가할 때 사망률이 어떻게 변하는지 "
                "곡선 피팅 (Hill 4PL · Probit 등).",
        'format': [
            "1열: 농도 (log-spaced, 예: 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)",
            "2열 이상: 각 반복실험의 사망률 (%)",
            "각 행: 한 농도에서의 측정 (반복은 열로 배치)",
        ],
        'example_key': 'dose_response',
    },
    'xy_error': {
        'when': "시간 경과·농도 변화 같은 XY 데이터에 에러바와 선을 함께 표시.",
        'format': [
            "1열: X 값 (시간·농도 등)",
            "2열 이상: Y 시리즈별 값 (각 행은 한 X 지점)",
        ],
        'example_key': 'xy_series',
    },
}


# 예시 데이터셋 (사용 가이드의 "예시 로드" 버튼이 로드함)
EXAMPLE_DATASETS = {
    'sample_heatmap': {
        'table_type': dm.DataTableType.COLUMN,
        'use_load_sample': True,  # 기존 샘플 데이터 사용
    },
    'column_groups': {
        'table_type': dm.DataTableType.COLUMN,
        'columns': ['', 'Acetamiprid', 'Chlorfenapyr', 'Abamectin', 'Spinetoram'],
        'rows': [
            ['Gochang', 11, 96, 89, 98],
            ['Gwangju', 11, 78, 57, 92],
            ['Namwon', 27, 85, 72, 97],
            ['Damyang', 20, 93, 63, 97],
            ['Boseong', 31, 77, 59, 81],
            ['Suncheon', 66, 82, 97, 99],
            ['Yeonggwang', 6, 75, 50, 81],
            ['Yeongam', 27, 77, 51, 76],
        ],
    },
    'xy_series': {
        'table_type': dm.DataTableType.XY,
        'columns': ['', 'Time (h)', 'Treatment A', 'Treatment B'],
        'rows': [
            ['t=0', 0, 100, 100],
            ['t=1', 1, 85, 92],
            ['t=2', 2, 70, 80],
            ['t=4', 4, 45, 60],
            ['t=8', 8, 20, 35],
            ['t=16', 16, 5, 15],
            ['t=24', 24, 2, 7],
        ],
    },
    'grouped': {
        'table_type': dm.DataTableType.GROUPED,
        'columns': ['', 'Region', 'Low', 'Mid', 'High'],
        'rows': [
            ['r1', 'Seoul', 10, 40, 85],
            ['r2', 'Seoul', 12, 45, 82],
            ['r3', 'Seoul', 11, 43, 88],
            ['r4', 'Busan', 15, 55, 90],
            ['r5', 'Busan', 18, 52, 88],
            ['r6', 'Busan', 14, 57, 91],
            ['r7', 'Daegu', 8, 38, 80],
            ['r8', 'Daegu', 11, 42, 83],
            ['r9', 'Daegu', 9, 40, 78],
        ],
    },
    'dose_response': {
        'table_type': dm.DataTableType.DOSE_RESPONSE,
        'columns': ['', 'Dose (ppm)', 'Rep 1 (%)', 'Rep 2 (%)', 'Rep 3 (%)'],
        'rows': [
            ['d0', 0.01, 2, 4, 3],
            ['d1', 0.03, 5, 6, 4],
            ['d2', 0.1, 12, 14, 10],
            ['d3', 0.3, 28, 30, 25],
            ['d4', 1, 55, 58, 52],
            ['d5', 3, 78, 80, 75],
            ['d6', 10, 92, 93, 90],
            ['d7', 30, 97, 98, 96],
            ['d8', 100, 99, 99, 99],
        ],
    },
}


# 테이블 유형별 추천 분석
ANALYSES_BY_TYPE = {
    dm.DataTableType.COLUMN: [
        'Auto (추천)', '기술통계',
        # ANOVA / 다중비교
        'One-way ANOVA + Tukey',
        'Holm-Sidak 다중비교', 'Scheffé 다중비교',
        # 비모수
        'Kruskal-Wallis + Dunn', 'Mann-Whitney (2그룹)',
        'Wilcoxon 대응표본 (2그룹)', 'Friedman 반복측정',
        # 정규성 / 등분산
        'Shapiro-Wilk 정규성', "D'Agostino-Pearson 정규성",
        'Anderson-Darling 정규성', 'Levene 등분산',
        # 효과크기 / 상관
        "Cohen's d (2그룹 효과크기)", '상관 행렬 (전체 쌍)',
    ],
    dm.DataTableType.XY: [
        '기술통계', '선형회귀', 'Pearson 상관', 'Spearman 상관',
    ],
    dm.DataTableType.GROUPED: [
        'Two-way ANOVA', 'Tukey HSD', "Dunnett's (vs. 대조)",
    ],
    dm.DataTableType.DOSE_RESPONSE: [
        'Hill 4PL 피팅', 'Hill 5PL 피팅', 'Probit 피팅', 'Logit 피팅',
        'LogLogistic 4PL 피팅', '모델 비교',
        'EC50/LC50 계산', 'LC10/LC90 외삽',
        'F-test (Hill 4PL vs 5PL)',
    ],
    dm.DataTableType.SURVIVAL: [
        '(Phase 6 예정)',
    ],
}

# 용량-반응 모델 선택지 (ComboBox에 그대로 표시)
DOSE_RESPONSE_MODELS = [
    'Hill 4PL', 'Hill 5PL', 'Probit', 'Logit', 'LogLogistic 4PL',
]

# 9개 차트 탭의 위젯 prefix (자동 직렬화 대상)
CHART_WIDGET_PREFIXES = ('hm', 'bx', 'br', 'vl', 'sc', 'gb', 'it', 'dr', 'xe')


from stats_utils import _fmt_num as _fmt_float  # 공용 포맷터


def _serialize_widget(w):
    """위젯 → JSON-safe dict. 직렬화 가능한 유형만 반환, 나머지는 None."""
    if isinstance(w, QLineEdit):
        return {'k': 'L', 'v': w.text()}
    if isinstance(w, QDoubleSpinBox):
        return {'k': 'D', 'v': float(w.value())}
    if isinstance(w, QSpinBox):
        return {'k': 'S', 'v': int(w.value())}
    if isinstance(w, QCheckBox):
        return {'k': 'C', 'v': bool(w.isChecked())}
    if isinstance(w, QComboBox):
        return {'k': 'B', 'v': w.currentText()}
    return None


def _apply_widget_value(w, d):
    """_serialize_widget 로 만든 dict를 위젯에 복원. 타입 불일치 시 무시."""
    if not isinstance(d, dict):
        return
    k, v = d.get('k'), d.get('v')
    try:
        if k == 'L' and isinstance(w, QLineEdit):
            w.setText(str(v))
        elif k == 'D' and isinstance(w, QDoubleSpinBox):
            w.setValue(float(v))
        elif k == 'S' and isinstance(w, QSpinBox):
            w.setValue(int(v))
        elif k == 'C' and isinstance(w, QCheckBox):
            w.setChecked(bool(v))
        elif k == 'B' and isinstance(w, QComboBox):
            idx = w.findText(str(v))
            if idx >= 0:
                w.setCurrentIndex(idx)
    except (ValueError, TypeError):
        pass


class GraphGenerator(QMainWindow):
    """메인 윈도우: 데이터 + 전역 설정 + 차트 탭 + 분석 패널."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Graph Generator")
        self.setGeometry(40, 40, 1920, 1080)

        # 상태
        self.current_fig = None
        self.current_graph_type = None
        self.current_table_type = dm.DataTableType.COLUMN
        self.current_fit_result = None      # 용량-반응 피팅 결과
        self.last_analysis = None           # 마지막 AnalysisResult (프로젝트/PDF용)
        self.collected_analyses = []        # 세션에서 실행한 분석들(PDF 리포트용)
        self.collected_fits = []            # 세션에서 실행한 피팅들
        self.dose_response_color = '#c0392b'  # 사용자 지정 단색
        self.dirty = prj.DirtyTracker(on_change=self._on_dirty_changed)

        # 차트 탭 인덱스 매핑 (탭 비활성화 용도)
        self._tab_index_by_key = {}

        # 모든 하위 위젯을 먼저 생성 (Qt 위젯은 순서와 무관하게 생성 가능)
        self.tab_widget = QTabWidget()
        self._setup_heatmap_tab()
        self._setup_boxplot_tab()
        self._setup_bar_tab()
        self._setup_violin_tab()
        self._setup_scatter_tab()
        self._setup_grouped_bar_tab()
        self._setup_interaction_tab()
        self._setup_dose_response_tab()
        self._setup_xy_error_tab()

        # 각 단계 페이지 위젯 구성
        step_data = self._build_step_widget(
            '①  데이터',
            '데이터 테이블 유형을 선택하고 값을 입력하세요. CSV·Excel도 불러올 수 있습니다.',
            self._setup_data_table,
        )
        step_chart = self._build_step_chart()
        step_analysis = self._build_step_widget(
            '③  분석',
            '현재 데이터에 적합한 통계 분석을 선택해 실행합니다.',
            self._setup_analysis_panel,
        )
        step_export = self._build_step_widget(
            '④  내보내기',
            '논문 형식에 맞는 크기·포맷을 선택해 저장하거나, 프로젝트·PDF 리포트로 보관하세요.',
            self._setup_global_settings,
        )

        # 미리보기 패널 (우측 상시)
        preview_panel = self._build_preview_panel()

        # 사이드바 + 스테이지 + 미리보기
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(10)

        self.sidebar_list = self._build_sidebar()
        body.addWidget(self.sidebar_list)

        self.stack = QStackedWidget()
        self.stack.addWidget(step_data)       # 0
        self.stack.addWidget(step_chart)      # 1
        self.stack.addWidget(step_analysis)   # 2
        self.stack.addWidget(step_export)     # 3
        self.sidebar_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        body.addWidget(self.stack, 3)

        body.addWidget(preview_panel, 3)

        body_container = QWidget(); body_container.setLayout(body)
        root.addWidget(body_container, 1)

        # 초기화
        self.sidebar_list.setCurrentRow(0)
        self.apply_table_size()
        self._refresh_tabs_enabled()
        self._refresh_analysis_choices()

        # 데이터 편집 → dirty 표시
        self.data_table.itemChanged.connect(lambda _i: self.dirty.mark_dirty())

    # =====================================================================
    # 단계형 UI 빌더
    # =====================================================================
    def _build_header(self):
        """상단 얇은 헤더 — 제목과 진행 상태만 표시."""
        w = QFrame()
        w.setFixedHeight(44)
        w.setStyleSheet(
            "QFrame { border-bottom: 1px solid #d0d0d0; background: transparent; }"
        )
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 6, 12, 6)

        title = QLabel("Scientific Graph Generator")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #222;")
        lay.addWidget(title)

        lay.addStretch()

        self.header_status = QLabel("")
        self.header_status.setStyleSheet("color: #666; font-size: 10pt;")
        lay.addWidget(self.header_status)
        return w

    def _build_sidebar(self):
        """단계 네비게이션 — 4개 큰 세로 버튼."""
        lst = QListWidget()
        lst.setFixedWidth(160)
        lst.setSpacing(2)
        lst.setStyleSheet("""
            QListWidget {
                background: #fafafa;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 6px;
                outline: 0;
            }
            QListWidget::item {
                padding: 18px 12px;
                font-size: 13pt;
                border-radius: 4px;
                color: #333;
            }
            QListWidget::item:hover {
                background: #e8f0fe;
            }
            QListWidget::item:selected {
                background: #2196F3;
                color: white;
                font-weight: bold;
            }
        """)
        for label in ['①  데이터', '②  차트', '③  분석', '④  내보내기']:
            lst.addItem(QListWidgetItem(label))
        return lst

    def _build_step_widget(self, title_text, hint_text, setup_fn):
        """공통 단계 페이지: 큰 제목 + 힌트 + setup_fn(layout)으로 본문 구성."""
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QLabel(title_text)
        title.setStyleSheet(
            "font-size: 20pt; font-weight: bold; color: #222; padding: 2px 0 2px 0;")
        outer.addWidget(title)

        if hint_text:
            hint = QLabel(hint_text)
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #666; font-size: 10pt; padding-bottom: 8px;")
            outer.addWidget(hint)

        # 본문을 스크롤 가능하게
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        setup_fn(body_lay)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll, 1)
        return w

    def _build_step_chart(self):
        """Step 2: 차트 탭을 스크롤 내부에 배치."""
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QLabel('②  차트')
        title.setStyleSheet(
            "font-size: 20pt; font-weight: bold; color: #222; padding: 2px 0 2px 0;")
        outer.addWidget(title)

        hint = QLabel('차트 종류를 탭에서 고르고 "그래프 생성" 버튼을 누르세요. '
                      '데이터 유형과 호환되는 차트만 활성화됩니다.')
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 10pt; padding-bottom: 8px;")
        outer.addWidget(hint)

        outer.addWidget(self.tab_widget, 1)
        return w

    def _build_preview_panel(self):
        """우측 상시 미리보기 패널."""
        group = QGroupBox("미리보기")
        lay = QVBoxLayout(group)
        self.preview_label = QLabel("그래프를 생성하면 여기에 표시됩니다")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(520, 460)
        self.preview_label.setStyleSheet(
            "background-color: #f7f7f7; border: 1px solid #ccc; border-radius: 4px;")
        lay.addWidget(self.preview_label, 1)
        return group

    # =====================================================================
    # 상단 전역 설정
    # =====================================================================
    def _setup_global_settings(self, parent):
        box = QGroupBox("출력 설정 (논문용)")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("저널 프리셋:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(presets.JOURNAL_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        grid.addWidget(self.preset_combo, 0, 1)

        grid.addWidget(QLabel("폭(mm):"), 0, 2)
        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(10, 1000); self.width_mm.setDecimals(1)
        self.width_mm.setValue(150)
        grid.addWidget(self.width_mm, 0, 3)

        grid.addWidget(QLabel("높이(mm):"), 0, 4)
        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(10, 1000); self.height_mm.setDecimals(1)
        self.height_mm.setValue(110)
        grid.addWidget(self.height_mm, 0, 5)

        grid.addWidget(QLabel("DPI:"), 0, 6)
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems([str(d) for d in presets.DPI_OPTIONS])
        self.dpi_combo.setCurrentText('300')
        grid.addWidget(self.dpi_combo, 0, 7)

        grid.addWidget(QLabel("폰트:"), 1, 0)
        self.font_combo = QComboBox()
        self.font_combo.addItems(presets.FONT_FAMILIES)
        # 기본값은 한글+라틴 모두 가능한 Malgun Gothic
        self.font_combo.setCurrentText('Malgun Gothic')
        self.font_combo.currentTextChanged.connect(lambda f: set_global_font(f))
        grid.addWidget(self.font_combo, 1, 1)

        grid.addWidget(QLabel("저장 포맷:"), 1, 2)
        self.fmt_combo = QComboBox()
        for label, _ in presets.EXPORT_FORMATS:
            self.fmt_combo.addItem(label)
        grid.addWidget(self.fmt_combo, 1, 3)

        grid.addWidget(QLabel("투명 배경:"), 1, 4)
        self.transparent_check = QCheckBox()
        grid.addWidget(self.transparent_check, 1, 5)

        self.save_btn = QPushButton("이미지 저장")
        self.save_btn.clicked.connect(self.save_current_graph)
        self.save_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        grid.addWidget(self.save_btn, 1, 6, 1, 2)

        # 프로젝트/리포트 버튼
        self.proj_save_btn = QPushButton("프로젝트 저장 (.gpj)")
        self.proj_save_btn.clicked.connect(self.save_project)
        grid.addWidget(self.proj_save_btn, 2, 0, 1, 2)

        self.proj_load_btn = QPushButton("프로젝트 불러오기 (.gpj)")
        self.proj_load_btn.clicked.connect(self.load_project)
        grid.addWidget(self.proj_load_btn, 2, 2, 1, 2)

        self.pdf_btn = QPushButton("PDF 리포트 생성")
        self.pdf_btn.clicked.connect(self.generate_pdf_report)
        self.pdf_btn.setStyleSheet(
            "background-color: #8e44ad; color: white; font-weight: bold; padding: 6px;")
        grid.addWidget(self.pdf_btn, 2, 4, 1, 2)

        grid.addWidget(QLabel("테마:"), 2, 6)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(QT_MATERIAL_THEMES)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        grid.addWidget(self.theme_combo, 2, 7)

        parent.addWidget(box)

    def _on_theme_changed(self, theme_name):
        app = getattr(self, '_app_ref', None) or QApplication.instance()
        if app is None:
            return
        apply_theme(app, theme_name)

    def _on_preset_changed(self, name):
        preset = presets.apply_journal_preset(name)
        if preset is None:
            return
        self.width_mm.setValue(preset['width_mm'])
        self.height_mm.setValue(preset['height_mm'])
        self.dpi_combo.setCurrentText(str(preset['dpi']))

    def _figure_size_inches(self):
        return (presets.mm_to_inch(self.width_mm.value()),
                presets.mm_to_inch(self.height_mm.value()))

    # =====================================================================
    # 데이터 패널
    # =====================================================================
    def _setup_data_table(self, parent_layout):
        # --- 테이블 유형 선택 ---
        type_group = QGroupBox("데이터 테이블 유형")
        type_layout = QVBoxLayout(type_group)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("유형:"))
        self.table_type_combo = QComboBox()
        for t, lbl in TABLE_TYPE_LABELS.items():
            self.table_type_combo.addItem(lbl)
        self.table_type_combo.currentTextChanged.connect(self._on_table_type_changed)
        top_row.addWidget(self.table_type_combo, 1)

        self.detect_btn = QPushButton("자동 감지")
        self.detect_btn.clicked.connect(self.auto_detect_table_type)
        top_row.addWidget(self.detect_btn)
        type_layout.addLayout(top_row)

        self.role_hint_label = QLabel("")
        self.role_hint_label.setWordWrap(True)
        self.role_hint_label.setStyleSheet(
            "color: #555; font-size: 11px; padding: 4px; background: #eef; border-radius: 3px;")
        type_layout.addWidget(self.role_hint_label)

        parent_layout.addWidget(type_group)

        # --- 테이블 크기/IO ---
        size_group = QGroupBox("테이블 크기 & 입출력")
        size_layout = QHBoxLayout(size_group)

        size_layout.addWidget(QLabel("행 수:"))
        self.row_spin = QSpinBox(); self.row_spin.setRange(1, 2000); self.row_spin.setValue(12)
        size_layout.addWidget(self.row_spin)

        size_layout.addWidget(QLabel("열 수:"))
        self.col_spin = QSpinBox(); self.col_spin.setRange(1, 2000); self.col_spin.setValue(9)
        size_layout.addWidget(self.col_spin)

        self.apply_size_btn = QPushButton("적용")
        self.apply_size_btn.clicked.connect(self.apply_table_size)
        size_layout.addWidget(self.apply_size_btn)

        self.load_csv_btn = QPushButton("CSV")
        self.load_csv_btn.clicked.connect(self.load_csv)
        size_layout.addWidget(self.load_csv_btn)

        self.load_excel_btn = QPushButton("Excel")
        self.load_excel_btn.clicked.connect(self.load_excel)
        size_layout.addWidget(self.load_excel_btn)

        self.load_sample_btn = QPushButton("샘플")
        self.load_sample_btn.clicked.connect(self.load_sample_data)
        size_layout.addWidget(self.load_sample_btn)

        self.clear_btn = QPushButton("초기화")
        self.clear_btn.clicked.connect(self.clear_table)
        size_layout.addWidget(self.clear_btn)

        parent_layout.addWidget(size_group)

        table_group = QGroupBox("데이터 (첫 행=열 이름, 첫 열=행 이름)")
        table_layout = QVBoxLayout(table_group)
        self.data_table = QTableWidget()
        self.data_table.setRowCount(13)
        self.data_table.setColumnCount(10)
        self.data_table.setMinimumHeight(260)
        table_layout.addWidget(self.data_table)
        parent_layout.addWidget(table_group)

        # 초기 힌트
        self._update_role_hint()

    # -----------------------------
    # 테이블 유형 관련
    # -----------------------------
    def _on_table_type_changed(self, label):
        t = LABEL_TO_TABLE_TYPE.get(label)
        if t is None:
            return
        self.current_table_type = t
        self._update_role_hint()
        self._refresh_tabs_enabled()
        self._refresh_analysis_choices()
        self.dirty.mark_dirty()

    def _update_role_hint(self):
        spec = dm.TABLE_SPECS.get(self.current_table_type)
        if spec is None:
            self.role_hint_label.setText("")
            return
        # 열 역할 요약
        roles = spec.column_roles
        parts = []
        for idx, role in roles.items():
            if idx == -1:
                parts.append(f"[3열~] {role}")
            else:
                parts.append(f"[{idx+1}열] {role}")
        self.role_hint_label.setText(
            f"{spec.description}\n열 역할: " + ", ".join(parts))

    def auto_detect_table_type(self):
        """현재 테이블 내용에서 유형을 자동 감지 → 콤보 업데이트."""
        try:
            df_wide = self.get_table_data(silent=True)
        except Exception:
            QMessageBox.warning(self, "감지 실패", "테이블 데이터를 읽을 수 없습니다.")
            return
        raw_df = self._df_for_parse(df_wide)
        t = dm.detect_table_type(raw_df)
        label = TABLE_TYPE_LABELS.get(t)
        if label:
            self.table_type_combo.setCurrentText(label)
            QMessageBox.information(self, "자동 감지", f"감지된 유형: {label}")

    def _refresh_tabs_enabled(self):
        """현재 테이블 유형에 따라 차트 탭을 enable/disable."""
        for key, idx in self._tab_index_by_key.items():
            allowed = CHART_ALLOWED_TYPES.get(key, set())
            enabled = self.current_table_type in allowed
            self.tab_widget.setTabEnabled(idx, enabled)

    def _refresh_analysis_choices(self):
        items = ANALYSES_BY_TYPE.get(self.current_table_type, ['기술통계'])
        self.analysis_combo.blockSignals(True)
        self.analysis_combo.clear()
        self.analysis_combo.addItems(items)
        self.analysis_combo.blockSignals(False)

    # -----------------------------
    # 테이블 크기/IO
    # -----------------------------
    def apply_table_size(self):
        rows = self.row_spin.value() + 1
        cols = self.col_spin.value() + 1
        self.data_table.setRowCount(rows)
        self.data_table.setColumnCount(cols)
        if self.data_table.item(0, 0) is None:
            self.data_table.setItem(0, 0, QTableWidgetItem(""))
        self.data_table.horizontalHeader().setVisible(False)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        for i in range(cols):
            self.data_table.setColumnWidth(i, 95)
        for i in range(rows):
            self.data_table.setRowHeight(i, 26)

    def clear_table(self):
        for i in range(self.data_table.rowCount()):
            for j in range(self.data_table.columnCount()):
                self.data_table.setItem(i, j, QTableWidgetItem(""))
        self.dirty.mark_dirty()

    def load_sample_data(self):
        columns = ['', 'Acrinathrin', 'Acetamiprid', 'Dinotefuran', 'Cyclaniliprole',
                   'Chlorfluazuron', 'Chlorfenapyr', 'Abamectin', 'Spinetoram',
                   'Emamectin benzoate']
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
        self.dirty.mark_clean()

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 선택", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        reply = QMessageBox.question(
            self, "옵션", "CSV 첫 번째 열을 행 이름으로 사용할까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        use_index = (reply == QMessageBox.Yes)
        try:
            df = pd.read_csv(file_path, index_col=0) if use_index else pd.read_csv(file_path)
            if not use_index:
                df.index = [f"Row{i+1}" for i in range(len(df))]
            self._load_dataframe(df)
            QMessageBox.information(self, "성공", "불러왔습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 로딩 실패:\n{e}")

    def load_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Excel 파일 선택", "", "Excel Files (*.xlsx *.xls);;All Files (*)")
        if not file_path:
            return
        try:
            df = pd.read_excel(file_path, index_col=0)
            self._load_dataframe(df)
            QMessageBox.information(self, "성공", "불러왔습니다.")
        except ImportError:
            QMessageBox.critical(self, "오류",
                "Excel을 읽으려면 'openpyxl'이 필요합니다.\n\npip install openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 로딩 실패:\n{e}")

    def _load_dataframe(self, df):
        n_rows, n_cols = len(df), len(df.columns)
        if n_rows > self.row_spin.maximum():
            self.row_spin.setMaximum(n_rows)
        if n_cols > self.col_spin.maximum():
            self.col_spin.setMaximum(n_cols)
        self.row_spin.setValue(n_rows)
        self.col_spin.setValue(n_cols)
        self.apply_table_size()
        self.data_table.setItem(0, 0, QTableWidgetItem(""))
        for j, col in enumerate(df.columns):
            self.data_table.setItem(0, j + 1, QTableWidgetItem(str(col)))
        for i in range(n_rows):
            self.data_table.setItem(i + 1, 0, QTableWidgetItem(str(df.index[i])))
            for j in range(n_cols):
                self.data_table.setItem(i + 1, j + 1,
                                        QTableWidgetItem(str(df.iloc[i, j])))
        self.dirty.mark_clean()

    def get_table_data(self, silent=False):
        """기존 컨벤션: index=행 이름, columns=열 이름(문자열), 값=float(NaN 가능)."""
        rows = self.data_table.rowCount()
        cols = self.data_table.columnCount()
        col_names = []
        for j in range(1, cols):
            it = self.data_table.item(0, j)
            col_names.append(it.text().strip() if it and it.text().strip() else f"Col{j}")
        row_names = []
        data = []
        invalid = []
        for i in range(1, rows):
            it = self.data_table.item(i, 0)
            row_names.append(it.text().strip() if it and it.text().strip() else f"Row{i}")
            row_data = []
            for j in range(1, cols):
                it = self.data_table.item(i, j)
                txt = it.text().strip() if it else ""
                if not txt:
                    row_data.append(np.nan)
                else:
                    try:
                        row_data.append(float(txt))
                    except ValueError:
                        row_data.append(np.nan)
                        invalid.append(f"({i},{j}):'{txt}'")
            data.append(row_data)
        df = pd.DataFrame(data, index=row_names, columns=col_names)
        if invalid and not silent:
            msg = "다음 셀은 숫자가 아니어서 NaN으로 처리됩니다:\n" + "\n".join(invalid[:10])
            if len(invalid) > 10:
                msg += f"\n... 외 {len(invalid) - 10}개"
            QMessageBox.warning(self, "데이터 경고", msg)
        return df

    def get_table_data_raw(self):
        """Grouped 등 비수치 열을 허용하는 원본(문자열) 읽기."""
        rows = self.data_table.rowCount()
        cols = self.data_table.columnCount()
        col_names = []
        for j in range(1, cols):
            it = self.data_table.item(0, j)
            col_names.append(it.text().strip() if it and it.text().strip() else f"Col{j}")
        row_names = []
        data = []
        for i in range(1, rows):
            it = self.data_table.item(i, 0)
            row_names.append(it.text().strip() if it and it.text().strip() else f"Row{i}")
            row_data = []
            for j in range(1, cols):
                it = self.data_table.item(i, j)
                txt = it.text().strip() if it else ""
                row_data.append(txt)
            data.append(row_data)
        df = pd.DataFrame(data, index=row_names, columns=col_names)
        return df

    def _df_for_parse(self, df_wide_or_raw):
        """data_model.parse_table 이 기대하는 '행 이름이 첫 열에 있는 DataFrame' 구성."""
        df = df_wide_or_raw.reset_index()
        df.columns = [str(c) for c in df.columns]
        return df

    # =====================================================================
    # 공통 필드 팩토리
    # =====================================================================
    def _font_row(self, layout, name_attr_prefix, title_bold_default=True,
                  axis_bold_default=True, tick_bold_default=True,
                  title_size=14, axis_size=11, tick_size=10,
                  show_annot=False, annot_size=11, annot_bold_default=True):
        """제목/축/눈금 스타일 필드 묶음 생성."""
        grp = QGroupBox("글꼴 설정")
        form = QFormLayout(grp)

        setattr(self, f'{name_attr_prefix}_title_edit', QLineEdit(""))
        getattr(self, f'{name_attr_prefix}_title_edit').setPlaceholderText("제목 (선택)")
        form.addRow("제목:", getattr(self, f'{name_attr_prefix}_title_edit'))

        setattr(self, f'{name_attr_prefix}_subtitle_edit', QLineEdit(""))
        getattr(self, f'{name_attr_prefix}_subtitle_edit').setPlaceholderText("부제 (선택)")
        form.addRow("부제:", getattr(self, f'{name_attr_prefix}_subtitle_edit'))

        s = QSpinBox(); s.setRange(6, 40); s.setValue(title_size)
        setattr(self, f'{name_attr_prefix}_title_size', s)
        form.addRow("제목 크기:", s)

        row = QHBoxLayout()
        b = QCheckBox("볼드"); b.setChecked(title_bold_default)
        i = QCheckBox("이탤릭")
        setattr(self, f'{name_attr_prefix}_title_bold', b)
        setattr(self, f'{name_attr_prefix}_title_italic', i)
        row.addWidget(b); row.addWidget(i); row.addStretch()
        form.addRow("제목 스타일:", row)

        s = QSpinBox(); s.setRange(6, 30); s.setValue(axis_size)
        setattr(self, f'{name_attr_prefix}_axis_size', s)
        form.addRow("축 라벨 크기:", s)

        row = QHBoxLayout()
        b = QCheckBox("볼드"); b.setChecked(axis_bold_default)
        i = QCheckBox("이탤릭")
        setattr(self, f'{name_attr_prefix}_axis_bold', b)
        setattr(self, f'{name_attr_prefix}_axis_italic', i)
        row.addWidget(b); row.addWidget(i); row.addStretch()
        form.addRow("축 라벨 스타일:", row)

        s = QSpinBox(); s.setRange(6, 24); s.setValue(tick_size)
        setattr(self, f'{name_attr_prefix}_tick_size', s)
        form.addRow("눈금 크기:", s)

        row = QHBoxLayout()
        b = QCheckBox("볼드"); b.setChecked(tick_bold_default)
        i = QCheckBox("이탤릭")
        setattr(self, f'{name_attr_prefix}_tick_bold', b)
        setattr(self, f'{name_attr_prefix}_tick_italic', i)
        row.addWidget(b); row.addWidget(i); row.addStretch()
        form.addRow("눈금 스타일:", row)

        if show_annot:
            s = QSpinBox(); s.setRange(6, 24); s.setValue(annot_size)
            setattr(self, f'{name_attr_prefix}_annot_size', s)
            form.addRow("셀 값 크기:", s)
            row = QHBoxLayout()
            b = QCheckBox("볼드"); b.setChecked(annot_bold_default)
            i = QCheckBox("이탤릭")
            setattr(self, f'{name_attr_prefix}_annot_bold', b)
            setattr(self, f'{name_attr_prefix}_annot_italic', i)
            row.addWidget(b); row.addWidget(i); row.addStretch()
            form.addRow("셀 값 스타일:", row)

        e = QLineEdit("")
        e.setPlaceholderText("예: Spodoptera frugiperda (쉼표로 구분)")
        setattr(self, f'{name_attr_prefix}_italic_texts', e)
        form.addRow("이탤릭 텍스트:", e)

        layout.addWidget(grp)

    def _axis_range_row(self, form, name_prefix, include_x=False,
                        y_default=(0, 100), x_default=(0, 100)):
        def dsb(v):
            s = QDoubleSpinBox(); s.setRange(-1e9, 1e9); s.setDecimals(2); s.setValue(v); return s
        if include_x:
            xmin = dsb(x_default[0]); xmax = dsb(x_default[1])
            setattr(self, f'{name_prefix}_xmin', xmin)
            setattr(self, f'{name_prefix}_xmax', xmax)
            row = QHBoxLayout(); row.addWidget(xmin); row.addWidget(QLabel("~")); row.addWidget(xmax)
            w = QWidget(); w.setLayout(row)
            form.addRow("X축 범위:", w)
        ymin = dsb(y_default[0]); ymax = dsb(y_default[1])
        setattr(self, f'{name_prefix}_ymin', ymin)
        setattr(self, f'{name_prefix}_ymax', ymax)
        row = QHBoxLayout(); row.addWidget(ymin); row.addWidget(QLabel("~")); row.addWidget(ymax)
        w = QWidget(); w.setLayout(row)
        form.addRow("Y축 범위:", w)

    def _stats_row(self, layout, prefix):
        grp = QGroupBox("통계 (유의성 브래킷)")
        form = QFormLayout(grp)

        cb = QCheckBox("브래킷 표시")
        setattr(self, f'{prefix}_show_brackets', cb)
        form.addRow("", cb)

        c = QComboBox(); c.addItems(['t-test', 'welch', 'mannwhitney'])
        setattr(self, f'{prefix}_test', c)
        form.addRow("검정:", c)

        c = QComboBox(); c.addItems(['bonferroni', 'none'])
        setattr(self, f'{prefix}_correction', c)
        form.addRow("보정:", c)

        layout.addWidget(grp)

    def _wrap_scroll(self, content_widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        return scroll

    def _tab_action_row(self, layout, generate_callback):
        row = QHBoxLayout()
        btn = QPushButton("그래프 생성 / 미리보기")
        btn.clicked.connect(generate_callback)
        btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        row.addWidget(btn)
        layout.addLayout(row)
        layout.addStretch()

    def _register_tab(self, content_widget, key, label):
        """차트 탭을 등록하고 유형-감지 활성화에 연결."""
        idx = self.tab_widget.addTab(self._wrap_scroll(content_widget), label)
        self._tab_index_by_key[key] = idx

    # =====================================================================
    # 각 탭 상단 사용 가이드 박스 (사용법 + 예시 로드)
    # =====================================================================
    def _add_help_box(self, layout, chart_key):
        """차트 탭 상단에 '언제 쓰나요 + 데이터 형식 + 예시 로드' 박스를 추가."""
        guide = CHART_GUIDES.get(chart_key)
        if not guide:
            return

        box = QGroupBox("사용 가이드")
        box.setStyleSheet(
            "QGroupBox { background: #f0f7ff; border: 1px solid #b4d2ff; "
            "border-radius: 6px; padding: 10px 8px 8px 8px; margin-top: 10px; "
            "font-weight: bold; color: #1a73e8; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 4px; }")
        inner = QVBoxLayout(box)
        inner.setSpacing(4)

        # 언제 쓰나요
        when_lbl = QLabel(f"<b>언제 쓰나요?</b>  {guide['when']}")
        when_lbl.setWordWrap(True)
        when_lbl.setStyleSheet(
            "color: #222; font-weight: normal; padding: 2px; background: transparent;")
        inner.addWidget(when_lbl)

        # 데이터 형식
        fmt_title = QLabel("<b>데이터 형식</b>")
        fmt_title.setStyleSheet(
            "color: #333; padding: 4px 2px 0 2px; font-weight: normal; "
            "background: transparent;")
        inner.addWidget(fmt_title)

        for line in guide['format']:
            bullet = QLabel(f"•  {line}")
            bullet.setWordWrap(True)
            bullet.setStyleSheet(
                "color: #444; padding: 1px 2px 1px 14px; "
                "font-weight: normal; background: transparent;")
            inner.addWidget(bullet)

        # 예시 로드 버튼
        btn = QPushButton("▶  이 차트용 예시 데이터 불러오기")
        btn.setStyleSheet(
            "background-color: #1a73e8; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 4px; margin-top: 4px;")
        example_key = guide['example_key']
        btn.clicked.connect(lambda _=False, k=example_key: self._load_example_data(k))
        inner.addWidget(btn)

        layout.addWidget(box)

    def _load_example_data(self, example_key):
        """EXAMPLE_DATASETS[key]에 따라 데이터 테이블을 채우고 테이블 유형 전환."""
        example = EXAMPLE_DATASETS.get(example_key)
        if not example:
            QMessageBox.warning(self, "예시 없음",
                                f"'{example_key}' 예시 데이터가 정의되어 있지 않습니다.")
            return

        # 특수 케이스: 기존 load_sample_data 재사용
        if example.get('use_load_sample'):
            self.load_sample_data()
            return

        # 테이블 유형 먼저 설정
        t = example['table_type']
        self.current_table_type = t
        self.table_type_combo.setCurrentText(TABLE_TYPE_LABELS[t])
        self._refresh_tabs_enabled()
        self._refresh_analysis_choices()

        # 테이블 크기 맞추고 값 채우기
        rows = example['rows']
        cols = example['columns']
        self.row_spin.setValue(len(rows))
        self.col_spin.setValue(len(cols) - 1)  # 첫 번째 빈 열 제외
        self.apply_table_size()

        # 첫 행: 열 이름
        for j, name in enumerate(cols):
            self.data_table.setItem(0, j, QTableWidgetItem(str(name)))
        # 데이터 행들
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                self.data_table.setItem(i + 1, j, QTableWidgetItem(str(val)))

        # 미리보기 초기화 + 안내
        self.stats_text.setPlainText(
            f"[예시 데이터 로드 완료]\n"
            f"  유형: {TABLE_TYPE_LABELS[t]}\n"
            f"  행 × 열: {len(rows)} × {len(cols) - 1}\n\n"
            f"이제 아래 '그래프 생성 / 미리보기' 버튼을 눌러보세요.")

    # =====================================================================
    # 기존 5종 차트 탭
    # =====================================================================
    def _setup_heatmap_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'heatmap')

        grp = QGroupBox("히트맵 설정")
        form = QFormLayout(grp)

        self.hm_cmap = QComboBox()
        self.hm_cmap.addItems(presets.DIVERGING_PALETTES + presets.SEQUENTIAL_PALETTES)
        self.hm_cmap.setCurrentText('RdYlGn')
        form.addRow("컬러맵:", self.hm_cmap)

        def dsb(v):
            s = QDoubleSpinBox(); s.setRange(-1e9, 1e9); s.setDecimals(2); s.setValue(v); return s
        self.hm_vmin = dsb(0); self.hm_vmax = dsb(100)
        rr = QHBoxLayout(); rr.addWidget(self.hm_vmin); rr.addWidget(QLabel("~")); rr.addWidget(self.hm_vmax)
        w = QWidget(); w.setLayout(rr)
        form.addRow("값 범위:", w)

        self.hm_cbar_label = QLineEdit("Value")
        form.addRow("컬러바 라벨:", self.hm_cbar_label)
        self.hm_cbar_size = QSpinBox(); self.hm_cbar_size.setRange(6, 24); self.hm_cbar_size.setValue(11)
        form.addRow("컬러바 크기:", self.hm_cbar_size)

        self.hm_fmt = QComboBox()
        self.hm_fmt.addItems(['.0f', '.1f', '.2f', '.3f', 'd', '.2e'])
        form.addRow("숫자 포맷:", self.hm_fmt)

        self.hm_square = QCheckBox("정사각형 셀"); form.addRow("", self.hm_square)
        self.hm_annot = QCheckBox("셀 값 표시"); self.hm_annot.setChecked(True); form.addRow("", self.hm_annot)
        self.hm_xrot = QSpinBox(); self.hm_xrot.setRange(0, 90); self.hm_xrot.setValue(45)
        form.addRow("X축 회전:", self.hm_xrot)

        layout.addWidget(grp)
        self._font_row(layout, 'hm', title_size=14, axis_size=11, tick_size=10,
                       show_annot=True, annot_size=11)
        self._tab_action_row(layout, self.generate_heatmap)
        self._register_tab(content, 'heatmap', CHART_TAB_NAMES['heatmap'])

    def _setup_boxplot_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'boxplot')

        grp = QGroupBox("박스플롯 설정")
        form = QFormLayout(grp)
        self.bx_sort = QComboBox()
        self.bx_sort.addItems(['평균값 내림차순', '평균값 오름차순', '중앙값 내림차순', '중앙값 오름차순', '원본 순서'])
        form.addRow("정렬:", self.bx_sort)
        self.bx_palette = QComboBox()
        self.bx_palette.addItems(presets.QUALITATIVE_PALETTES)
        form.addRow("팔레트:", self.bx_palette)
        self.bx_xlabel = QLineEdit("Group"); form.addRow("X축 라벨:", self.bx_xlabel)
        self.bx_ylabel = QLineEdit("Value"); form.addRow("Y축 라벨:", self.bx_ylabel)
        self._axis_range_row(form, 'bx', y_default=(0, 100))

        self.bx_notch = QCheckBox("노치 박스"); form.addRow("", self.bx_notch)
        self.bx_fliers = QCheckBox("이상치 표시"); self.bx_fliers.setChecked(True); form.addRow("", self.bx_fliers)
        self.bx_show_mean = QCheckBox("평균 마커"); self.bx_show_mean.setChecked(True); form.addRow("", self.bx_show_mean)
        self.bx_show_mean_lbl = QCheckBox("평균 숫자"); self.bx_show_mean_lbl.setChecked(True); form.addRow("", self.bx_show_mean_lbl)
        self.bx_show_points = QCheckBox("개별 점 오버레이"); form.addRow("", self.bx_show_points)
        self.bx_despine = QCheckBox("위/오른쪽 축선 제거"); self.bx_despine.setChecked(True); form.addRow("", self.bx_despine)
        self.bx_grid = QCheckBox("Y 그리드"); self.bx_grid.setChecked(True); form.addRow("", self.bx_grid)
        self.bx_xrot = QSpinBox(); self.bx_xrot.setRange(0, 90); self.bx_xrot.setValue(45)
        form.addRow("X축 회전:", self.bx_xrot)

        layout.addWidget(grp)
        self._font_row(layout, 'bx', title_size=13, axis_size=11, tick_size=10)
        self._stats_row(layout, 'bx')
        self._tab_action_row(layout, self.generate_boxplot)
        self._register_tab(content, 'boxplot', CHART_TAB_NAMES['boxplot'])

    def _setup_bar_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'bar')

        grp = QGroupBox("막대그래프 설정")
        form = QFormLayout(grp)
        self.br_sort = QComboBox()
        self.br_sort.addItems(['평균값 내림차순', '평균값 오름차순', '원본 순서'])
        form.addRow("정렬:", self.br_sort)
        self.br_palette = QComboBox()
        self.br_palette.addItems(presets.QUALITATIVE_PALETTES)
        form.addRow("팔레트:", self.br_palette)
        self.br_err_mode = QComboBox()
        self.br_err_mode.addItems(['SEM', 'SD', '95%CI', 'none'])
        form.addRow("오차막대:", self.br_err_mode)

        self.br_xlabel = QLineEdit("Group"); form.addRow("X축 라벨:", self.br_xlabel)
        self.br_ylabel = QLineEdit("Value"); form.addRow("Y축 라벨:", self.br_ylabel)
        self._axis_range_row(form, 'br', y_default=(0, 100))

        self.br_show_points = QCheckBox("개별 점 표시"); self.br_show_points.setChecked(True); form.addRow("", self.br_show_points)
        self.br_show_mean_lbl = QCheckBox("평균 숫자"); form.addRow("", self.br_show_mean_lbl)
        self.br_despine = QCheckBox("위/오른쪽 축선 제거"); self.br_despine.setChecked(True); form.addRow("", self.br_despine)
        self.br_grid = QCheckBox("Y 그리드"); self.br_grid.setChecked(True); form.addRow("", self.br_grid)
        self.br_xrot = QSpinBox(); self.br_xrot.setRange(0, 90); self.br_xrot.setValue(45)
        form.addRow("X축 회전:", self.br_xrot)

        layout.addWidget(grp)
        self._font_row(layout, 'br', title_size=13, axis_size=11, tick_size=10)
        self._stats_row(layout, 'br')
        self._tab_action_row(layout, self.generate_bar)
        self._register_tab(content, 'bar', CHART_TAB_NAMES['bar'])

    def _setup_violin_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'violin')

        grp = QGroupBox("바이올린 설정")
        form = QFormLayout(grp)
        self.vl_sort = QComboBox()
        self.vl_sort.addItems(['평균값 내림차순', '평균값 오름차순', '중앙값 내림차순', '중앙값 오름차순', '원본 순서'])
        form.addRow("정렬:", self.vl_sort)
        self.vl_palette = QComboBox()
        self.vl_palette.addItems(presets.QUALITATIVE_PALETTES)
        form.addRow("팔레트:", self.vl_palette)
        self.vl_xlabel = QLineEdit("Group"); form.addRow("X축 라벨:", self.vl_xlabel)
        self.vl_ylabel = QLineEdit("Value"); form.addRow("Y축 라벨:", self.vl_ylabel)
        self._axis_range_row(form, 'vl', y_default=(0, 100))

        self.vl_inner_box = QCheckBox("내부 박스플롯"); self.vl_inner_box.setChecked(True); form.addRow("", self.vl_inner_box)
        self.vl_show_points = QCheckBox("개별 점 오버레이"); form.addRow("", self.vl_show_points)
        self.vl_despine = QCheckBox("위/오른쪽 축선 제거"); self.vl_despine.setChecked(True); form.addRow("", self.vl_despine)
        self.vl_grid = QCheckBox("Y 그리드"); self.vl_grid.setChecked(True); form.addRow("", self.vl_grid)
        self.vl_xrot = QSpinBox(); self.vl_xrot.setRange(0, 90); self.vl_xrot.setValue(45)
        form.addRow("X축 회전:", self.vl_xrot)

        layout.addWidget(grp)
        self._font_row(layout, 'vl', title_size=13, axis_size=11, tick_size=10)
        self._stats_row(layout, 'vl')
        self._tab_action_row(layout, self.generate_violin)
        self._register_tab(content, 'violin', CHART_TAB_NAMES['violin'])

    def _setup_scatter_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'scatter')

        grp = QGroupBox("산점도 설정 (첫 열=X, 나머지=Y 시리즈)")
        form = QFormLayout(grp)
        self.sc_palette = QComboBox()
        self.sc_palette.addItems(presets.QUALITATIVE_PALETTES)
        self.sc_palette.setCurrentText('tab10')
        form.addRow("팔레트:", self.sc_palette)
        self.sc_xlabel = QLineEdit(""); self.sc_xlabel.setPlaceholderText("첫 열 이름 사용")
        form.addRow("X축 라벨:", self.sc_xlabel)
        self.sc_ylabel = QLineEdit("Y"); form.addRow("Y축 라벨:", self.sc_ylabel)
        self._axis_range_row(form, 'sc', include_x=True, x_default=(0, 100), y_default=(0, 100))

        self.sc_point_size = QSpinBox(); self.sc_point_size.setRange(5, 200); self.sc_point_size.setValue(45)
        form.addRow("점 크기:", self.sc_point_size)

        self.sc_show_reg = QCheckBox("회귀선"); self.sc_show_reg.setChecked(True); form.addRow("", self.sc_show_reg)
        self.sc_show_eq = QCheckBox("방정식/R² 표시"); self.sc_show_eq.setChecked(True); form.addRow("", self.sc_show_eq)
        self.sc_despine = QCheckBox("위/오른쪽 축선 제거"); self.sc_despine.setChecked(True); form.addRow("", self.sc_despine)
        self.sc_grid = QCheckBox("그리드"); self.sc_grid.setChecked(True); form.addRow("", self.sc_grid)

        self.sc_legend_loc = QComboBox()
        self.sc_legend_loc.addItems(['best', 'upper right', 'upper left', 'lower right', 'lower left',
                                      'upper center', 'lower center', 'center left', 'center right', 'center'])
        form.addRow("범례 위치:", self.sc_legend_loc)

        layout.addWidget(grp)
        self._font_row(layout, 'sc', title_size=13, axis_size=11, tick_size=10)
        self._tab_action_row(layout, self.generate_scatter)
        self._register_tab(content, 'scatter', CHART_TAB_NAMES['scatter'])

    # =====================================================================
    # 신규 4종 차트 탭
    # =====================================================================
    def _setup_grouped_bar_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'grouped_bar')

        grp = QGroupBox("그룹 막대그래프 설정 (GROUPED 전용)")
        form = QFormLayout(grp)
        self.gb_palette = QComboBox()
        self.gb_palette.addItems(presets.QUALITATIVE_PALETTES)
        self.gb_palette.setCurrentText('Set2')
        form.addRow("팔레트:", self.gb_palette)

        self.gb_err_mode = QComboBox()
        self.gb_err_mode.addItems(['SEM', 'SD', '95%CI', 'none'])
        form.addRow("오차막대:", self.gb_err_mode)

        self.gb_xlabel = QLineEdit(""); self.gb_xlabel.setPlaceholderText("요인1 이름 사용")
        form.addRow("X축 라벨:", self.gb_xlabel)
        self.gb_ylabel = QLineEdit("Value"); form.addRow("Y축 라벨:", self.gb_ylabel)
        self._axis_range_row(form, 'gb', y_default=(0, 100))

        self.gb_show_points = QCheckBox("개별 점 표시"); form.addRow("", self.gb_show_points)
        self.gb_show_brackets = QCheckBox("유의성 브래킷"); form.addRow("", self.gb_show_brackets)

        self.gb_xrot = QSpinBox(); self.gb_xrot.setRange(0, 90); self.gb_xrot.setValue(0)
        form.addRow("X축 회전:", self.gb_xrot)

        layout.addWidget(grp)
        self._font_row(layout, 'gb', title_size=13, axis_size=11, tick_size=10)
        self._tab_action_row(layout, self.generate_grouped_bar)
        self._register_tab(content, 'grouped_bar', CHART_TAB_NAMES['grouped_bar'])

    def _setup_interaction_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'interaction')

        grp = QGroupBox("상호작용 플롯 설정 (GROUPED 전용)")
        form = QFormLayout(grp)
        self.it_palette = QComboBox()
        self.it_palette.addItems(presets.QUALITATIVE_PALETTES)
        self.it_palette.setCurrentText('Set1')
        form.addRow("팔레트:", self.it_palette)

        self.it_err_mode = QComboBox()
        self.it_err_mode.addItems(['SEM', 'SD', '95%CI', 'none'])
        form.addRow("오차바:", self.it_err_mode)

        self.it_show_errors = QCheckBox("에러바 표시"); self.it_show_errors.setChecked(True)
        form.addRow("", self.it_show_errors)

        self.it_xlabel = QLineEdit(""); self.it_xlabel.setPlaceholderText("요인1 이름 사용")
        form.addRow("X축 라벨:", self.it_xlabel)
        self.it_ylabel = QLineEdit("Value"); form.addRow("Y축 라벨:", self.it_ylabel)
        self._axis_range_row(form, 'it', y_default=(0, 100))

        self.it_xrot = QSpinBox(); self.it_xrot.setRange(0, 90); self.it_xrot.setValue(0)
        form.addRow("X축 회전:", self.it_xrot)

        layout.addWidget(grp)
        self._font_row(layout, 'it', title_size=13, axis_size=11, tick_size=10)
        self._tab_action_row(layout, self.generate_interaction_plot)
        self._register_tab(content, 'interaction', CHART_TAB_NAMES['interaction'])

    def _setup_dose_response_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'dose_response')

        grp = QGroupBox("용량-반응 곡선 설정 (DOSE_RESPONSE 전용)")
        form = QFormLayout(grp)

        self.dr_model = QComboBox()
        self.dr_model.addItems(DOSE_RESPONSE_MODELS)
        form.addRow("모델:", self.dr_model)

        # 팔레트 + 색 선택
        color_row = QHBoxLayout()
        self.dr_color_btn = QPushButton(" 색 선택 ")
        self.dr_color_btn.clicked.connect(self._pick_dose_response_color)
        self._refresh_dr_color_button()
        color_row.addWidget(self.dr_color_btn)
        color_row.addStretch()
        w = QWidget(); w.setLayout(color_row)
        form.addRow("곡선 색:", w)

        self.dr_dose_unit = QLineEdit(""); self.dr_dose_unit.setPlaceholderText("예: mg/L")
        form.addRow("용량 단위:", self.dr_dose_unit)

        self.dr_xlabel = QLineEdit(""); self.dr_xlabel.setPlaceholderText("비우면 자동 라벨")
        form.addRow("X축 라벨:", self.dr_xlabel)
        self.dr_ylabel = QLineEdit("Mortality (%)")
        form.addRow("Y축 라벨:", self.dr_ylabel)
        self._axis_range_row(form, 'dr', y_default=(0, 100))

        self.dr_show_ci = QCheckBox("95% 신뢰구간 밴드"); self.dr_show_ci.setChecked(True)
        form.addRow("", self.dr_show_ci)
        self.dr_show_ec50 = QCheckBox("EC50 마커/주석"); self.dr_show_ec50.setChecked(True)
        form.addRow("", self.dr_show_ec50)

        self.dr_err_mode = QComboBox()
        self.dr_err_mode.addItems(['SEM', 'SD', 'none'])
        form.addRow("오차바:", self.dr_err_mode)

        self.dr_point_size = QSpinBox(); self.dr_point_size.setRange(5, 200); self.dr_point_size.setValue(36)
        form.addRow("점 크기:", self.dr_point_size)

        layout.addWidget(grp)
        self._font_row(layout, 'dr', title_size=13, axis_size=11, tick_size=10)

        # 피팅 + 렌더를 하나의 버튼으로
        row = QHBoxLayout()
        btn = QPushButton("피팅 + 곡선 그리기")
        btn.clicked.connect(self.generate_dose_response)
        btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        row.addWidget(btn)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        btn2 = QPushButton("곡선만 다시 그리기 (마지막 피팅 사용)")
        btn2.clicked.connect(self._redraw_dose_response)
        row2.addWidget(btn2)
        layout.addLayout(row2)
        layout.addStretch()

        self._register_tab(content, 'dose_response', CHART_TAB_NAMES['dose_response'])

    def _setup_xy_error_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self._add_help_box(layout, 'xy_error')

        grp = QGroupBox("XY + 에러바 설정 (XY 전용)")
        form = QFormLayout(grp)

        self.xe_palette = QComboBox()
        self.xe_palette.addItems(presets.QUALITATIVE_PALETTES)
        self.xe_palette.setCurrentText('tab10')
        form.addRow("팔레트:", self.xe_palette)

        self.xe_err_mode = QComboBox()
        self.xe_err_mode.addItems(['SEM', 'SD', '95%CI', 'none'])
        form.addRow("오차바:", self.xe_err_mode)

        self.xe_connect = QCheckBox("라인 연결"); self.xe_connect.setChecked(True); form.addRow("", self.xe_connect)
        self.xe_log_x = QCheckBox("로그 X축"); form.addRow("", self.xe_log_x)
        self.xe_log_y = QCheckBox("로그 Y축"); form.addRow("", self.xe_log_y)

        self.xe_line_width = QDoubleSpinBox(); self.xe_line_width.setRange(0.1, 10.0)
        self.xe_line_width.setDecimals(1); self.xe_line_width.setValue(1.4)
        form.addRow("라인 굵기:", self.xe_line_width)

        self.xe_marker_size = QSpinBox(); self.xe_marker_size.setRange(1, 40); self.xe_marker_size.setValue(6)
        form.addRow("마커 크기:", self.xe_marker_size)

        self.xe_xlabel = QLineEdit(""); self.xe_xlabel.setPlaceholderText("X 열 이름 사용")
        form.addRow("X축 라벨:", self.xe_xlabel)
        self.xe_ylabel = QLineEdit("Y"); form.addRow("Y축 라벨:", self.xe_ylabel)
        self._axis_range_row(form, 'xe', include_x=True, x_default=(0, 100), y_default=(0, 100))

        layout.addWidget(grp)
        self._font_row(layout, 'xe', title_size=13, axis_size=11, tick_size=10)
        self._tab_action_row(layout, self.generate_xy_error)
        self._register_tab(content, 'xy_error', CHART_TAB_NAMES['xy_error'])

    # =====================================================================
    # 분석 패널
    # =====================================================================
    def _setup_analysis_panel(self, parent_layout):
        stats_group = QGroupBox("통계 분석 결과")
        stats_layout = QVBoxLayout(stats_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("분석 선택:"))
        self.analysis_combo = QComboBox()
        row.addWidget(self.analysis_combo, 1)
        self.run_analysis_btn = QPushButton("분석 실행")
        self.run_analysis_btn.clicked.connect(self.run_selected_analysis)
        self.run_analysis_btn.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 6px;")
        row.addWidget(self.run_analysis_btn)
        stats_layout.addLayout(row)

        self.stats_text = QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        # 라틴(Consolas) + 한글(Malgun Gothic) 폴백 체인
        # Qt는 문자별로 font-family 리스트에서 글리프를 가진 폰트를 자동 선택
        self.stats_text.setStyleSheet(
            "font-family: 'D2Coding', 'NanumGothicCoding', 'Cascadia Mono', "
            "Consolas, 'Courier New', 'Malgun Gothic', monospace; "
            "font-size: 11px;")
        stats_layout.addWidget(self.stats_text)
        parent_layout.addWidget(stats_group, 2)

    # =====================================================================
    # 설정 수집 헬퍼
    # =====================================================================
    def _base_font_cfg(self, p):
        return {
            'title': getattr(self, f'{p}_title_edit').text(),
            'subtitle': getattr(self, f'{p}_subtitle_edit').text()
                        if hasattr(self, f'{p}_subtitle_edit') else '',
            'title_size': getattr(self, f'{p}_title_size').value(),
            'title_bold': getattr(self, f'{p}_title_bold').isChecked(),
            'title_italic': getattr(self, f'{p}_title_italic').isChecked(),
            'axis_size': getattr(self, f'{p}_axis_size').value(),
            'axis_bold': getattr(self, f'{p}_axis_bold').isChecked(),
            'axis_italic': getattr(self, f'{p}_axis_italic').isChecked(),
            'tick_size': getattr(self, f'{p}_tick_size').value(),
            'tick_bold': getattr(self, f'{p}_tick_bold').isChecked(),
            'tick_italic': getattr(self, f'{p}_tick_italic').isChecked(),
            'italic_texts': self._parse_italic(getattr(self, f'{p}_italic_texts').text()),
        }

    def _parse_italic(self, text):
        text = text.strip()
        if not text:
            return []
        return [t.strip() for t in text.split(',') if t.strip()]

    def _stats_cfg(self, p):
        return {
            'show_brackets': getattr(self, f'{p}_show_brackets').isChecked(),
            'test': getattr(self, f'{p}_test').currentText(),
            'correction': getattr(self, f'{p}_correction').currentText(),
        }

    # =====================================================================
    # 생성 헬퍼
    # =====================================================================
    def _validate_df(self, df):
        if df.empty or df.isna().all().all():
            QMessageBox.warning(self, "경고", "유효한 데이터를 입력해주세요.")
            return False
        return True

    def _set_current_fig(self, fig, graph_type):
        if self.current_fig is not None and self.current_fig is not fig:
            plt.close(self.current_fig)
        self.current_fig = fig
        self.current_graph_type = graph_type
        self._show_preview(fig)

    def _require_table_type(self, chart_key, required_type):
        """현재 테이블 유형이 요구 유형과 다르면 친절한 메시지 표시 후 False."""
        if self.current_table_type == required_type:
            return True
        label = TABLE_TYPE_LABELS[required_type]
        QMessageBox.information(
            self, "데이터 유형 불일치",
            f"이 차트는 '{label}' 데이터 타입에서 사용 가능합니다.\n위에서 타입을 변경하세요.")
        return False

    # =====================================================================
    # 한글 폰트 자동 전환 가드
    # =====================================================================
    def _guard_font_for_render(self, prefix):
        """이 차트의 텍스트에 한글이 있으면 matplotlib 폰트를 한글용으로 임시 전환.

        prefix: 'hm'/'bx'/... 9개 차트 prefix 중 하나
        """
        user_font = self.font_combo.currentText()
        texts = []
        # 탭 내부 텍스트 위젯들
        for suffix in ('title_edit', 'subtitle_edit', 'xlabel', 'ylabel',
                       'italic_texts', 'dose_unit'):
            w = getattr(self, f'{prefix}_{suffix}', None)
            if w is not None and hasattr(w, 'text'):
                texts.append(w.text())
        # 데이터 테이블의 행/열 이름 (축 라벨로 실제 렌더됨)
        try:
            df = self.get_table_data_raw()
            texts.extend(str(c) for c in df.columns)
            texts.extend(str(i) for i in df.index)
        except Exception:
            pass

        eff = choose_render_font(user_font, texts)
        if eff != user_font:
            set_global_font(eff)

    # =====================================================================
    # 기존 5종 렌더
    # =====================================================================
    def generate_heatmap(self):
        self._guard_font_for_render('hm')
        try:
            df = self.get_table_data()
            if not self._validate_df(df):
                return
            cfg = self._base_font_cfg('hm')
            cfg.update({
                'cmap': self.hm_cmap.currentText(),
                'vmin': self.hm_vmin.value(),
                'vmax': self.hm_vmax.value(),
                'cbar_label': self.hm_cbar_label.text(),
                'cbar_size': self.hm_cbar_size.value(),
                'fmt': self.hm_fmt.currentText(),
                'square': self.hm_square.isChecked(),
                'annot': self.hm_annot.isChecked(),
                'xtick_rotation': self.hm_xrot.value(),
                'annot_size': self.hm_annot_size.value(),
                'annot_bold': self.hm_annot_bold.isChecked(),
                'annot_italic': self.hm_annot_italic.isChecked(),
            })
            fig = charts.render_heatmap(df, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'heatmap')
            self._update_stats(df, include_pairwise=False)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"히트맵 생성 실패:\n{e}")

    def generate_boxplot(self):
        self._guard_font_for_render('bx')
        try:
            df = self.get_table_data()
            if not self._validate_df(df):
                return
            cfg = self._base_font_cfg('bx')
            cfg.update({
                'sort': self.bx_sort.currentText(),
                'palette': self.bx_palette.currentText(),
                'xlabel': self.bx_xlabel.text(),
                'ylabel': self.bx_ylabel.text(),
                'ymin': self.bx_ymin.value(),
                'ymax': self.bx_ymax.value(),
                'notch': self.bx_notch.isChecked(),
                'showfliers': self.bx_fliers.isChecked(),
                'show_mean': self.bx_show_mean.isChecked(),
                'show_mean_label': self.bx_show_mean_lbl.isChecked(),
                'show_points': self.bx_show_points.isChecked(),
                'despine': self.bx_despine.isChecked(),
                'grid': self.bx_grid.isChecked(),
                'xtick_rotation': self.bx_xrot.value(),
            })
            cfg.update(self._stats_cfg('bx'))
            fig = charts.render_boxplot(df, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'boxplot')
            self._update_stats(df, include_pairwise=cfg['show_brackets'],
                               test=cfg['test'], correction=cfg['correction'])
        except Exception as e:
            QMessageBox.critical(self, "오류", f"박스플롯 생성 실패:\n{e}")

    def generate_bar(self):
        self._guard_font_for_render('br')
        try:
            df = self.get_table_data()
            if not self._validate_df(df):
                return
            cfg = self._base_font_cfg('br')
            cfg.update({
                'sort': self.br_sort.currentText(),
                'palette': self.br_palette.currentText(),
                'err_mode': self.br_err_mode.currentText(),
                'xlabel': self.br_xlabel.text(),
                'ylabel': self.br_ylabel.text(),
                'ymin': self.br_ymin.value(),
                'ymax': self.br_ymax.value(),
                'show_points': self.br_show_points.isChecked(),
                'show_mean_label': self.br_show_mean_lbl.isChecked(),
                'despine': self.br_despine.isChecked(),
                'grid': self.br_grid.isChecked(),
                'xtick_rotation': self.br_xrot.value(),
            })
            cfg.update(self._stats_cfg('br'))
            fig = charts.render_barplot(df, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'bar')
            self._update_stats(df, include_pairwise=cfg['show_brackets'],
                               test=cfg['test'], correction=cfg['correction'])
        except Exception as e:
            QMessageBox.critical(self, "오류", f"막대그래프 생성 실패:\n{e}")

    def generate_violin(self):
        self._guard_font_for_render('vl')
        try:
            df = self.get_table_data()
            if not self._validate_df(df):
                return
            cfg = self._base_font_cfg('vl')
            cfg.update({
                'sort': self.vl_sort.currentText(),
                'palette': self.vl_palette.currentText(),
                'xlabel': self.vl_xlabel.text(),
                'ylabel': self.vl_ylabel.text(),
                'ymin': self.vl_ymin.value(),
                'ymax': self.vl_ymax.value(),
                'inner_box': self.vl_inner_box.isChecked(),
                'show_points': self.vl_show_points.isChecked(),
                'despine': self.vl_despine.isChecked(),
                'grid': self.vl_grid.isChecked(),
                'xtick_rotation': self.vl_xrot.value(),
            })
            cfg.update(self._stats_cfg('vl'))
            fig = charts.render_violin(df, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'violin')
            self._update_stats(df, include_pairwise=cfg['show_brackets'],
                               test=cfg['test'], correction=cfg['correction'])
        except Exception as e:
            QMessageBox.critical(self, "오류", f"바이올린 생성 실패:\n{e}")

    def generate_scatter(self):
        self._guard_font_for_render('sc')
        try:
            df = self.get_table_data()
            if not self._validate_df(df):
                return
            if len(df.columns) < 2:
                QMessageBox.warning(self, "경고",
                    "산점도는 최소 2개 열이 필요합니다 (첫 열=X, 이후=Y).")
                return
            cfg = self._base_font_cfg('sc')
            xlabel = self.sc_xlabel.text().strip() or str(df.columns[0])
            cfg.update({
                'palette': self.sc_palette.currentText(),
                'xlabel': xlabel,
                'ylabel': self.sc_ylabel.text(),
                'xmin': self.sc_xmin.value(),
                'xmax': self.sc_xmax.value(),
                'ymin': self.sc_ymin.value(),
                'ymax': self.sc_ymax.value(),
                'point_size': self.sc_point_size.value(),
                'show_regression': self.sc_show_reg.isChecked(),
                'show_reg_equation': self.sc_show_eq.isChecked(),
                'despine': self.sc_despine.isChecked(),
                'grid': self.sc_grid.isChecked(),
                'legend_loc': self.sc_legend_loc.currentText(),
            })
            fig = charts.render_scatter(df, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'scatter')
            self._update_scatter_stats(df)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"산점도 생성 실패:\n{e}")

    # =====================================================================
    # 신규 4종 렌더
    # =====================================================================
    def _parse_grouped_from_table(self):
        """테이블 → data_model.parse_table(GROUPED). 성공시 parsed dict, 실패시 None."""
        df_raw = self.get_table_data_raw()
        df = self._df_for_parse(df_raw)
        try:
            return dm.parse_table(df, dm.DataTableType.GROUPED)
        except Exception as e:
            QMessageBox.critical(self, "GROUPED 파싱 실패", str(e))
            return None

    def _parse_xy_from_table(self):
        df_raw = self.get_table_data()
        df = self._df_for_parse(df_raw)
        try:
            return dm.parse_table(df, dm.DataTableType.XY)
        except Exception as e:
            QMessageBox.critical(self, "XY 파싱 실패", str(e))
            return None

    def _parse_dose_response_from_table(self):
        df_raw = self.get_table_data()
        df = self._df_for_parse(df_raw)
        try:
            return dm.parse_table(df, dm.DataTableType.DOSE_RESPONSE)
        except Exception as e:
            QMessageBox.critical(self, "DOSE-RESPONSE 파싱 실패", str(e))
            return None

    def generate_grouped_bar(self):
        self._guard_font_for_render('gb')
        if not self._require_table_type('grouped_bar', dm.DataTableType.GROUPED):
            return
        parsed = self._parse_grouped_from_table()
        if parsed is None:
            return
        try:
            cfg = self._base_font_cfg('gb')
            cfg.update({
                'palette': self.gb_palette.currentText(),
                'err_mode': self.gb_err_mode.currentText(),
                'show_points': self.gb_show_points.isChecked(),
                'show_brackets': self.gb_show_brackets.isChecked(),
                'xlabel': self.gb_xlabel.text() or parsed.get('factor1_name', 'factor1'),
                'ylabel': self.gb_ylabel.text(),
                'ymin': self.gb_ymin.value(),
                'ymax': self.gb_ymax.value(),
                'xtick_rotation': self.gb_xrot.value(),
                'test': 't-test',
                'correction': 'bonferroni',
            })
            fig = charts.render_grouped_bar(parsed, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'grouped_bar')
            self.stats_text.setPlainText(
                "[Grouped Bar 생성] 분석 패널의 '분석 선택'에서 "
                "Two-way ANOVA / Tukey HSD / Dunnett 을 실행해 보세요.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"그룹 막대 생성 실패:\n{e}")

    def generate_interaction_plot(self):
        self._guard_font_for_render('it')
        if not self._require_table_type('interaction', dm.DataTableType.GROUPED):
            return
        parsed = self._parse_grouped_from_table()
        if parsed is None:
            return
        try:
            cfg = self._base_font_cfg('it')
            cfg.update({
                'palette': self.it_palette.currentText(),
                'err_mode': self.it_err_mode.currentText(),
                'show_errors': self.it_show_errors.isChecked(),
                'xlabel': self.it_xlabel.text() or parsed.get('factor1_name', 'factor1'),
                'ylabel': self.it_ylabel.text(),
                'ymin': self.it_ymin.value(),
                'ymax': self.it_ymax.value(),
                'xtick_rotation': self.it_xrot.value(),
            })
            fig = charts.render_interaction_plot(parsed, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'interaction')
            self.stats_text.setPlainText(
                "[Interaction Plot 생성] 두 선이 평행하지 않으면 상호작용이 존재할 수 있습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"상호작용 플롯 생성 실패:\n{e}")

    def generate_dose_response(self):
        self._guard_font_for_render('dr')
        """피팅 + 곡선 렌더 원스텝."""
        if not self._require_table_type('dose_response', dm.DataTableType.DOSE_RESPONSE):
            return
        parsed = self._parse_dose_response_from_table()
        if parsed is None:
            return
        # 피팅
        fit_result = self._fit_dose_response(parsed, self.dr_model.currentText())
        self.current_fit_result = fit_result
        self._render_dose_response(parsed, fit_result)
        if fit_result is not None and fit_result.converged:
            # 통계 패널에 결과 표시
            self.stats_text.setPlainText(self._format_fit_result_kor(fit_result, parsed))
            self.collected_fits.append(fit_result)
        elif fit_result is not None:
            self.stats_text.setPlainText(
                f"피팅 실패: {fit_result.message}\n\n"
                f"모델: {fit_result.model_name}")

    def _redraw_dose_response(self):
        """마지막 피팅을 유지한 채 UI 옵션만 바꿔서 다시 그리기."""
        if not self._require_table_type('dose_response', dm.DataTableType.DOSE_RESPONSE):
            return
        parsed = self._parse_dose_response_from_table()
        if parsed is None:
            return
        self._render_dose_response(parsed, self.current_fit_result)

    def _fit_dose_response(self, parsed, model_name):
        """parsed(DOSE_RESPONSE) + 모델 이름 → FitResult. 피팅 대상은 (x=dose, y=response)."""
        model_cls = cm.AVAILABLE_MODELS.get(model_name)
        if model_cls is None:
            QMessageBox.warning(self, "모델 오류", f"알 수 없는 모델: {model_name}")
            return None

        dose = np.asarray(parsed['dose'], dtype=float)
        # 모든 반복측정을 (x, y) 쌍으로 펼침
        xs, ys = [], []
        for rep in parsed.get('responses', []):
            arr = np.asarray(rep, dtype=float)
            n = min(len(dose), len(arr))
            for i in range(n):
                if np.isfinite(dose[i]) and np.isfinite(arr[i]):
                    xs.append(dose[i])
                    ys.append(arr[i])
        if len(xs) < 3:
            QMessageBox.warning(self, "피팅 불가", "유효한 (용량, 반응) 쌍이 3개 이상 필요합니다.")
            return None

        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)

        # Hill/LogLogistic 등은 log-dose 입력
        use_log_x = model_name in ('Hill 4PL', 'Hill 5PL', 'LogLogistic 4PL')
        if use_log_x:
            mask = x_arr > 0
            if not np.all(mask):
                x_arr = x_arr[mask]
                y_arr = y_arr[mask]
            if len(x_arr) < 3:
                QMessageBox.warning(self, "피팅 불가", "양수 용량이 3개 이상 필요합니다.")
                return None
            x_arr = np.log10(x_arr)

        try:
            fit = model_cls().fit(x_arr, y_arr)
        except Exception as e:
            QMessageBox.critical(self, "피팅 오류", str(e))
            return None
        return fit

    def _render_dose_response(self, parsed, fit_result):
        try:
            cfg = self._base_font_cfg('dr')
            cfg.update({
                'palette_single': self.dose_response_color,
                'dose_unit': self.dr_dose_unit.text(),
                'xlabel': self.dr_xlabel.text(),
                'ylabel': self.dr_ylabel.text(),
                'ymin': self.dr_ymin.value(),
                'ymax': self.dr_ymax.value(),
                'show_ci_band': self.dr_show_ci.isChecked(),
                'show_ec50_marker': self.dr_show_ec50.isChecked(),
                'err_mode': self.dr_err_mode.currentText(),
                'point_size': self.dr_point_size.value(),
            })
            fig = charts.render_dose_response(parsed, fit_result, cfg,
                                              self._figure_size_inches())
            self._set_current_fig(fig, 'dose_response')
        except Exception as e:
            QMessageBox.critical(self, "오류", f"용량-반응 렌더 실패:\n{e}")

    def _refresh_dr_color_button(self):
        self.dr_color_btn.setStyleSheet(
            f"background-color: {self.dose_response_color}; color: white;"
            " font-weight: bold; padding: 4px 12px;")

    def _pick_dose_response_color(self):
        initial = QColor(self.dose_response_color)
        c = QColorDialog.getColor(initial, self, "곡선 색 선택")
        if c.isValid():
            self.dose_response_color = c.name()
            self._refresh_dr_color_button()

    def generate_xy_error(self):
        self._guard_font_for_render('xe')
        if not self._require_table_type('xy_error', dm.DataTableType.XY):
            return
        parsed = self._parse_xy_from_table()
        if parsed is None:
            return
        try:
            cfg = self._base_font_cfg('xe')
            cfg.update({
                'palette': self.xe_palette.currentText(),
                'err_mode': self.xe_err_mode.currentText(),
                'connect_lines': self.xe_connect.isChecked(),
                'log_x': self.xe_log_x.isChecked(),
                'log_y': self.xe_log_y.isChecked(),
                'line_width': float(self.xe_line_width.value()),
                'marker_size': self.xe_marker_size.value(),
                'xlabel': self.xe_xlabel.text() or parsed.get('x_name', 'X'),
                'ylabel': self.xe_ylabel.text(),
                'xmin': self.xe_xmin.value(),
                'xmax': self.xe_xmax.value(),
                'ymin': self.xe_ymin.value(),
                'ymax': self.xe_ymax.value(),
            })
            fig = charts.render_xy_errorbars(parsed, cfg, self._figure_size_inches())
            self._set_current_fig(fig, 'xy_error')
            self.stats_text.setPlainText(
                "[XY+에러바 생성] 분석 패널에서 '선형회귀' 또는 'Pearson 상관'을 실행해 보세요.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"XY+에러바 생성 실패:\n{e}")

    # =====================================================================
    # 분석 실행
    # =====================================================================
    def run_selected_analysis(self):
        """분석 콤보 선택 → 적절한 stats_utils / curve_models 함수 호출."""
        t = self.current_table_type
        name = self.analysis_combo.currentText()
        try:
            if t == dm.DataTableType.COLUMN:
                self._run_column_analysis(name)
            elif t == dm.DataTableType.XY:
                self._run_xy_analysis(name)
            elif t == dm.DataTableType.GROUPED:
                self._run_grouped_analysis(name)
            elif t == dm.DataTableType.DOSE_RESPONSE:
                self._run_dose_response_analysis(name)
            elif t == dm.DataTableType.SURVIVAL:
                QMessageBox.information(self, "예정", "생존분석은 Phase 6+에 추가됩니다.")
            else:
                QMessageBox.information(self, "선택", f"{name}은(는) 현재 지원되지 않습니다.")
        except Exception as e:
            QMessageBox.critical(self, "분석 실패", str(e))

    def _display_analysis_result(self, result):
        """AnalysisResult 를 stats_text에 표시하고 수집."""
        if result is None:
            self.stats_text.setPlainText("결과 없음 (데이터가 충분하지 않거나 분석에 실패했습니다).")
            return
        self.last_analysis = result
        self.collected_analyses.append(result)
        try:
            self.stats_text.setPlainText(result.to_text())
        except Exception:
            # AnalysisResult가 아닌 경우 (dict 등)
            self.stats_text.setPlainText(str(result))

    # -----------------------------
    # Column 분석
    # -----------------------------
    def _run_column_analysis(self, name):
        df = self.get_table_data()
        if not self._validate_df(df):
            return
        long_df = dm.to_long(self._df_for_parse(df), dm.DataTableType.COLUMN)

        if name == 'Auto (추천)':
            res = stx.auto_compare(long_df, 'value', 'group')
            self._display_analysis_result(res)
            return
        if name == '기술통계':
            self._update_stats(df, include_pairwise=False)
            return
        if name == 'One-way ANOVA + Tukey':
            aov = stx.one_way_anova_long(long_df, 'value', 'group')
            post = None
            if stx.HAS_STATSMODELS:
                try:
                    post = stx.tukey_hsd(long_df, 'value', 'group')
                except Exception:
                    post = None
            if aov is None:
                self._display_analysis_result(None)
                return
            if post is not None:
                combined = stx.AnalysisResult(
                    analysis_name='One-way ANOVA + Tukey HSD',
                    summary={**aov.summary,
                             **{'posthoc_' + k: v for k, v in post.summary.items()}},
                    table=post.table,
                    notes=aov.notes + post.notes,
                )
                self._display_analysis_result(combined)
            else:
                self._display_analysis_result(aov)
            return
        if name == 'Kruskal-Wallis + Dunn':
            kw = stx.kruskal_wallis(long_df, 'value', 'group')
            post = None
            try:
                post = stx.dunn_test(long_df, 'value', 'group', p_adjust='bonferroni')
            except Exception:
                post = None
            if kw is None:
                self._display_analysis_result(None)
                return
            if post is not None:
                combined = stx.AnalysisResult(
                    analysis_name='Kruskal-Wallis + Dunn (Bonferroni)',
                    summary={**kw.summary,
                             **{'posthoc_' + k: v for k, v in post.summary.items()}},
                    table=post.table,
                    notes=kw.notes + post.notes,
                )
                self._display_analysis_result(combined)
            else:
                self._display_analysis_result(kw)
            return
        if name == 'Mann-Whitney (2그룹)':
            groups = list(df.columns)
            if len(groups) < 2:
                QMessageBox.information(self, "선택", "최소 2개 열이 필요합니다.")
                return
            a, ok1 = QInputDialog.getItem(self, "A 그룹", "첫 그룹:", groups, 0, False)
            if not ok1:
                return
            b_items = [g for g in groups if g != a]
            b, ok2 = QInputDialog.getItem(self, "B 그룹", "두 번째 그룹:", b_items, 0, False)
            if not ok2:
                return
            res = stx.mann_whitney(df[a].dropna().values, df[b].dropna().values)
            if res is not None:
                res.analysis_name = f"Mann-Whitney U ({a} vs {b})"
            self._display_analysis_result(res)
            return
        if name == 'Shapiro-Wilk 정규성':
            all_vals = df.values.astype(float).ravel()
            res = stx.shapiro_wilk(all_vals)
            self._display_analysis_result(res)
            return
        if name == 'Levene 등분산':
            cols = [df[c].dropna().values.astype(float) for c in df.columns]
            res = stx.levene(*cols)
            self._display_analysis_result(res)
            return

        # === 추가 정규성 검정 ===
        if name == "D'Agostino-Pearson 정규성":
            all_vals = df.values.astype(float).ravel()
            self._display_analysis_result(stx.dagostino_pearson(all_vals))
            return
        if name == 'Anderson-Darling 정규성':
            all_vals = df.values.astype(float).ravel()
            self._display_analysis_result(stx.anderson_darling(all_vals))
            return

        # === 추가 Post-hoc ===
        if name == 'Holm-Sidak 다중비교':
            self._display_analysis_result(stx.holm_sidak(long_df, 'value', 'group'))
            return
        if name == 'Scheffé 다중비교':
            self._display_analysis_result(stx.scheffe(long_df, 'value', 'group'))
            return

        # === 대응/반복측정 비모수 ===
        if name == 'Wilcoxon 대응표본 (2그룹)':
            groups = list(df.columns)
            if len(groups) < 2:
                QMessageBox.information(self, "선택", "최소 2개 열이 필요합니다.")
                return
            a, ok1 = QInputDialog.getItem(self, "A 그룹", "첫 그룹:", groups, 0, False)
            if not ok1:
                return
            b_items = [g for g in groups if g != a]
            b, ok2 = QInputDialog.getItem(self, "B 그룹", "두 번째 그룹:", b_items, 0, False)
            if not ok2:
                return
            # 대응표본 → 길이 일치 필요, NaN 동시 제거
            pair = df[[a, b]].dropna()
            res = stx.wilcoxon_signed_rank(pair[a].values, pair[b].values)
            if res is not None:
                res.analysis_name = f"Wilcoxon signed-rank ({a} vs {b})"
            self._display_analysis_result(res)
            return
        if name == 'Friedman 반복측정':
            # Friedman은 wide DataFrame 직접 받음
            self._display_analysis_result(stx.friedman(df))
            return

        # === 효과크기 ===
        if name == "Cohen's d (2그룹 효과크기)":
            groups = list(df.columns)
            if len(groups) < 2:
                QMessageBox.information(self, "선택", "최소 2개 열이 필요합니다.")
                return
            a, ok1 = QInputDialog.getItem(self, "A 그룹", "첫 그룹:", groups, 0, False)
            if not ok1:
                return
            b_items = [g for g in groups if g != a]
            b, ok2 = QInputDialog.getItem(self, "B 그룹", "두 번째 그룹:", b_items, 0, False)
            if not ok2:
                return
            res = stx.cohens_d(df[a].dropna().values, df[b].dropna().values)
            if res is not None:
                res.analysis_name = f"Cohen's d ({a} vs {b})"
            self._display_analysis_result(res)
            return

        # === 전체 쌍 상관 행렬 ===
        if name == '상관 행렬 (전체 쌍)':
            method, ok = QInputDialog.getItem(
                self, "상관 방법", "상관 계수 유형:",
                ['pearson', 'spearman', 'kendall'], 0, False)
            if not ok:
                return
            self._display_analysis_result(stx.correlation_matrix(df, method=method))
            return

    # -----------------------------
    # XY 분석
    # -----------------------------
    def _run_xy_analysis(self, name):
        df = self.get_table_data()
        if not self._validate_df(df):
            return
        if len(df.columns) < 2:
            QMessageBox.information(self, "선택", "최소 2개 열(X + Y)이 필요합니다.")
            return

        if name == '기술통계':
            self._update_stats(df, include_pairwise=False)
            return
        if name == '선형회귀':
            self._update_scatter_stats(df)
            return
        x = df.iloc[:, 0].dropna().values.astype(float)
        y_col = df.columns[1]
        # Y가 여러 개라면 선택
        if len(df.columns) > 2:
            chosen, ok = QInputDialog.getItem(self, "Y 열 선택",
                                              "분석할 Y 열:", list(df.columns[1:]), 0, False)
            if not ok:
                return
            y_col = chosen
        y = df[y_col].values.astype(float)
        # x,y mask
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        if name == 'Pearson 상관':
            res = stx.correlation(x, y, method='pearson')
            self._display_analysis_result(res)
        elif name == 'Spearman 상관':
            res = stx.correlation(x, y, method='spearman')
            self._display_analysis_result(res)

    # -----------------------------
    # Grouped 분석
    # -----------------------------
    def _run_grouped_analysis(self, name):
        parsed = self._parse_grouped_from_table()
        if parsed is None:
            return
        long_df = parsed['long_df']
        if long_df.empty:
            self._display_analysis_result(None)
            return
        if name == 'Two-way ANOVA':
            if not stx.HAS_STATSMODELS:
                QMessageBox.warning(self, "statsmodels 필요",
                    "Two-way ANOVA를 위해 'pip install statsmodels' 필요")
                return
            res = stx.two_way_anova(long_df, 'value', 'factor1', 'factor2')
            self._display_analysis_result(res)
            return
        if name == 'Tukey HSD':
            # factor1+factor2 조합을 하나의 그룹 키로 만들어 다중비교
            lf = long_df.copy()
            lf['combined'] = lf['factor1'].astype(str) + ' × ' + lf['factor2'].astype(str)
            res = stx.tukey_hsd(lf, 'value', 'combined')
            self._display_analysis_result(res)
            return
        if name == "Dunnett's (vs. 대조)":
            # 어느 요인에서 대조군을 선택할지 묻기
            factor_axis, ok1 = QInputDialog.getItem(
                self, "기준 요인 선택", "어느 요인에서 대조군을 지정하시겠습니까?",
                ['factor1', 'factor2'], 0, False)
            if not ok1:
                return
            levels = sorted(long_df[factor_axis].astype(str).unique().tolist())
            if len(levels) < 2:
                QMessageBox.information(self, "선택", "수준이 2개 이상이어야 합니다.")
                return
            control, ok2 = QInputDialog.getItem(
                self, "대조군 선택", "대조군(control):", levels, 0, False)
            if not ok2:
                return
            try:
                res = stx.dunnett(long_df, 'value', factor_axis, control)
                self._display_analysis_result(res)
            except Exception as e:
                QMessageBox.critical(self, "Dunnett 오류", str(e))
            return

    # -----------------------------
    # Dose-Response 분석
    # -----------------------------
    def _run_dose_response_analysis(self, name):
        parsed = self._parse_dose_response_from_table()
        if parsed is None:
            return
        if name.endswith('피팅'):
            model_map = {
                'Hill 4PL 피팅': 'Hill 4PL',
                'Hill 5PL 피팅': 'Hill 5PL',
                'Probit 피팅': 'Probit',
                'Logit 피팅': 'Logit',
                'LogLogistic 4PL 피팅': 'LogLogistic 4PL',
            }
            model_name = model_map.get(name)
            if model_name is None:
                QMessageBox.information(self, "선택", f"지원되지 않는 모델: {name}")
                return
            fit = self._fit_dose_response(parsed, model_name)
            self.current_fit_result = fit
            if fit is None:
                return
            # 분석 패널에 결과 표시 + 차트 재렌더
            self.stats_text.setPlainText(self._format_fit_result_kor(fit, parsed))
            if fit.converged:
                self.collected_fits.append(fit)
            self._render_dose_response(parsed, fit)
            return
        if name == '모델 비교':
            # 5개 모델 모두 피팅 → compare table
            models_to_try = [
                cm.Hill4PL(), cm.Hill5PL(), cm.Probit(), cm.Logit(), cm.LogLogistic4P(),
            ]
            # x/y 구성 (log-dose 기준은 Hill 계열에만 적용되므로 원본으로 전달 → 각 모델 내부처리 없음)
            # 실제 모델은 external log 변환을 요구하므로, 여기서는 log-dose로 변환된 단일 세트만 사용
            dose = np.asarray(parsed['dose'], dtype=float)
            xs, ys = [], []
            for rep in parsed.get('responses', []):
                arr = np.asarray(rep, dtype=float)
                n = min(len(dose), len(arr))
                for i in range(n):
                    if np.isfinite(dose[i]) and np.isfinite(arr[i]) and dose[i] > 0:
                        xs.append(np.log10(dose[i]))
                        ys.append(arr[i])
            if len(xs) < 4:
                QMessageBox.warning(self, "모델 비교 불가",
                    "log-dose로 변환된 유효한 (용량, 반응) 쌍이 4개 이상 필요합니다.")
                return
            tbl = cm.compare_models(np.array(xs), np.array(ys), models_to_try)
            # 비교 결과를 텍스트로
            buf = ["모델 비교 (AICc 기준, log10(dose) 입력):", "-" * 72]
            for _, row in tbl.iterrows():
                buf.append(
                    f"{str(row['model']):<25} "
                    f"converged={bool(row.get('converged', False))}, "
                    f"n={int(row.get('n', 0))}, k={int(row.get('k', 0))}, "
                    f"R²={_fmt_float(row.get('r_squared'))}, "
                    f"AICc={_fmt_float(row.get('aicc'), digits=3)}, "
                    f"BIC={_fmt_float(row.get('bic'), digits=3)}"
                )
            self.stats_text.setPlainText('\n'.join(buf))
            # 결과 요약을 수집 분석으로 저장
            res = stx.AnalysisResult(
                analysis_name='Dose-Response Model Comparison',
                table=tbl.drop(columns=['params', 'derived'], errors='ignore'),
                notes=['AICc 최소 모델을 선호합니다.'],
            )
            self.collected_analyses.append(res)
            return
        if name == 'EC50/LC50 계산':
            if self.current_fit_result is None or not self.current_fit_result.converged:
                QMessageBox.information(self, "피팅 필요",
                    "먼저 용량-반응 탭에서 '피팅 + 곡선 그리기'를 실행해 주세요.")
                return
            derived = self.current_fit_result.derived or {}
            if not derived:
                QMessageBox.information(self, "결과 없음", "이 모델에서는 EC50 등을 제공하지 않습니다.")
                return
            lines = ["[ EC50 / 파생 수치 ]"]
            for k, v in derived.items():
                if isinstance(v, tuple) and len(v) == 2:
                    lines.append(f"  {k} = [{_fmt_float(v[0])}, {_fmt_float(v[1])}]")
                else:
                    lines.append(f"  {k} = {_fmt_float(v)}")
            self.stats_text.setPlainText('\n'.join(lines))
            return

        if name == 'LC10/LC90 외삽':
            if self.current_fit_result is None or not self.current_fit_result.converged:
                QMessageBox.information(self, "피팅 필요",
                    "먼저 용량-반응 탭에서 '피팅 + 곡선 그리기'를 실행해 주세요.")
                return
            # 모델 객체 재구성 (현재 fit_result의 model_name으로)
            fit = self.current_fit_result
            model_cls = cm.AVAILABLE_MODELS.get(fit.model_name)
            if model_cls is None:
                QMessageBox.warning(self, "모델 없음",
                    f"'{fit.model_name}' 모델을 찾을 수 없습니다.")
                return
            model = model_cls()
            lines = [f"[ LCx 외삽: {fit.model_name} ]", "-" * 60]
            # 사용자가 원하는 수준들
            levels, ok = QInputDialog.getText(
                self, "반응 수준 지정",
                "계산할 반응 수준 (%) 목록, 쉼표 구분 (예: 10, 25, 50, 75, 90):",
                text="10, 25, 50, 75, 90")
            if not ok:
                return
            try:
                pct_list = [float(x.strip()) / 100.0 for x in levels.split(',') if x.strip()]
            except ValueError:
                QMessageBox.warning(self, "입력 오류", "숫자만 쉼표로 구분해 입력하세요.")
                return
            for pct in pct_list:
                try:
                    val, lo, hi = cm.extrapolate_ecX(fit, model, level=pct)
                    label = f"LC{int(pct * 100)}"
                    # fit.model_name이 log 계열이면 10^x로 역변환 고려
                    if fit.model_name in ('Hill 4PL', 'Hill 5PL', 'LogLogistic 4PL',
                                           'Probit', 'Logit'):
                        val_lin = 10 ** val if np.isfinite(val) else val
                        lo_lin = 10 ** lo if np.isfinite(lo) else lo
                        hi_lin = 10 ** hi if np.isfinite(hi) else hi
                        lines.append(
                            f"  {label}: log-dose={_fmt_float(val)} "
                            f"→ dose={_fmt_float(val_lin)} "
                            f"(95% CI: [{_fmt_float(lo_lin)}, {_fmt_float(hi_lin)}])"
                        )
                    else:
                        lines.append(
                            f"  {label}: {_fmt_float(val)} "
                            f"(95% CI: [{_fmt_float(lo)}, {_fmt_float(hi)}])"
                        )
                except Exception as e:
                    lines.append(f"  {int(pct * 100)}%: 계산 실패 ({e})")
            self.stats_text.setPlainText('\n'.join(lines))
            # 수집 분석에 추가
            res = stx.AnalysisResult(
                analysis_name=f'LCx 외삽 ({fit.model_name})',
                summary={'levels_pct': [round(p * 100, 1) for p in pct_list]},
                notes=['delta method 기반 95% 신뢰구간.'],
            )
            self.collected_analyses.append(res)
            return

        if name == 'F-test (Hill 4PL vs 5PL)':
            # 두 모델을 동시에 피팅 후 중첩 F-test
            parsed = self._parse_dose_response_from_table()
            if parsed is None:
                return
            dose = np.asarray(parsed['dose'], dtype=float)
            xs, ys = [], []
            for rep in parsed.get('responses', []):
                arr = np.asarray(rep, dtype=float)
                n = min(len(dose), len(arr))
                for i in range(n):
                    if np.isfinite(dose[i]) and np.isfinite(arr[i]) and dose[i] > 0:
                        xs.append(np.log10(dose[i]))
                        ys.append(arr[i])
            if len(xs) < 5:
                QMessageBox.warning(self, "F-test 불가",
                    "log-dose 유효 쌍이 5개 이상 필요합니다.")
                return
            x_arr = np.array(xs); y_arr = np.array(ys)
            fit4 = cm.Hill4PL().fit(x_arr, y_arr)
            fit5 = cm.Hill5PL().fit(x_arr, y_arr)
            if not (fit4.converged and fit5.converged):
                QMessageBox.warning(self, "수렴 실패",
                    "두 모델 중 하나가 수렴하지 않았습니다.")
                return
            ft = cm.f_test_nested(fit4, fit5)
            lines = [
                "[ F-test: Hill 4PL (단순) vs Hill 5PL (복합) ]",
                "-" * 60,
                f"  Hill 4PL: R² = {_fmt_float(fit4.r_squared)}, AICc = {_fmt_float(fit4.aicc, digits=3)}",
                f"  Hill 5PL: R² = {_fmt_float(fit5.r_squared)}, AICc = {_fmt_float(fit5.aicc, digits=3)}",
                "",
                f"  F = {_fmt_float(ft.get('F'))}",
                f"  df_num = {ft.get('df_num')}, df_den = {ft.get('df_den')}",
                f"  p = {_fmt_float(ft.get('p'))}",
                f"  권장 모델: {ft.get('recommend', '판정 불가')}",
            ]
            self.stats_text.setPlainText('\n'.join(lines))
            res = stx.AnalysisResult(
                analysis_name='F-test: Hill 4PL vs Hill 5PL',
                summary={k: ft.get(k) for k in ('F', 'df_num', 'df_den', 'p', 'recommend')},
                notes=['p < 0.05 이면 복합 모델(5PL)이 유의하게 더 적합.'],
            )
            self.collected_analyses.append(res)
            return

    def _format_fit_result_kor(self, fit, parsed):
        """FitResult + DOSE_RESPONSE parsed → 한국어 요약."""
        if fit is None:
            return "피팅 결과 없음."
        lines = []
        lines.append("=" * 72)
        lines.append(f"  용량-반응 피팅: {fit.model_name}")
        lines.append("=" * 72)
        if not fit.converged:
            lines.append(f"피팅 실패: {fit.message}")
            return '\n'.join(lines)
        lines.append(f"데이터 포인트 n = {fit.n}")
        lines.append(f"R² = {_fmt_float(fit.r_squared, 4)}"
                     f"  (adj. R² = {_fmt_float(fit.adjusted_r2, 4)})")
        lines.append(f"RMSE = {_fmt_float(fit.rmse)}  "
                     f"AICc = {_fmt_float(fit.aicc, 3)}  "
                     f"BIC = {_fmt_float(fit.bic, 3)}")
        lines.append("")
        lines.append("[ 파라미터 ± SE (95% CI) ]")
        for k in (fit._param_order or list(fit.params.keys())):
            val = fit.params.get(k, float('nan'))
            se = fit.param_se.get(k, float('nan'))
            ci = fit.param_ci_95.get(k, (float('nan'), float('nan')))
            lines.append(
                f"  {k:<12} = {_fmt_float(val)} ± {_fmt_float(se)}  "
                f"[{_fmt_float(ci[0])}, {_fmt_float(ci[1])}]")
        if fit.derived:
            lines.append("")
            lines.append("[ 파생 수치 ]")
            for k, v in fit.derived.items():
                if isinstance(v, tuple) and len(v) == 2:
                    lines.append(f"  {k:<18} = [{_fmt_float(v[0])}, {_fmt_float(v[1])}]")
                else:
                    lines.append(f"  {k:<18} = {_fmt_float(v)}")
        if fit.message:
            lines.append("")
            lines.append(f"메시지: {fit.message}")
        return '\n'.join(lines)

    # =====================================================================
    # 기존 통계 요약 (기술통계 / 쌍별검정)
    # =====================================================================
    def _update_stats(self, df, include_pairwise=False, test='t-test', correction='bonferroni'):
        parts = ["[기술통계]", stx.format_describe_text(stx.describe_frame(df))]

        if not stx.HAS_SCIPY:
            parts.append("\n※ scipy 미설치로 통계 검정 생략. 'pip install scipy'로 활성화 가능.")
            self.stats_text.setPlainText('\n'.join(parts))
            return

        # 2개 이상 열이 있어야 ANOVA 의미 있음
        if df.shape[1] >= 2:
            long_df = df.melt(var_name='group', value_name='value').dropna()
            n_groups = long_df['group'].nunique()
            if n_groups >= 2:
                aov = stx.one_way_anova_long(long_df, 'value', 'group')
                if aov is not None:
                    parts.append("")
                    parts.append(aov.to_text())

        # 유의성 브래킷용 쌍별 검정 (차트에 표시된 것과 동일 통계)
        if include_pairwise:
            res = stx.pairwise_tests(df, test=test, correction=correction)
            if res:
                parts.append(f"\n[쌍별 검정: {test}, 보정: {correction}]")
                parts.append(f"{'A':<20}{'B':<20}{'p':>12}{'p_adj':>12}{'sig':>6}")
                for a, b, p, p_adj, stars in res:
                    p_s = _fmt_float(p) if not np.isnan(p) else "—"
                    pa_s = _fmt_float(p_adj) if not np.isnan(p_adj) else "—"
                    parts.append(
                        f"{str(a)[:19]:<20}{str(b)[:19]:<20}"
                        f"{p_s:>12}{pa_s:>12}{stars:>6}")

        self.stats_text.setPlainText('\n'.join(parts))

    def _update_scatter_stats(self, df):
        parts = ["[선형회귀 결과]"]
        x_col = df.columns[0]
        x = df[x_col].values.astype(float)
        for y_col in df.columns[1:]:
            y = df[y_col].values.astype(float)
            reg = stx.linear_regression(x, y)
            if reg is None:
                parts.append(f"{y_col}: 데이터 부족")
                continue
            p_s = f"{reg['p']:.4g}" if not np.isnan(reg['p']) else "—"
            parts.append(
                f"{y_col}:  y = {reg['slope']:.4g}·x + {reg['intercept']:.4g}   "
                f"R² = {reg['r2']:.4f}   p = {p_s}   n = {reg['n']}"
            )
        self.stats_text.setPlainText('\n'.join(parts))

    # =====================================================================
    # 프로젝트 저장/불러오기
    # =====================================================================
    def _global_config_snapshot(self):
        return {
            'preset': self.preset_combo.currentText(),
            'width_mm': float(self.width_mm.value()),
            'height_mm': float(self.height_mm.value()),
            'dpi': self.dpi_combo.currentText(),
            'font': self.font_combo.currentText(),
            'export_format_label': self.fmt_combo.currentText(),
            'transparent': bool(self.transparent_check.isChecked()),
        }

    def _apply_global_config_snapshot(self, cfg):
        if not cfg:
            return
        try:
            if cfg.get('preset'):
                self.preset_combo.setCurrentText(cfg['preset'])
            if cfg.get('width_mm') is not None:
                self.width_mm.setValue(float(cfg['width_mm']))
            if cfg.get('height_mm') is not None:
                self.height_mm.setValue(float(cfg['height_mm']))
            if cfg.get('dpi'):
                self.dpi_combo.setCurrentText(str(cfg['dpi']))
            if cfg.get('font'):
                self.font_combo.setCurrentText(cfg['font'])
            if cfg.get('export_format_label'):
                self.fmt_combo.setCurrentText(cfg['export_format_label'])
            if cfg.get('transparent') is not None:
                self.transparent_check.setChecked(bool(cfg['transparent']))
        except Exception:
            pass

    def _collect_chart_config(self):
        """9개 차트 탭의 모든 위젯 상태 + 활성 탭 + 사용자 색상 → JSON-safe dict."""
        widgets = {}
        for attr in dir(self):
            if not any(attr.startswith(p + '_') for p in CHART_WIDGET_PREFIXES):
                continue
            try:
                w = getattr(self, attr)
            except AttributeError:
                continue
            ser = _serialize_widget(w)
            if ser is not None:
                widgets[attr] = ser

        # 활성 탭 키 추출
        active_key = None
        idx = self.tab_widget.currentIndex()
        for k, i in self._tab_index_by_key.items():
            if i == idx:
                active_key = k
                break

        return {
            'widgets': widgets,
            'active_tab': active_key,
            'dose_response_color': self.dose_response_color,
        }

    def _apply_chart_config(self, cfg):
        """_collect_chart_config 로 저장한 dict를 UI에 복원."""
        if not cfg:
            return
        widgets = cfg.get('widgets') or {}
        for attr, d in widgets.items():
            if hasattr(self, attr):
                _apply_widget_value(getattr(self, attr), d)

        # 용량-반응 단색
        color = cfg.get('dose_response_color')
        if color:
            self.dose_response_color = color
            if hasattr(self, 'dr_color_btn'):
                self.dr_color_btn.setStyleSheet(
                    f"background-color: {color}; color: white;")

        # 활성 차트 탭
        active_key = cfg.get('active_tab')
        if active_key and active_key in self._tab_index_by_key:
            self.tab_widget.setCurrentIndex(self._tab_index_by_key[active_key])

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장", "project.gpj", "Graph Project (*.gpj)")
        if not path:
            return
        try:
            df = self.get_table_data_raw()
            df_save = df.reset_index().rename(columns={'index': 'row_name'})
            png_bytes = None
            if self.current_fig is not None:
                with BytesIO() as buf:
                    self.current_fig.savefig(buf, format='png', dpi=120,
                                             bbox_inches='tight', facecolor='white')
                    png_bytes = buf.getvalue()

            project = prj.ProjectFile(
                title=os.path.splitext(os.path.basename(path))[0],
                authors='',
                notes='',
                table_type=self.current_table_type.value,
                data=df_save,
                chart_type=self.current_graph_type or 'heatmap',
                chart_config=self._collect_chart_config(),
                global_config=self._global_config_snapshot(),
                analyses=list(self.collected_analyses),
                fits=list(self.collected_fits),
                preview_png_bytes=png_bytes,
            )
            project.save(path)
            self.dirty.mark_clean()
            QMessageBox.information(self, "저장 완료", f"저장되었습니다:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 불러오기", "", "Graph Project (*.gpj);;All Files (*)")
        if not path:
            return
        try:
            project = prj.ProjectFile.load(path)
        except Exception as e:
            QMessageBox.critical(self, "불러오기 실패", str(e))
            return

        # 테이블 유형 복원
        try:
            t = dm.DataTableType(project.table_type)
        except Exception:
            t = dm.DataTableType.COLUMN
        self.current_table_type = t
        self.table_type_combo.setCurrentText(TABLE_TYPE_LABELS[t])

        # 전역 설정
        self._apply_global_config_snapshot(project.global_config or {})

        # 데이터 로드
        if project.data is not None and len(project.data):
            df = project.data.copy()
            # 첫 열이 row_name이면 index로 사용
            if df.columns[0] in ('row_name', 'index'):
                df = df.set_index(df.columns[0])
            self._load_dataframe(df)

        # 차트 탭 위젯 상태 복원 (새 Phase 5.1)
        self._apply_chart_config(project.chart_config or {})

        # 누적 분석 / 피팅 결과 복원 (PDF 리포트에 포함용)
        self.collected_analyses = list(project.analyses) if project.analyses else []
        self.collected_fits = list(project.fits) if project.fits else []
        self.current_fit_result = None  # FitResult는 재피팅 필요

        self.dirty.mark_clean()
        summary = [f"프로젝트를 불러왔습니다:\n{path}"]
        if self.collected_analyses:
            summary.append(f"• 분석 기록 {len(self.collected_analyses)}건 복원")
        if self.collected_fits:
            summary.append(f"• 곡선 피팅 {len(self.collected_fits)}건 복원 "
                           f"(다시 그리려면 '피팅 + 그리기'를 재실행)")
        QMessageBox.information(self, "불러오기 완료", '\n'.join(summary))

    # =====================================================================
    # PDF 리포트
    # =====================================================================
    def generate_pdf_report(self):
        if self.current_fig is None and not self.collected_analyses and not self.collected_fits:
            QMessageBox.warning(self, "데이터 없음",
                "리포트에 포함할 그래프나 분석 결과가 없습니다.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "PDF 리포트 저장", "report.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        title, ok = QInputDialog.getText(self, "리포트 제목", "제목:",
                                         text=self.current_graph_type or "Analysis Report")
        if not ok:
            title = 'Analysis Report'
        authors, _ = QInputDialog.getText(self, "작성자(선택)", "작성자:")

        try:
            # 현재 figure를 PNG bytes로 (일회성 - 원본 figure는 보존)
            figs = []
            if self.current_fig is not None:
                with BytesIO() as buf:
                    self.current_fig.savefig(buf, format='png', dpi=200,
                                             bbox_inches='tight', facecolor='white')
                    figs.append(buf.getvalue())

            df_for_pdf = None
            try:
                df_for_pdf = self.get_table_data_raw()
            except Exception:
                pass

            prj.generate_report_pdf(
                fig_paths_or_buffers=figs,
                analyses=self.collected_analyses,
                fits=self.collected_fits,
                title=title,
                authors=authors or '',
                data_df=df_for_pdf,
                output_path=path,
            )
            QMessageBox.information(self, "완료", f"PDF 리포트 저장:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "PDF 생성 실패", str(e))

    # =====================================================================
    # Dirty / 제목표시
    # =====================================================================
    def _on_dirty_changed(self, is_dirty):
        title = "Scientific Graph Generator"
        if is_dirty:
            title = "* " + title
        self.setWindowTitle(title)

    # =====================================================================
    # 미리보기 / 이미지 저장
    # =====================================================================
    def _show_preview(self, fig):
        with BytesIO() as buf:
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                        facecolor='white' if not self.transparent_check.isChecked() else 'none')
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
        scaled = pixmap.scaled(self.preview_label.size(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def save_current_graph(self):
        if self.current_fig is None:
            QMessageBox.warning(self, "경고", "먼저 그래프를 생성해주세요.")
            return
        label = self.fmt_combo.currentText()
        ext = next(e for l, e in presets.EXPORT_FORMATS if l == label)

        default_name = f"{self.current_graph_type or 'graph'}.{ext}"
        filt_map = {
            'png': "PNG (*.png)", 'svg': "SVG (*.svg)", 'pdf': "PDF (*.pdf)",
            'eps': "EPS (*.eps)", 'tiff': "TIFF (*.tiff)", 'jpg': "JPEG (*.jpg)",
        }
        file_path, _ = QFileDialog.getSaveFileName(
            self, "이미지 저장", default_name, filt_map.get(ext, "All Files (*)"))
        if not file_path:
            return
        try:
            dpi = int(self.dpi_combo.currentText())
            transparent = self.transparent_check.isChecked()
            save_kwargs = dict(dpi=dpi, bbox_inches='tight',
                               facecolor='none' if transparent else 'white',
                               transparent=transparent)
            if ext == 'tiff':
                save_kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
            self.current_fig.savefig(file_path, **save_kwargs)
            QMessageBox.information(self, "성공", f"저장 완료:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 실패:\n{e}")

    def closeEvent(self, event):
        if self.current_fig is not None:
            plt.close(self.current_fig)
            self.current_fig = None
        super().closeEvent(event)


QT_MATERIAL_THEMES = [
    '기본 (Fusion)',
    'light_blue.xml', 'light_cyan.xml', 'light_cyan_500.xml',
    'light_lightgreen.xml', 'light_pink.xml', 'light_purple.xml',
    'light_red.xml', 'light_teal.xml', 'light_yellow.xml',
    'dark_amber.xml', 'dark_blue.xml', 'dark_cyan.xml',
    'dark_lightgreen.xml', 'dark_medical.xml', 'dark_pink.xml',
    'dark_purple.xml', 'dark_red.xml', 'dark_teal.xml', 'dark_yellow.xml',
]


def apply_theme(app, theme_name):
    """테마 적용. '기본 (Fusion)'이면 Fusion 스타일로 복귀."""
    if theme_name == '기본 (Fusion)':
        app.setStyle('Fusion')
        app.setStyleSheet('')
        return
    try:
        from qt_material import apply_stylesheet
        apply_stylesheet(app, theme=theme_name)
    except ImportError:
        app.setStyle('Fusion')


def main():
    app = QApplication(sys.argv)
    # 기본 테마는 환경변수 GRAPHGEN_THEME로 덮어쓸 수 있음
    default_theme = os.environ.get('GRAPHGEN_THEME', '기본 (Fusion)')
    apply_theme(app, default_theme)
    window = GraphGenerator()
    window._app_ref = app  # 테마 변경 시 사용
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""논문용 프리셋: 저널 사이즈, 색상 팔레트, 폰트 패밀리"""

JOURNAL_PRESETS = {
    '사용자 지정': None,
    'Nature 1-column (89mm)': {'width_mm': 89, 'height_mm': 70, 'font_size': 7, 'dpi': 600},
    'Nature 2-column (183mm)': {'width_mm': 183, 'height_mm': 120, 'font_size': 8, 'dpi': 600},
    'Science 1-column (55mm)': {'width_mm': 55, 'height_mm': 55, 'font_size': 7, 'dpi': 600},
    'Science 2-column (120mm)': {'width_mm': 120, 'height_mm': 90, 'font_size': 8, 'dpi': 600},
    'Cell 1-column (85mm)': {'width_mm': 85, 'height_mm': 70, 'font_size': 8, 'dpi': 600},
    'Cell 2-column (174mm)': {'width_mm': 174, 'height_mm': 120, 'font_size': 9, 'dpi': 600},
    'PLoS 1-column (83mm)': {'width_mm': 83, 'height_mm': 70, 'font_size': 8, 'dpi': 300},
    'PLoS 2-column (173mm)': {'width_mm': 173, 'height_mm': 120, 'font_size': 9, 'dpi': 300},
    'Slide 16:9': {'width_mm': 254, 'height_mm': 143, 'font_size': 14, 'dpi': 300},
    'Poster A0 panel': {'width_mm': 300, 'height_mm': 200, 'font_size': 18, 'dpi': 300},
}

SEQUENTIAL_PALETTES = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys',
    'YlGnBu', 'YlOrRd', 'YlOrBr', 'BuPu', 'BuGn',
]

DIVERGING_PALETTES = [
    'RdYlGn', 'RdYlBu', 'RdBu', 'coolwarm', 'bwr', 'seismic',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'Spectral',
]

QUALITATIVE_PALETTES = [
    'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2',
    'Paired', 'Dark2', 'Accent', 'tab10', 'tab20',
    'husl', 'hls', 'colorblind', 'deep', 'muted',
]

ALL_CMAPS = SEQUENTIAL_PALETTES + DIVERGING_PALETTES

FONT_FAMILIES = [
    'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans',
    'Calibri', 'Segoe UI', 'Tahoma', 'Verdana',
    'Times New Roman', 'Liberation Serif', 'DejaVu Serif',
    'Malgun Gothic', 'Nanum Gothic',
]

EXPORT_FORMATS = [
    ('PNG (래스터, 투명)', 'png'),
    ('SVG (벡터, 편집 가능)', 'svg'),
    ('PDF (벡터, 인쇄)', 'pdf'),
    ('EPS (벡터, 논문 제출)', 'eps'),
    ('TIFF (래스터, LZW 압축)', 'tiff'),
    ('JPEG (래스터, 프리뷰)', 'jpg'),
]

DPI_OPTIONS = [72, 96, 150, 300, 600, 1200]


def mm_to_inch(mm):
    return mm / 25.4


def apply_journal_preset(preset_name):
    """저널 프리셋을 dict로 반환 (None이면 사용자 지정)"""
    return JOURNAL_PRESETS.get(preset_name)

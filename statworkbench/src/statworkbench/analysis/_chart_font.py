"""matplotlib 한글 폰트 설정 공용 헬퍼.

차트를 직접 생성(savefig)하는 분석 모듈(mixed_anova, pca, two_way_anova 등)은
visualization 엔진을 거치지 않으므로 한글 폰트가 설정되지 않은 채 그려질 수 있다.
이 경우 DejaVu Sans로 폴백되어 한글 라벨이 □(tofu)로 깨진다.
각 모듈은 차트 생성 직전 ``ensure_korean_font()``를 호출해 한글 폰트를 보장한다.
"""

from __future__ import annotations

import platform


def ensure_korean_font() -> None:
    """matplotlib 전역 rcParams에 한글 폰트를 설정한다.

    맑은 고딕(Malgun Gothic)을 우선 적용하고, 없으면 나눔/애플 고딕 순으로 탐색한다.
    어느 것도 없으면 Windows 폰트 경로를 직접 등록하고, 최후에는 DejaVu Sans로 폴백한다.
    음수 기호가 깨지지 않도록 ``axes.unicode_minus``도 함께 비활성화한다.
    """
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic"]
    available = {f.name for f in fm.fontManager.ttflist}

    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:  # pragma: no cover - 플랫폼별 폴백 경로
        if platform.system() == "Windows":
            import os

            win_font = "C:/Windows/Fonts/malgun.ttf"
            if os.path.exists(win_font):
                fm.fontManager.addfont(win_font)
                plt.rcParams["font.family"] = fm.FontProperties(fname=win_font).get_name()
            else:
                plt.rcParams["font.family"] = "DejaVu Sans"
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False

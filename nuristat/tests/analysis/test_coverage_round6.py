"""커버리지 6라운드 — 최종 도달 가능 라인.

대상:
  analysis/python_bridge.py 136-137 : plt.savefig → 그림 파일 수집 경로

나머지 미커버 라인 최종 dead code 확인:
  validation.py 90    : sanitize 후 항상 패턴 매칭 → 도달 불가
  validation.py 120   : 모든 MeasureType이 _MEASURE_STORAGE_COMPAT에 등록 → 도달 불가
  csv_reader.py 128   : text.strip() 비어있으면 line 124에서 이미 중단 → 도달 불가
  csv_reader.py 137-138: csv.Error는 일반 텍스트에서 발생 안 함 → 도달 불가
  ml_engine.py 37-38/98-99/166-167: sklearn 설치됨 → ImportError 분기 도달 불가
  nonparametric.py 50: k>=2이면 ss_total>0 수학적으로 보장 → 도달 불가
  nonparametric.py 244: wilcoxon n=0이면 scipy가 먼저 예외 → 도달 불가
  assumptions.py 163  : 6개 MissingPolicy 모두 커버 → else 도달 불가
  survival_analysis 269/281-282/373: 수학적으로 도달 불가
"""

from __future__ import annotations

from nuristat.analysis.python_bridge import PythonBridge


# ---------------------------------------------------------------------------
# python_bridge.py 136-137: plt.savefig → 그림 파일 경로 수집
# ---------------------------------------------------------------------------

class TestPythonBridgePlotCollection:

    def test_plot_file_detected_after_savefig(self):
        """plt.savefig(_plot_dir + '/f.png') → lines 136-137 실행, plots 목록에 추가."""
        bridge = PythonBridge()

        script = (
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 2, 3])\n"
            "plt.savefig(_plot_dir + '/test.png')\n"
            "plt.close()\n"
        )

        result = bridge.execute(script)

        assert result["success"] is True
        assert len(result["plots"]) >= 1
        assert any(p.endswith(".png") for p in result["plots"])

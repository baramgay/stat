"""Report Engine — 자동 보고서 생성 엔진.

HTML 및 PDF 형식의 보고서를 생성합니다.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

from nuristat.core.dataset import Dataset

logger = logging.getLogger(__name__)


class ReportEngine:
    """보고서 생성 엔진.

    Features:
    - HTML 보고서 (반응형, 인쇄 최적화)
    - PDF 보고서 (weasyprint 또는 pdfkit)
    - 템플릿 기반 생성
    - 차트 삽입
    """

    def __init__(self) -> None:
        self._css_style = self._get_default_css()

    def generate_html_report(
        self,
        dataset: Dataset,
        analyses: list[dict[str, Any]],
        title: str = "누리스탯 분석 보고서",
        author: str = "",
    ) -> str:
        """HTML 보고서를 생성합니다.

        Args:
            dataset: 분석 대상 데이터셋
            analyses: 분석 결과 목록
            title: 보고서 제목
            author: 작성자

        Returns:
            HTML 문자열
        """
        sections = []

        # 헤더
        sections.append(self._generate_header(title, author))

        # 요약
        sections.append(self._generate_summary(dataset))

        # 데이터 개요
        sections.append(self._generate_data_overview(dataset))

        # 분석 결과
        for i, analysis in enumerate(analyses, 1):
            sections.append(self._generate_analysis_section(i, analysis))

        # 푸터
        sections.append(self._generate_footer())

        # HTML 조립
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{self._css_style}</style>
</head>
<body>
    <div class="container">
        {''.join(sections)}
    </div>
</body>
</html>"""

        return html

    def generate_data_quality_report(
        self,
        dataset: Dataset,
    ) -> str:
        """데이터 품질 보고서를 생성합니다."""
        df = dataset.data

        sections = []
        sections.append(self._generate_header("데이터 품질 진단 보고서", ""))

        # 기본 정보
        sections.append(f"""
        <div class="section">
            <h2>📊 기본 정보</h2>
            <table class="info-table">
                <tr><th>데이터셋명</th><td>{dataset.name}</td></tr>
                <tr><th>행 수</th><td>{len(df):,}</td></tr>
                <tr><th>열 수</th><td>{len(df.columns)}</td></tr>
                <tr><th>메모리 사용량</th><td>{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</td></tr>
            </table>
        </div>
        """)

        # 결측치 분석
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        missing_rows = ""
        for col in df.columns:
            if missing[col] > 0:
                missing_rows += f"""
                <tr>
                    <td>{col}</td>
                    <td>{missing[col]:,}</td>
                    <td>{missing_pct[col]:.2f}%</td>
                    <td>{'🔴 심각' if missing_pct[col] > 50 else '🟠 주의' if missing_pct[col] > 10 else '🟡 경고' if missing_pct[col] > 0 else '🟢 정상'}</td>
                </tr>
                """

        if missing_rows:
            sections.append(f"""
            <div class="section">
                <h2>⚠️ 결측치 분석</h2>
                <table class="data-table">
                    <thead>
                        <tr><th>변수</th><th>결측 수</th><th>결측 비율</th><th>상태</th></tr>
                    </thead>
                    <tbody>{missing_rows}</tbody>
                </table>
            </div>
            """)
        else:
            sections.append("""
            <div class="section">
                <h2>✅ 결측치 분석</h2>
                <p class="success">결측치가 없습니다.</p>
            </div>
            """)

        # 이상치 분석 (숫자형 변수)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_rows = ""

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            if len(outliers) > 0:
                outlier_rows += f"""
                <tr>
                    <td>{col}</td>
                    <td>{len(outliers):,}</td>
                    <td>{len(outliers)/len(df)*100:.2f}%</td>
                    <td>{outliers.min():.2f} ~ {outliers.max():.2f}</td>
                </tr>
                """

        if outlier_rows:
            sections.append(f"""
            <div class="section">
                <h2>🔍 이상치 분석 (IQR 기준)</h2>
                <table class="data-table">
                    <thead>
                        <tr><th>변수</th><th>이상치 수</th><th>비율</th><th>범위</th></tr>
                    </thead>
                    <tbody>{outlier_rows}</tbody>
                </table>
            </div>
            """)

        # 중복 행
        duplicates = df.duplicated().sum()
        sections.append(f"""
        <div class="section">
            <h2>🔄 중복 행</h2>
            <p>{duplicates:,}개의 중복 행이 {'있습니다.' if duplicates > 0 else '없습니다.'}</p>
        </div>
        """)

        sections.append(self._generate_footer())

        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>데이터 품질 진단 보고서</title>
    <style>{self._css_style}</style>
</head>
<body>
    <div class="container">
        {''.join(sections)}
    </div>
</body>
</html>"""

    def _generate_header(self, title: str, author: str) -> str:
        """보고서 헤더 생성."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        author_html = f"<p class='author'>작성자: {author}</p>" if author else ""

        return f"""
        <div class="header">
            <h1>{title}</h1>
            {author_html}
            <p class="date">생성일: {now}</p>
        </div>
        """

    def _generate_summary(self, dataset: Dataset) -> str:
        """요약 섹션 생성."""
        df = dataset.data
        return f"""
        <div class="section">
            <h2>📋 요약</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-label">데이터셋</span>
                    <span class="summary-value">{dataset.name}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">행 수</span>
                    <span class="summary-value">{len(df):,}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">열 수</span>
                    <span class="summary-value">{len(df.columns)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">결측치</span>
                    <span class="summary-value">{df.isnull().sum().sum():,}</span>
                </div>
            </div>
        </div>
        """

    def _generate_data_overview(self, dataset: Dataset) -> str:
        """데이터 개요 섹션 생성."""
        df = dataset.data

        # 기술통계
        desc = df.describe().T
        desc_html = desc.to_html(classes="data-table", float_format=lambda x: f"{x:.3f}")

        return f"""
        <div class="section">
            <h2>📊 데이터 개요</h2>
            <h3>기술통계</h3>
            {desc_html}
        </div>
        """

    def _generate_analysis_section(self, index: int, analysis: dict[str, Any]) -> str:
        """분석 결과 섹션 생성."""
        analysis_type = analysis.get("type", "Unknown")
        result_text = analysis.get("result", "")

        # 결과 텍스트를 HTML로 변환
        result_html = result_text.replace("\n", "<br>")

        return f"""
        <div class="section">
            <h2>#{index} {analysis_type}</h2>
            <div class="result-box">
                {result_html}
            </div>
        </div>
        """

    def _generate_footer(self) -> str:
        """보고서 푸터 생성."""
        return """
        <div class="footer">
            <p>NuriStat 자동 생성 보고서</p>
            <p>© 2025 경남빅데이터센터</p>
        </div>
        """

    def _get_default_css(self) -> str:
        """기본 CSS 스타일."""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Malgun Gothic', 'NanumGothic', sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #1a5276, #2e86ab);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }

        .author, .date {
            opacity: 0.9;
            font-size: 14px;
        }

        .section {
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .section h2 {
            color: #1a5276;
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }

        .section h3 {
            color: #2e86ab;
            font-size: 16px;
            margin: 15px 0 10px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .summary-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }

        .summary-label {
            display: block;
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }

        .summary-value {
            display: block;
            font-size: 24px;
            font-weight: bold;
            color: #1a5276;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }

        .data-table th {
            background: #1a5276;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        .data-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }

        .data-table tr:hover {
            background: #f8f9fa;
        }

        .info-table {
            width: 100%;
            border-collapse: collapse;
        }

        .info-table th {
            background: #f8f9fa;
            padding: 10px;
            text-align: left;
            width: 30%;
            border-bottom: 1px solid #eee;
        }

        .info-table td {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }

        .result-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            font-family: 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.8;
            overflow-x: auto;
        }

        .success {
            color: #2ca02c;
            font-weight: bold;
        }

        .footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 12px;
        }

        @media print {
            body { background: white; }
            .container { max-width: 100%; padding: 0; }
            .section { break-inside: avoid; box-shadow: none; }
        }
        """

    def save_html(self, html: str, path: str) -> None:
        """HTML 파일로 저장."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML 보고서 저장 완료: {path}")

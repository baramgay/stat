"""AnalysisPlugin Protocol 테스트.

검증 항목:
- AnalysisPlugin은 runtime_checkable Protocol
- 필수 속성(id, name, category, description, variable_requirements) 충족 시 isinstance 통과
- 속성 누락 시 isinstance 실패
- 실제 분석 엔진(TtestEngine, DescriptiveEngine 등)이 Protocol을 만족하는지 확인
"""

from __future__ import annotations

import pytest

from statworkbench.analysis.base import AnalysisPlugin


# ──────────────────────────────────────────────────────────────
# 프로토콜 준수 구현체
# ──────────────────────────────────────────────────────────────

class _ConformingPlugin:
    """AnalysisPlugin을 완전히 구현한 더미 클래스."""

    id = "test_plugin"
    name = "테스트 플러그인"
    category = "Test"
    description = "테스트용 분석 플러그인"
    variable_requirements = [{"role": "dependent", "required": True}]
    implemented = True

    def validate(self, dataset, spec):
        return []

    def run(self, dataset, spec):
        from statworkbench.analysis.result import AnalysisResult
        return AnalysisResult(title="test", tables=[])


class _MissingMethodPlugin:
    """run 메서드가 없는 불완전 구현체."""

    id = "incomplete"
    name = "미완성"
    category = "Test"
    description = "run 없음"
    variable_requirements = []

    def validate(self, dataset, spec):
        return []


class _NoAttributePlugin:
    """필수 속성이 없는 더미 클래스."""

    def validate(self, dataset, spec):
        return []

    def run(self, dataset, spec):
        pass


# ──────────────────────────────────────────────────────────────
# 1. Protocol 기본 검증
# ──────────────────────────────────────────────────────────────

class TestAnalysisPluginProtocol:

    def test_conforming_class_is_instance(self):
        plugin = _ConformingPlugin()
        assert isinstance(plugin, AnalysisPlugin)

    def test_missing_run_is_not_instance(self):
        plugin = _MissingMethodPlugin()
        assert not isinstance(plugin, AnalysisPlugin)

    def test_no_attributes_is_not_instance(self):
        plugin = _NoAttributePlugin()
        assert not isinstance(plugin, AnalysisPlugin)

    def test_protocol_is_runtime_checkable(self):
        """isinstance 호출 자체가 TypeError 없이 실행되어야 한다."""
        try:
            _ = isinstance(object(), AnalysisPlugin)
        except TypeError:
            pytest.fail("AnalysisPlugin은 runtime_checkable이어야 합니다")


# ──────────────────────────────────────────────────────────────
# 2. 필수 속성 확인
# ──────────────────────────────────────────────────────────────

class TestAnalysisPluginAttributes:

    def test_id_attribute(self):
        plugin = _ConformingPlugin()
        assert hasattr(plugin, "id")
        assert isinstance(plugin.id, str)

    def test_name_attribute(self):
        plugin = _ConformingPlugin()
        assert hasattr(plugin, "name")
        assert isinstance(plugin.name, str)

    def test_category_attribute(self):
        plugin = _ConformingPlugin()
        assert hasattr(plugin, "category")
        assert isinstance(plugin.category, str)

    def test_description_attribute(self):
        plugin = _ConformingPlugin()
        assert hasattr(plugin, "description")

    def test_variable_requirements_is_list(self):
        plugin = _ConformingPlugin()
        assert isinstance(plugin.variable_requirements, list)

    def test_validate_returns_list(self):
        plugin = _ConformingPlugin()
        result = plugin.validate(None, {})
        assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────
# 3. 실제 분석 엔진이 Protocol 만족하는지 확인
# ──────────────────────────────────────────────────────────────

class TestRealEnginesConformToProtocol:

    def test_ttests_engine_conforms(self):
        try:
            from statworkbench.analysis.ttests import TtestEngine
            engine = TtestEngine()
            assert isinstance(engine, AnalysisPlugin)
        except (ImportError, AttributeError):
            pytest.skip("TtestEngine 없음")

    def test_descriptive_engine_conforms(self):
        try:
            from statworkbench.analysis.descriptive import DescriptiveEngine
            engine = DescriptiveEngine()
            assert isinstance(engine, AnalysisPlugin)
        except (ImportError, AttributeError):
            pytest.skip("DescriptiveEngine 없음")

    def test_anova_engine_conforms(self):
        try:
            from statworkbench.analysis.anova import AnovaEngine
            engine = AnovaEngine()
            assert isinstance(engine, AnalysisPlugin)
        except (ImportError, AttributeError):
            pytest.skip("AnovaEngine 없음")

    def test_correlation_engine_conforms(self):
        try:
            from statworkbench.analysis.correlation import CorrelationEngine
            engine = CorrelationEngine()
            assert isinstance(engine, AnalysisPlugin)
        except (ImportError, AttributeError):
            pytest.skip("CorrelationEngine 없음")

    def test_regression_engine_conforms(self):
        try:
            from statworkbench.analysis.regression import RegressionEngine
            engine = RegressionEngine()
            assert isinstance(engine, AnalysisPlugin)
        except (ImportError, AttributeError):
            pytest.skip("RegressionEngine 없음")

    def test_frequencies_engine_conforms(self):
        try:
            from statworkbench.analysis.frequencies import FrequenciesEngine
            engine = FrequenciesEngine()
            assert isinstance(engine, AnalysisPlugin)
        except (ImportError, AttributeError):
            pytest.skip("FrequenciesEngine 없음")

import pytest

from b2d_collector.experts.template import ExpertTemplate, get_entry_point


def test_expert_template_contract():
    assert get_entry_point() == "ExpertTemplate"
    expert = ExpertTemplate()
    expert.setup("")
    expert.set_global_plan([], [])
    assert expert.sensors() == []
    assert expert.get_expert_assessment() is None
    with pytest.raises(NotImplementedError):
        expert.run_step({}, 0.0)
    expert.destroy()

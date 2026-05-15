from project.experiments import main_experiment as exp
import drl_train


def test_revenue_model_returns_float() -> None:
    agent = exp.Agent(0, "High", 120, 60)
    task = exp.Task(0, real_workload=30, real_cpi=2.0)

    value = exp.calc_revenue(task, agent, concurrency=1, omega=task.wk / agent.capacity)

    assert isinstance(value, float)


def test_core_algorithms_return_expected_metrics() -> None:
    agents = exp.create_agents_from_real_data(5, real_distribution=None)
    tasks = exp.create_tasks_from_real_data(
        5,
        real_data={
            "workloads": [30, 40, 50, 60, 70],
            "cpis": [2.0, 2.0, 2.0, 2.0, 2.0],
        },
    )

    for alg in ["MCT", "CHASE", "IRS", "DyLAN", "BRTOA"]:
        fresh_agents = [exp.Agent(a.id, a.type, a.capacity, a.output) for a in agents]
        result = exp.run_algorithm(alg, fresh_agents, tasks)

        assert {"revenue", "success_rate", "time", "utilization"} <= set(result)


def test_agents_and_tasks_expose_skills_and_subtasks() -> None:
    agent = exp.Agent(0, "Mid", 70, 35)
    task = exp.Task(12, real_workload=65, real_cpi=2.0)

    assert agent.skills
    assert set(agent.skills) <= set(agent.skill_efficiency)
    assert exp.Agent(0, "Low", 25, 12.5).skills == ("sense", "actuate")
    assert exp.Agent(1, "Mid", 70, 35).skills == ("sense", "preprocess", "aggregate")
    assert exp.Agent(2, "High", 120, 60).skills == ("preprocess", "infer", "aggregate")
    assert task.subtasks
    assert task.task_type == "Control"
    assert task.required_skills == ("sense", "preprocess", "infer", "actuate")
    assert set(task.required_skills) == {subtask.skill for subtask in task.subtasks}
    assert abs(sum(subtask.wk for subtask in task.subtasks) - task.wk) < 1e-9


def test_group_algorithms_cover_skills_and_respect_capacity() -> None:
    base_agents = exp.create_agents_from_real_data(8, real_distribution=None)
    tasks = [
        exp.Task(0, real_workload=35, real_cpi=2.0),
        exp.Task(1, real_workload=55, real_cpi=2.0),
        exp.Task(2, real_workload=75, real_cpi=2.0),
    ]

    for alg in ["MCT", "CHASE", "IRS", "DyLAN", "BRTOA"]:
        agents = [exp.Agent(a.id, a.type, a.capacity, a.output) for a in base_agents]
        result = exp.run_algorithm(alg, agents, tasks)

        assert {"revenue", "success_rate", "time", "utilization", "concurrency"} <= set(result)
        for agent in agents:
            assert agent.current_load <= 1.0 + 1e-9
            task_ids = [record.task.id for record in agent.concurrent_tasks]
            assert len(task_ids) == len(set(task_ids))
            for record in agent.concurrent_tasks:
                assert agent.can_execute(record.subtask.skill)

        for task in tasks:
            records = exp.task_records_from_agents(agents, task)
            if records:
                assert exp.task_is_covered(task, records)


def test_mct_brtoa_and_dylan_do_not_collapse_to_same_assignment() -> None:
    base_agents = exp.create_agents_from_real_data(12, real_distribution=None)
    tasks = [exp.Task(i, real_workload=35 + i * 7, real_cpi=2.0) for i in range(8)]

    signatures = {}
    for alg in ["MCT", "BRTOA", "DyLAN"]:
        agents = [exp.Agent(a.id, a.type, a.capacity, a.output) for a in base_agents]
        exp.run_algorithm(alg, agents, tasks)
        signatures[alg] = tuple(
            sorted(
                (record.task.id, record.subtask.id, agent.id)
                for agent in agents
                for record in agent.concurrent_tasks
            )
        )

    assert len(set(signatures.values())) == len(signatures)


def test_baselines_do_not_call_chase_group_search(monkeypatch) -> None:
    class FakeFLDRLModel:
        def build_state(self, task, subtask, agents, progress=0.0):
            return [task.wk, subtask.wk, progress]

        def choose_action(self, state, feasible_mask):
            for idx, feasible in enumerate(feasible_mask):
                if feasible:
                    return idx
            return None

    def fail_chase_group(*args, **kwargs):
        raise AssertionError("baseline must not call CHASE group search")

    monkeypatch.setattr(exp, "tas_find_group", fail_chase_group)
    monkeypatch.setattr(exp, "_find_chase_group", fail_chase_group)
    monkeypatch.setattr(exp, "get_fl_drl_model", lambda: FakeFLDRLModel())

    base_agents = exp.create_agents_from_real_data(8, real_distribution=None)
    tasks = [
        exp.Task(0, real_workload=35, real_cpi=2.0),
        exp.Task(1, real_workload=45, real_cpi=2.0),
    ]

    for alg in ["MCT", "IRS", "DyLAN", "FL_DRL", "BRTOA"]:
        agents = [exp.Agent(a.id, a.type, a.capacity, a.output) for a in base_agents]
        result = exp.run_algorithm(alg, agents, tasks)

        assert {"revenue", "success_rate", "time", "utilization", "concurrency"} <= set(result)


def test_brtoa_uses_tier_actions_and_stops_without_updates() -> None:
    agents = [
        exp.Agent(0, "Low", 25, 12.5),
        exp.Agent(1, "Low", 25, 12.5),
        exp.Agent(2, "Mid", 70, 35),
        exp.Agent(3, "Mid", 70, 35),
        exp.Agent(4, "High", 120, 60),
        exp.Agent(5, "High", 120, 60),
    ]
    tasks = [exp.Task(0, real_workload=35, real_cpi=2.0)]
    solver = exp.Baseline_BRTOA(agents, tasks)
    solver.max_iterations = 10

    assignment = solver.run()

    assert set(solver.initial_actions.values()) == {0}
    assert set(solver.final_actions.values()) <= {0, 1, 2}
    assert solver._feasible_actions(tasks[0], {tasks[0].id: 0})
    assert solver.iterations < solver.max_iterations
    task_records = exp.task_records_from_assignment(assignment, tasks[0])
    assert exp.task_is_covered(tasks[0], task_records)
    for records in assignment.values():
        for record in records:
            assert record.agent.can_execute(record.subtask.skill)


def test_brtoa_preferred_tier_falls_back_for_multiskill_tasks(monkeypatch) -> None:
    def fail_chase_group(*args, **kwargs):
        raise AssertionError("BRTOA must not call CHASE group search")

    monkeypatch.setattr(exp, "tas_find_group", fail_chase_group)
    monkeypatch.setattr(exp, "_find_chase_group", fail_chase_group)

    agents = [
        exp.Agent(0, "Low", 25, 12.5),
        exp.Agent(1, "Low", 25, 12.5),
        exp.Agent(2, "Mid", 70, 35),
        exp.Agent(3, "Mid", 70, 35),
        exp.Agent(4, "High", 120, 60),
        exp.Agent(5, "High", 120, 60),
    ]
    tasks = [
        exp.Task(2, real_workload=45, real_cpi=2.0),  # Control
        exp.Task(3, real_workload=55, real_cpi=2.0),  # Fusion
    ]
    solver = exp.Baseline_BRTOA(agents, tasks)
    solver.max_iterations = 10

    assignment = solver.run()

    for task in tasks:
        records = exp.task_records_from_assignment(assignment, task)
        assert exp.task_is_covered(task, records)
        assert {record.subtask.skill for record in records} == set(task.required_skills)
        for record in records:
            assert record.agent.can_execute(record.subtask.skill)


def test_dylan_two_stage_surrogate_collects_graph_and_importance() -> None:
    agents = exp.create_agents_from_real_data(8, real_distribution=None)
    tasks = [
        exp.Task(0, real_workload=35, real_cpi=2.0),
        exp.Task(1, real_workload=45, real_cpi=2.0),
    ]
    solver = exp.DyLAN(agents, tasks, T_max=3, K_select=4, K_keep=4, consistency_theta=1.1)

    selected_agents, importance = solver.run_team_optimization()
    result = solver.run_task_solving()

    assert solver.preliminary is not None
    graph, messages, peer_scores, output = solver.preliminary
    assert graph[0]
    assert messages
    assert peer_scores
    assert output["assignment"] is not None
    assert len(selected_agents) == 4
    assert solver.selected_covers_required
    selected_skills = {
        skill
        for agent in solver.selected_agents
        for skill in agent.skills
    }
    required_skills = {
        subtask.skill
        for task in tasks
        for subtask in task.subtasks
    }
    assert required_skills <= selected_skills
    assert set(importance) == {agent.id for agent in agents}
    assert solver.final_output is not None
    assert result.keys() == {agent.id for agent in agents}

    early_stop_solver = exp.DyLAN(agents, tasks)
    should_stop, group = early_stop_solver._early_stopping(
        [
            {"signature": ("same",), "revenue": 1.0},
            {"signature": ("same",), "revenue": 2.0},
            {"signature": ("same",), "revenue": 3.0},
            {"signature": ("other",), "revenue": 4.0},
        ]
    )
    assert should_stop
    assert len(group) == 3


def test_dylan_uses_preliminary_output_when_selected_output_is_worse(monkeypatch) -> None:
    agents = [
        exp.Agent(0, "Low", 25, 12.5),
        exp.Agent(1, "Mid", 70, 35),
        exp.Agent(2, "High", 120, 60),
    ]
    tasks = [exp.Task(2, real_workload=45, real_cpi=2.0)]
    solver = exp.DyLAN(agents, tasks, T_max=2, K_select=3, K_keep=3, consistency_theta=1.1)

    preliminary_assignment = exp._empty_assignment(agents)
    selected_assignment = exp._empty_assignment(agents)

    def fake_dylan_inference(active_agents, collect_peer_score=False):
        nodes = [(1, agent.id) for agent in active_agents]
        if collect_peer_score:
            output = {
                "assignment": preliminary_assignment,
                "revenue": 100.0,
                "success_rate": 1.0,
                "signature": ("preliminary",),
            }
            messages = {
                node: {"revenue": 100.0, "signature": ("preliminary",)}
                for node in nodes
            }
            return ([nodes], []), messages, {}, output
        output = {
            "assignment": selected_assignment,
            "revenue": 1.0,
            "success_rate": 0.0,
            "signature": ("selected",),
        }
        messages = {
            node: {"revenue": 1.0, "signature": ("selected",)}
            for node in nodes
        }
        return ([nodes], []), messages, {}, output

    monkeypatch.setattr(solver, "_dylan_inference", fake_dylan_inference)

    solver.run_task_solving()

    assert solver.selected_covers_required
    assert solver.final_output["signature"] == ("preliminary",)


def test_chase_assignment_index_matches_uncached_revenue_and_delta() -> None:
    agents = exp.create_agents_from_real_data(8, real_distribution=None)
    tasks = [
        exp.Task(0, real_workload=35, real_cpi=2.0),
        exp.Task(1, real_workload=45, real_cpi=2.0),
        exp.Task(2, real_workload=55, real_cpi=2.0),
    ]
    assignment = exp._empty_assignment(agents)
    for task in tasks:
        group = exp._find_chase_group(task, agents, assignment)
        if group:
            exp.commit_group_assignment(assignment, group)

    current_index = exp._build_assignment_index(assignment)
    for task in tasks:
        assert exp._task_revenue_from_index(task, assignment, current_index) == exp._task_revenue_from_assignment(task, assignment)

    projected = exp.clone_assignment(assignment)
    removed_records = exp.remove_task_from_assignment(projected, tasks[0].id)
    projected_index = exp._build_assignment_index(projected)

    uncached_delta = exp._replacement_delta(assignment, projected, removed_records, [])
    cached_delta = exp._replacement_delta(
        assignment,
        projected,
        removed_records,
        [],
        current_index=current_index,
        projected_index=projected_index,
    )
    assert cached_delta == uncached_delta


def test_chase_value_replacement_is_optional() -> None:
    agents = exp.create_agents_from_real_data(8, real_distribution=None)
    tasks = [
        exp.Task(0, real_workload=35, real_cpi=2.0),
        exp.Task(1, real_workload=45, real_cpi=2.0),
        exp.Task(2, real_workload=55, real_cpi=2.0),
    ]

    default_agents = [exp.Agent(a.id, a.type, a.capacity, a.output) for a in agents]
    enhanced_agents = [exp.Agent(a.id, a.type, a.capacity, a.output) for a in agents]

    exp.run_CHASE_full(default_agents, tasks)
    exp.run_CHASE_ablation(enhanced_agents, tasks, use_value_replacement=True)

    for agent_group in (default_agents, enhanced_agents):
        for agent in agent_group:
            assert agent.current_load <= 1.0 + 1e-9
            for record in agent.concurrent_tasks:
                assert agent.can_execute(record.subtask.skill)


def test_fldrl_training_env_uses_subtask_skill_mask() -> None:
    env = drl_train.TaskSchedulingEnv(n_agents=6)
    env.agents = drl_train.create_agents(6)
    task = drl_train.SimpleTask(2, wk=75, cpi=2.0)
    env.tasks = [task]
    env.current_idx = 0
    env.current_subtask_idx = 1
    env.task_records = {}
    env.total_revenue = 0
    env.success_count = 0

    state = env._get_state()
    subtask = task.subtasks[1]
    mask = env.get_feasible_mask()

    assert len(state) == env.state_dim
    assert env.state_dim == 2 + len(drl_train.SKILL_NAMES) + 1 + 3 * env.n_agents
    for idx, agent in enumerate(env.agents):
        if not agent.can_execute(subtask.skill):
            assert not mask[idx]


def test_high_pressure_task_clone_scales_subtasks_and_deadline() -> None:
    task = exp.Task(7, real_workload=70, real_cpi=2.0)
    cloned = exp.clone_tasks_for_scenario([task], deadline=0.75, workload_scale=1.15)[0]

    assert cloned.deadline == 0.75
    assert abs(cloned.wk - task.wk * 1.15) < 1e-9
    for original, scaled in zip(task.subtasks, cloned.subtasks, strict=True):
        assert original.skill == scaled.skill
        assert abs(scaled.wk - original.wk * 1.15) < 1e-9

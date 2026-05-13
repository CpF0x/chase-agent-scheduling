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
    task = exp.Task(0, real_workload=65, real_cpi=2.0)

    assert agent.skills
    assert set(agent.skills) <= set(agent.skill_efficiency)
    assert task.subtasks
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

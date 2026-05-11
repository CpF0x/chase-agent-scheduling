from after_project.experiments import main_experiment as exp


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

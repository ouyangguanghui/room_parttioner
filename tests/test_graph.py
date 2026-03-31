"""RoomGraph 确定性回归测试。"""

from app.utils.graph import RoomGraph


def test_assign_colors_is_deterministic_when_fallback_needed():
    graph_builder = RoomGraph({})
    # K6 完全图会触发第 6 个节点的回退分配。
    graph = {i: [j for j in range(6) if j != i] for i in range(6)}

    first = graph_builder.assign_colors(graph)
    second = graph_builder.assign_colors(graph)
    assert first == second


def test_assign_color_for_room_fallback_is_deterministic():
    graph_builder = RoomGraph({})
    graph = {5: [0, 1, 2, 3, 4]}
    current_colors = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    c1 = graph_builder.assign_color_for_room(5, graph, current_colors)
    c2 = graph_builder.assign_color_for_room(5, graph, current_colors)
    assert c1 == c2
